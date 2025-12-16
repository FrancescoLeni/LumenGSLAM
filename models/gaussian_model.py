import torch
import torch.nn.functional as F
import numpy as np
from simple_knn._C import distCUDA2
from gaussian_norms import compute_gauss_norm
from functools import partial

from utils.data_process.preprocessing import get_pointcloud, get_depth_and_silhouette
from utils.data_process.outlier_handling import energy_mask
from utils.data_process.pose_handling import quat2rot, rotation_aligned_with_normal
from utils.data_process.sh_utils import RGB2SH, eval_sh
from utils.general import my_logger


# from utils.data_process.geometry import strip_lowerdiag


class GaussianModel:
    def __init__(self, device, config, save_dst):

        self.device = device
        self.data_config = config['gaussians']
        self.densification_config = config['densification']
        self.num_mappings_iter = config['mapping']['num_iter']
        self.do_pbr = config['do_pbr']
        self.cull_outside_depth = config['cull_outside_depth']

        self.do_energy_mask = config['use_energy_mask']

        self.lr_config = config['lr_scheduler']

        self.save_dst = save_dst

        # call .init_model to initialize
        self.params, self.variables = None, None

        self.active_sh_degree = self.data_config['max_sh_degree']

        self.hooks = None

        self.stored_params = {}

        self.just_reset_flag = False

    def init_model(self, first_color, first_depth, intrinsics, w2c=None):
        # scene_radius_depth_ratio should be in config

        # Mask out invalid depth values and bright
        mask = (first_depth > 0)

        if self.do_energy_mask:
            mask = mask & energy_mask(first_color)

        mask = mask.reshape(-1)

        init_pt_cld = self._get_first_pointcloud(first_color, first_depth, intrinsics, mask=mask, w2c=w2c)
        first_scales = self._init_scales(init_pt_cld, first_depth, intrinsics, mask=mask, mode=self.data_config['scale_init_mode'])
        self.params, self.variables = self._init_params_vars(init_pt_cld, first_scales)
        # 'scene_radius' used for pruning/densification
        self.variables['scene_radius'] = torch.max(first_depth)/self.data_config['scene_radius_depth_ratio']

        # setting lr scheduler
        self.set_grad_hooks()

    def add_new_gaussians(self, curr_data, intrinsics, depth_sil: tuple, frame_id, logger, sil_thres=0.9):
        """
        adds new gaussians if a new object appears in front of the previous ones and if silhouette is too low
        N.B. Doesn't update optimizer!

        args:
            -- curr_data: dict containing 'im', 'depth', 'time_idx', 'w2c' of current frame
            -- intrinsics: intrinsics camera parameters
            -- sil_thres: silhouette threshold
            -- depth_sil: rendered depth and silhouette
        """

        render_depth, silhouette, _  = depth_sil

        non_presence_sil_mask = (silhouette < sil_thres)
        gt_depth = curr_data['depth'][0, :, :]

        valid_depth_mask = (gt_depth > 0) & (gt_depth < 1e10)

        # checks if an object is appearing in front of the existing ones
        depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)

        non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > (20 * depth_error.mean()))

        non_presence_mask = non_presence_sil_mask | non_presence_depth_mask if not self.just_reset_flag else non_presence_depth_mask
        self.just_reset_flag = False
        non_presence_mask = non_presence_mask.reshape(-1)

        # Get the new frame Gaussians based on the Silhouette
        if torch.sum(non_presence_mask) > 0:

            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
            if self.do_energy_mask:
                valid_color_mask = energy_mask(curr_data['im']).squeeze()
                non_presence_mask = non_presence_mask & valid_color_mask.reshape(-1)
            else:
                valid_color_mask = torch.ones_like(non_presence_mask).to(self.device)

            logger.get_masks_growth((non_presence_sil_mask, non_presence_depth_mask, valid_depth_mask, valid_color_mask), frame_id)

            # backprojecting points and initilizing scales
            new_pt_cld = get_pointcloud(curr_data['im'], curr_data['depth'], intrinsics, curr_data['w2c'], mask=non_presence_mask)
            new_scales = self._init_scales(new_pt_cld, curr_data['depth'], intrinsics, mask=non_presence_mask, mode=self.data_config['scale_init_mode'])

            new_params = self._init_params_vars(new_pt_cld, new_scales, return_vars=False, adding=False)

            for k, v in new_params.items():
                self.params[k] = torch.nn.Parameter(torch.cat((self.params[k], v), dim=0).requires_grad_(True))

            num_pts = new_params['means3D'].shape[0]
            self.variables['means2D_gradient_accum'] = torch.cat((self.variables['max_2D_radius'], torch.zeros(num_pts, device=self.device).float()), dim=0)
            self.variables['denom'] = torch.cat((self.variables['max_2D_radius'], torch.zeros(num_pts, device=self.device).float()), dim=0)
            self.variables['max_2D_radius'] = torch.cat((self.variables['max_2D_radius'], torch.zeros(num_pts, device=self.device).float()), dim=0)
            self.variables['param_update_count'] = torch.cat((self.variables['param_update_count'], torch.zeros(num_pts, device=self.device).float()), dim=0)
            self.variables['param_update_count_opacity'] = torch.cat((self.variables['param_update_count_opacity'], torch.zeros(num_pts, device=self.device).float()), dim=0)


            # lr scheduler update
            self.reset_hooks()
            self.set_grad_hooks()

    def densify_prune(self, optimizer, mapping_iter, frame_id, w2c, camera, logger):
        def accumulate_mean2d_gradient(var):
            var['means2D_gradient_accum'][var['seen']] += torch.norm(var['means2D'].grad[var['seen'], :2], dim=-1)
            var['denom'][var['seen']] += 1
            return var

        reset_hooks_flag = False

        # total number of iteration since the beginning
        total_iters = frame_id * self.num_mappings_iter + mapping_iter

        to_split = torch.tensor([0]).to(self.device)
        to_clone = torch.tensor([0]).to(self.device)
        if self.densification_config['densify_from_iter'] <= (total_iters - self.densification_config['densify_from_iter']) <= self.densification_config['densify_until_iter']:
            if self.densification_config['do_legacy_densification']:
                self.variables = accumulate_mean2d_gradient(self.variables)
                grad_thresh = self.densification_config['grad_thresh']
                if (total_iters - self.densification_config['densify_from_iter']) % self.densification_config['densify_every'] == 0:
                # GS densification according to grad magnitude
                    grads = self.variables['means2D_gradient_accum'] / self.variables['denom']
                    grads[grads.isnan()] = 0.0

                    counts = self.variables['denom'].clone()

                    liable_points = torch.logical_and(counts >= (self.densification_config['densify_every'] // 2),
                                                      (self.variables['param_update_count_opacity'] - self.densification_config['densify_from_iter']) >= (self.densification_config['densify_every'] // 2))


                    num_pts = self.params['means3D'].shape[0]

                    # under reconstruction
                    # gaussians too small with too high spatial grad are cloned (simply copyied and cat to opt)
                    to_clone = torch.logical_and(grads >= grad_thresh,
                                                 (torch.max(torch.exp(self.params['log_scales']), dim=1).values <= 0.01 * self.variables['scene_radius']))

                    # to_clone = torch.logical_and(to_clone, counts >= (self.densification_config['densify_every'] // 2))  # this last added to ensure fresh params are not used
                    to_clone = torch.logical_and(to_clone, liable_points)
                    if torch.any(to_clone):
                        reset_hooks_flag = True
                        new_params = {k: v[to_clone] if k != 'logit_opacity' else torch.zeros((to_clone.shape[0], 1)).float().to(self.device) for k, v in self.params.items()}

                        self.params = optimizer.cat_params(new_params, self.params, self.variables)
                        num_pts = self.params['means3D'].shape[0]


                    # grads are padded becouse we have cat some new points
                    padded_grad = torch.zeros(num_pts, device=self.device)
                    padded_grad[:grads.shape[0]] = grads
                    padded_counts = torch.zeros(num_pts, device=self.device)
                    padded_counts[:counts.shape[0]] = counts
                    padded_liable_points = torch.zeros(num_pts, device=self.device)
                    padded_liable_points[:liable_points.shape[0]] = liable_points

                    # over reconstruction
                    to_split = torch.logical_and(padded_grad >= grad_thresh,
                                                 torch.max(torch.exp(self.params['log_scales']), dim=1).values > 0.01 * self.variables['scene_radius'])

                    to_split = torch.logical_and(to_split, padded_liable_points)

                    if torch.any(to_split):
                        reset_hooks_flag = True
                        n = self.densification_config['num_to_split_into']  # number to split into

                        new_params = {k: v[to_split].repeat(n, 1) if k != 'feature_rest' else v[to_split].repeat(n,1,1) for k, v in self.params.items()}

                        # we sample the new gaussians location using the initial big one as prob. distribution
                        if self.params['log_scales'].shape[-1] == 1:
                            stds = torch.exp(self.params['log_scales'])[to_split].repeat(n, 3)
                        else:
                            stds = torch.exp(self.params['log_scales'])[to_split].repeat(n, 1)

                        # means are zeros becouse then we add to actual mean (we sample from the unbias distribution and then we add bias)
                        means = torch.zeros((stds.size(0), 3), device=self.device)
                        samples = torch.normal(mean=means, std=stds)
                        rots = quat2rot(F.normalize(self.params['unnorm_rotations'][to_split])).repeat(n, 1, 1)
                        new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)

                        if self.densification_config['init_split_scale'] == 'as_init':
                            new_scales = self._init_scales_dens(new_params['means3D'], w2c, camera.k, mode=self.data_config['scale_init_mode'])
                            if self.data_config['isotropy']:
                                scales = torch.tile(torch.sqrt(new_scales)[..., None], (1, 1))
                            else:
                                scales = torch.tile(torch.sqrt(new_scales)[..., None], (1, 3))
                                scales[:, 2] = new_scales / 10
                            new_params['log_scales'] = torch.log(scales)
                        elif self.densification_config['init_split_scale'] == 'fraction':
                            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (1 * n))

                        self.params = optimizer.cat_params(new_params, self.params, self.variables)

                        # removing big gaussians that have been split
                        to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device=self.device)))

                        self.params, self.variables = optimizer.prune(to_remove, self.params, self.variables)

                    self.variables['means2D_gradient_accum'] = torch.zeros_like(self.variables['means2D_gradient_accum'])
                    self.variables['denom'] = torch.zeros_like(self.variables['denom'])
                    self.variables['max_2D_radius'] = torch.zeros_like(self.variables['max_2D_radius'])

        # done from the beginning
        to_remove = torch.tensor([0]).to(self.device)
        if  (total_iters>0) and (total_iters % self.densification_config['prune_every_map_iter'] == 0):
            reset_hooks_flag = True
            # removing low opacity gaussians
            remove_threshold = self.densification_config['removal_opacity_threshold']
            to_remove = (torch.sigmoid(self.params['logit_opacities']) < remove_threshold).squeeze()

            # removing too big gaussians (useless if we densify)
            if not self.densification_config['do_legacy_densification']:
                big_points_ws = torch.exp(self.params['log_scales']).max(dim=1).values > 0.1 * self.variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)

            self.params, self.variables = optimizer.prune(to_remove, self.params, self.variables)

        # Reset Opacities for all Gaussians (This is not desired for mapping on only current frame)
        if (total_iters % self.densification_config['reset_opacities_every'] == 0 and mapping_iter > 0
            and self.densification_config['densify_from_iter'] <= total_iters <= self.densification_config['densify_until_iter']) :

            reset_hooks_flag = True
            self.params = optimizer.reset_opacity(self.params, self.variables)
            self.just_reset_flag = True

        # lr scheduler update and logging (only if something changed)
        if reset_hooks_flag:
            logger.after_dens(frame_id, (torch.sum(to_remove).cpu().item(), torch.sum(to_split).cpu().item(), torch.sum(to_clone).cpu().item()), self.variables['scene_radius'])

            if (torch.sum(to_remove).cpu().item() + torch.sum(to_split).cpu().item()) >= 5000:
                self.reset_opacity_scheduler()

            self.reset_hooks()
            self.set_grad_hooks()

    # culls points according to distance from 3 KNN
    def cull_outliers(self, k=2):
        with torch.no_grad():
            pcd = self.params['means3D'].detach()
            N = pcd.shape[0]
            initial_scales = torch.clamp_min(distCUDA2(pcd[:, :3].float().cuda()), 0.0000001)  # * 1.5
            q1 = torch.quantile(initial_scales, 0.25)
            q3 = torch.quantile(initial_scales, 0.75)
            IQR = q3 - q1

            to_keep = (initial_scales < (q3 + k * IQR))

            my_logger.info(f'   - Removed {(~to_keep).detach().cpu().sum().item()} outliers; mean dis {initial_scales.mean().detach().cpu().item(): .4f}; th: {(q3 + k * IQR).detach().cpu().item(): .4f}')

        culled_params = {k: v[to_keep].detach().requires_grad_(True) if v.shape[0] == N else v.detach().requires_grad_(True) for k, v in self.params.items()}
        culled_vars = {k: v[to_keep] if k not in ["seen", "means2D", "scene_radius"] else v for k, v in self.variables.items()}

        self.params = culled_params
        self.variables = culled_vars

    def _init_params_vars(self, init_pt_cld, initial_scales, return_vars=True, adding=False):
        """
        initializes the gaussian's parameters

        args:
            -- init_pt_cld: the initial point cloud
            -- initial_scales: the initial computed scales
            -- return_vars: whether to return variables
            -- adding: whether it is used for addition of gaussians

        """

        num_pts = init_pt_cld.shape[0]
        colors = self.params['rgb_colors'].mean(dim=0).detach() * torch.ones_like(init_pt_cld[:,:3]).float().to(self.device) if (adding and self.do_pbr) else init_pt_cld[:, 3:6]

        means3D = init_pt_cld[:, :3]  # [num_gaussians, 3]

        if adding:
            dists = torch.cdist(means3D.to(self.device), self.params['means3D'], p=2)  # (M, N)
            closest_indices = torch.argmin(dists, dim=1)  # (M,)
            unnorm_rots = self.params['unnorm_rotations'][closest_indices, :].clone()
        else:
            if self.data_config['align_init_axis']:
                unnorm_rots = rotation_aligned_with_normal(init_pt_cld[:, 6:9], True)
            else:
                unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))

        if self.data_config['isotropy']:
            log_scales = torch.tile(torch.log(torch.sqrt(initial_scales))[..., None], (1, 1))
        else:
            a = torch.tile(torch.sqrt(initial_scales)[..., None], (1, 3))

            # getting z to be smaller so to align it with surface normal
            a[:,2] = initial_scales / 10
            log_scales = torch.log(a)

        logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float)



        params = {
            'means3D': means3D,
            'rgb_colors': colors if self.active_sh_degree < 1 else RGB2SH(colors),
            'unnorm_rotations': unnorm_rots,
            'logit_opacities': logit_opacities,
            'log_scales': log_scales,
        }

        if self.data_config['max_sh_degree'] > 0:
            # setting sh coefficient for R G B channels (order 0 is still 'rgb_colors' so -1)
            params['feature_rest'] = torch.zeros((num_pts, 3, ((self.data_config['max_sh_degree'] + 1) ** 2) - 1)).float().to(self.device)

        if self.do_pbr:
            # if not adding:
            params['F0'] = (0.035 * torch.ones((num_pts, 1))).float().to(self.device)
            params['roughness'] = (0.3 * torch.ones((num_pts, 1))).float().to(self.device)
            # else:
            #     params['F0'] = self.params['F0'].mean(dim=0).detach() * torch.ones((num_pts, 1)).float().to(self.device)
            #     params['roughness'] = self.params['roughness'].mean(dim=0).detach() * torch.ones((num_pts, 1)).float().to(self.device)

        # sets every param as a torch tensor and brings them to device
        for k, v in params.items():
            # Check if value is already a torch tensor
            if not isinstance(v, torch.Tensor):
                params[k] = torch.nn.Parameter(torch.tensor(v).to(self.device).float().contiguous().requires_grad_(True))
            else:
                params[k] = torch.nn.Parameter(v.to(self.device).float().contiguous().requires_grad_(True))

        if return_vars:
            variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
                         'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
                         'denom': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
                         'param_update_count': torch.zeros(params['means3D'].shape[0]).to(self.device).float(),
                         'param_update_count_opacity': torch.zeros(params['means3D'].shape[0]).to(self.device).float()}

            return params, variables
        else:
            return params

    def _init_scales(self, init_pt_cld, depth, intrinsics, mask=None, mode='depth'):

        if mode == 'depth':
            # from EndoGSLAM
            # Projective Geometry (this is fast, farther -> larger radius)
            FX, FY = intrinsics[0][0], intrinsics[1][1]
            depth_z = depth[0].reshape(-1)
            scale_gaussian = depth_z / ((FX + FY)/2)
            initial_scales = scale_gaussian**2 #* 2
            if isinstance(mask, torch.Tensor):
                initial_scales = initial_scales[mask]
            return initial_scales
        elif mode == 'nearest':
            # from GS pancakes
            # find mean dist to nearest 3  (not suited with dense map)
            with torch.no_grad():
                initial_scales = torch.clamp_min(distCUDA2(init_pt_cld[:, :3].clone().detach().float().to(self.device)), 0.0000001)
            return initial_scales.detach()
        else:
            raise NotImplementedError(f'mode {mode} not implemented')

    def _init_scales_dens(self, pt_cld, w2c, intrinsics, mode='depth'):
        if mode == 'depth':
            # from EndoGSLAM
            # Projective Geometry (this is fast, farther -> larger radius)
            FX, FY = intrinsics[0][0], intrinsics[1][1]

            depth_z = get_depth_and_silhouette(pt_cld, w2c)[:, 0]
            scale_gaussian = depth_z / ((FX + FY)/2)
            initial_scales = scale_gaussian**2

            return initial_scales
        elif mode == 'nearest':
            # from GS pancakes
            # find mean dist to nearest 3 neighbors
            initial_scales = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pt_cld[:, :3])).float().to(self.device)), 0.0000001)
            return initial_scales
        else:
            raise NotImplementedError(f'mode {mode} not implemented')

    def oneupSHdegree(self):
        self.active_sh_degree += 1

    def set_seen(self, radii):
        seen = radii > 0
        self.variables['max_2D_radius'][seen] = torch.max(radii[seen], self.variables['max_2D_radius'][seen])
        self.variables['seen'] = seen
        self.variables['param_update_count'][seen] += 1
        self.variables['param_update_count_opacity'][seen] += 1

    def compute_gaussian_rgb(self, camera_center):
        n = self.params['rgb_colors'].shape[0]
        shs_view = torch.cat((self.params['rgb_colors'].reshape(n, 3, -1).transpose(1, 2),
                                          self.params['feature_rest'].transpose(1, 2)), dim=1).transpose(1,2)
        dir_pp = (self.params['means3D'] - camera_center.repeat(n, 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        return sh2rgb +0.5 #torch.clamp_min(sh2rgb + 0.5, 0.0)

    def compute_pbr_color(self, light, camera_center, return_pbr_params=False, light_center_sep=None, diffuse_only=False,
                          return_coeffs=False):

        if torch.is_tensor(light_center_sep):
            light_center_raw = light_center_sep
        else:
            light_center_raw = camera_center

        # PR-Endo uses SH to compute albedo
        if self.active_sh_degree > 0:
            base_color = self.compute_gaussian_rgb(camera_center)
        else:
            base_color = self.params['rgb_colors']

        light_intensity, attenuation_k, attenuation_power = light[0], light[1], light[2]

        # light directions
        light_center = light_center_raw
        dir_pp_light = (self.params['means3D'] - light_center.repeat(self.params['means3D'].shape[0], 1))
        dir_pp_light = torch.nn.functional.normalize(dir_pp_light, dim=1)

        dir_gauss_lightcenter = (self.params['means3D'] - light_center.repeat(self.params['means3D'].shape[0], 1))
        light_gauss_dist = dir_gauss_lightcenter.norm(dim=1, keepdim=True) #/ 3

        dir_pp_camera = (self.params['means3D'] - camera_center.repeat(self.params['means3D'].shape[0], 1))

        normal = self.get_unnorms_norms

        # Normalize the input vectors
        N = torch.nn.functional.normalize(normal, dim=1)
        L = -torch.nn.functional.normalize(dir_pp_light, dim=1)
        V = -torch.nn.functional.normalize(dir_pp_camera, dim=1)

        # normals always towards camera
        N_dot_V = torch.sum(N * V, dim=1, keepdim=True)  # [N, 1]
        N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0

        # cosine
        N_dot_L = torch.clamp(torch.sum(N * L, dim=1, keepdim=True), min=0.0)

        # Compute distance attenuation (light intensity fades with distance)
        attenuation_raw_coeffs = 1.0 / (1.0 + attenuation_k * light_gauss_dist ** attenuation_power)
        attenuation_coeffs = torch.clamp(attenuation_raw_coeffs, 0, 1)

        # Diffuse component - from coefficients
        I_diffuse_color_coeffs = light_intensity * N_dot_L * attenuation_coeffs * base_color

        # ======== PBR reflections

        # Compute halfway vector for specular component
        H = F.normalize(L + V, dim=1)

        # Fresnel Effect (Schlick's approximation) for non-metals (F0 = 0.04) -value from gpt
        fresnel = self.get_clamped_F0 + (1 - self.get_clamped_F0) * (1 - torch.sum(H * V, dim=1, keepdim=True)) ** 5

        # Specular component using Cook-Torrance model
        N_dot_H = torch.clamp(torch.sum(N * H, dim=1, keepdim=True), min=0.0)
        alpha = (self.params['roughness'].clamp(0, 0.7)) ** 2

        # Microfacet distribution function (Trowbridge-Reitz GGX)
        D = (alpha ** 2) / (torch.pi * ((N_dot_H ** 2) * (alpha ** 2 - 1) + 1) ** 2)

        # Geometric attenuation (Smith GGX)
        k = (self.params['roughness'] + 1) ** 2 / 8
        N_dot_V_clamped = torch.clamp(torch.sum(N * V, dim=1, keepdim=True), min=0.0)
        N_dot_L_clamped = torch.clamp(N_dot_L, min=0.0)
        G1_V = N_dot_V_clamped / (N_dot_V_clamped * (1 - k) + k)
        G1_L = N_dot_L_clamped / (N_dot_L_clamped * (1 - k) + k)
        G = G1_V * G1_L

        # Final Cook-Torrance specular term
        specular_component_coeffs = (fresnel * D * G) / (4 * N_dot_V_clamped * N_dot_L_clamped + 1e-5)
        I_specular_coeffs = light_intensity * attenuation_coeffs * specular_component_coeffs * N_dot_L

        I_diffuse_rough, I_specular_rough = I_diffuse_color_coeffs * (1 - fresnel), I_specular_coeffs

        I_diffuse_final, I_specular_final = torch.clamp_min(I_diffuse_rough, 0), torch.clamp_min(I_specular_rough, 0)

        reflected_rgb = I_diffuse_final + I_specular_final if not diffuse_only else I_diffuse_final

        if return_pbr_params:
            if return_coeffs:
                coeffs = torch.cat((N_dot_L * light_intensity * (1 - fresnel), attenuation_coeffs, I_specular_final), dim=1)  # Nx3
                return reflected_rgb, (base_color, self.params['roughness'], self.params['F0']), coeffs
            else:
                return reflected_rgb, (base_color, self.params['roughness'], self.params['F0'])
        else:
            if return_coeffs:
                coeffs = torch.cat((N_dot_L * light_intensity * (1 - fresnel), attenuation_coeffs, I_specular_final), dim=1)
                return reflected_rgb, coeffs
            else:
                return reflected_rgb

    # grad hooks for parameter-wise lr scheduler
    def set_grad_hooks(self):
        if self.lr_config['use_scheduler']:
            # register grad hook
            hooks = []
            for k in self.params.keys():
                if k == 'feature_rest':
                    alpha = self.lr_config['alpha_sh']
                    x_offset = self.lr_config['x_offset_sh']
                    y_offset = self.lr_config['y_offset_sh']
                    visited = self.variables['param_update_count']
                elif k == 'logit_opacities':
                    alpha = self.lr_config['alpha_opacity']
                    x_offset = self.lr_config['x_offset_opacity']
                    y_offset = self.lr_config['y_offset_opacity']
                    visited = self.variables['param_update_count_opacity']
                elif k == 'rgb_colors':
                    alpha = self.lr_config['alpha_color']
                    x_offset = self.lr_config['x_offset_color']
                    y_offset = self.lr_config['y_offset_color']
                    visited = self.variables['param_update_count']
                elif k == 'F0' or k == 'roughness':
                    alpha = self.lr_config['alpha_pbr']
                    x_offset = self.lr_config['x_offset_pbr']
                    y_offset = self.lr_config['y_offset_pbr']
                    visited = self.variables['param_update_count']
                elif k == 'unnorm_rotations':
                    alpha = self.lr_config['alpha_rot']
                    x_offset = self.lr_config['x_offset_rot']
                    y_offset = self.lr_config['y_offset_rot']
                    visited = self.variables['param_update_count']
                elif k == 'log_scales':
                    alpha = self.lr_config['alpha_scale']
                    x_offset = self.lr_config['x_offset_scale']
                    y_offset = self.lr_config['y_offset_scale']
                    visited = self.variables['param_update_count']
                else:
                    # position
                    alpha = self.lr_config['alpha_spatial']
                    x_offset = self.lr_config['x_offset_spatial']
                    y_offset = self.lr_config['y_offset_spatial']
                    visited = self.variables['param_update_count']

                norm = self.grad_weight_sched(torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"), alpha, x_offset,
                                              torch.ones(1, device="cuda"), y_offset=y_offset)
                hooks.append(self.params[k].register_hook(partial(self.grad_weight_sched, visited=visited, visit_alpha=alpha,
                                                                  x_offset=x_offset, norm=norm, y_offset=y_offset)))
            self.hooks = hooks

    def reset_hooks(self):
        if self.hooks is not None:
            [h.remove() for h in self.hooks]
            self.hooks = None

    @staticmethod
    def grad_weight_sched(grad, visited, visit_alpha, x_offset, norm, y_offset=0.0):
        """
            weight gradient by visit function -> points that have often been updated will get smaller gradient
        """
        # ToDo make broadcasting without transpose as it uses non contiguous views
        return (grad.transpose(0,-1) * ((1.0 + y_offset) - torch.sigmoid(visit_alpha * (visited.squeeze() - x_offset))) / norm).transpose(0,-1)

    def reset_scheduler(self, frame_id):
        self.stored_params[f'frame_{frame_id:04d}'] = {k: self.params[k].detach().clone() for k in self.params.keys()}
        self.reset_hooks()
        self.set_grad_hooks()
        self.variables['param_update_count']= torch.zeros(self.params['means3D'].shape[0]).to(self.device).float()
        self.variables['param_update_count_opacity'] = torch.zeros(self.params['means3D'].shape[0]).to(self.device).float()
        self.variables['denom'] = torch.zeros(self.params['means3D'].shape[0]).to(self.device).float()
        self.variables['means2D_gradient_accum'] = torch.zeros(self.params['means3D'].shape[0]).to(self.device).float()

    def reset_opacity_scheduler(self):
        self.variables['param_update_count_opacity'] = torch.zeros(self.params['means3D'].shape[0]).to(self.device).float()

    @staticmethod
    def _get_first_pointcloud(first_color, first_depth, intrinsics, mask=None, w2c=None):
        """
        args:
            -- color: the image frame (CxHxW)
            -- depth: the depth map (CxHxW)
            -- intrinsics: the camera intrinsics
            -- w2c: the world to camera transformation matrix, if not None it uses it to transform the point from camera coord into world frame
            -- mask: mask tensor for valid pixels
        """

        pts_cld = get_pointcloud(first_color, first_depth, intrinsics, w2c=w2c, mask=mask)
        return pts_cld

    @property
    def get_params(self):
        assert self.params is not None, "Should call '.init_model' first"
        return self.params

    @property
    def get_clamped_F0(self):
        # return self.params['F0'].clamp(0, 0.035)
        return self.params['F0'].clamp(min=0)

    @property
    def get_variables(self):
        assert self.variables is not None, "Should call '.init_model' first"
        return self.variables

    @property
    def get_cov_detached(self):
        return self.get_cov.detach()

    @property
    def get_cov(self):
        q = F.normalize(self.params['unnorm_rotations'])

        if self.params['log_scales'].shape[1] == 1:
            s = torch.exp(torch.tile(self.params['log_scales'], (1, 3)))
        else:
            s = torch.exp(self.params['log_scales'])

        R = quat2rot(q)  # (N, 3, 3)

        S = torch.diag_embed(s)  # (N, 3, 3)

        L = torch.bmm(R, S)

        cov = L @ L.transpose(1, 2)  # (N, 3, 3)

        return cov

    @property
    def get_gaussians_normals(self):
        #GS pancakes

        scale = torch.exp(self.params['log_scales'])
        if scale.shape[1] == 1:
            scale = torch.tile(scale, (1, 3))

        rot = F.normalize(self.params['unnorm_rotations'])

        cuda_normal = compute_gauss_norm(scale, rot, 1) # scaling modifier == 1
        return cuda_normal

    @property
    def get_smaller_axis(self):
        #Endo-4DGS

        scale = torch.exp(self.params['log_scales'])
        if scale.shape[1] == 1:
            scale = torch.tile(scale, (1, 3))

        rot = quat2rot(F.normalize(self.params['unnorm_rotations']))
        smallest_axis_idx = scale.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rot.gather(2, smallest_axis_idx)

        return smallest_axis.squeeze(-1)  # in world frame

    @property
    def get_unnorms_norms(self):
        mode = self.data_config['normals_mode']
        if mode == 'torch':
            return self.get_smaller_axis
        elif mode == 'cuda':
            return self.get_gaussians_normals
        else:
            raise NotImplementedError




