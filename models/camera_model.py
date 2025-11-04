import torch
import torch.nn.functional as F
# from diff_gaussian_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer as LegacyRenderer
from diff_gaussian_rasterization import GaussianRasterizationSettings
# from diff_gaussian_rasterization_depth import GaussianRasterizer as Renderer
import numpy as np
from matching import get_matcher

from LumenGSLAM.utils.data_process.pose_handling import rot2quat, quat2rot, align_features_pnp
from LumenGSLAM.utils.data_process.preprocessing import get_depth_and_silhouette
from LumenGSLAM.utils.data_process.geometry import vectors2frame, SE3_exp
from LumenGSLAM.utils.general import my_logger
from LumenGSLAM.models.feature_matcher import get_keypoints_3D_2D


class Camera:
    def __init__(self, w, h, k, device, near=0.01, far=100, bg=[0,0,0], save_path=None):
        """

        args:
            -- w: width of image to render
            -- h: height of image to render
            -- k: intrinsic camera parameters
            -- device: device to render
            -- near: distance of near plane of camera frustrum
            -- far: distance of far plane of camera frustrum
            -- bg: color of rendered background

        """

        self.save_path = save_path
        self.device = device
        self.w = w
        self.h = h
        self.k = k.to(device)
        self.near = near
        self.far = far
        self.bg = bg

        self.w2c_traj = []
        self.quat_trasl_traj = []

        self.opengl_proj = self.get_opengl_proj

        self.light = torch.nn.Parameter(torch.tensor([3.0, 0.13, 0.80]).float().to(self.device).requires_grad_(True))

        self.delta_cam = {'cam_rot_delta': torch.nn.Parameter(torch.zeros(3, requires_grad=True, device=device)),
                          'cam_trans_delta': torch.nn.Parameter(torch.zeros(3, requires_grad=True, device=device))}

        self.matcher = None


    def set_w2c(self, w2c: torch.Tensor):
        # use it when NOT tracking (gt poses used)
        self.w2c_traj.append(w2c.detach().to(self.device))

    def init_quat_trasl(self):
        # use it when tracking (estimating pose)
        cam_rots = np.tile([1, 0, 0, 0], (1, 1))

        quat = torch.Tensor(cam_rots).to(self.device).float().contiguous().requires_grad_(True)
        transl = torch.Tensor(np.zeros((1, 3))).to(self.device).float().contiguous().requires_grad_(True)
        self.quat_trasl_traj.append({'cam_unnorm_rots': quat,
                                     'cam_trans': transl})

        self.set_w2c_from_best_quat_transl(quat, transl)


    def set_w2c_from_best_quat_transl(self, quat, transl):
        # use it when tracking (after having estimated the pose)
        R = quat2rot(F.normalize(quat.detach()))
        t = transl.detach()

        self.quat_trasl_traj[-1] = {'cam_unnorm_rots': quat.to(self.device).float().contiguous().requires_grad_(True),
                                    'cam_trans': transl.to(self.device).float().contiguous().requires_grad_(True)}

        w2c = torch.eye(4).to(self.device).float()
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        self.w2c_traj.append(w2c.detach())

    def init_next_pose_constant_speed(self):
        if len(self.quat_trasl_traj) > 1:

            prev_rot1 = F.normalize(self.quat_trasl_traj[-1]['cam_unnorm_rots'].clone().detach())
            prev_rot2 = F.normalize(self.quat_trasl_traj[-2]['cam_unnorm_rots'].clone().detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))

            # Translation
            prev_tran1 = self.quat_trasl_traj[-1]['cam_trans'].clone().detach()
            prev_tran2 = self.quat_trasl_traj[-2]['cam_trans'].clone().detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
        else:
            new_rot = self.quat_trasl_traj[-1]['cam_unnorm_rots'].clone().detach()
            new_tran = self.quat_trasl_traj[-1]['cam_trans'].clone().detach()

        my_logger.info(f"   - pose estimated with constant speed")

        return new_rot, new_tran

    def init_nest_pose_identity(self, last_frame, current_frame, loader, last_view, return_keypoints=True):
        if self.matcher is None:
            self.matcher = get_matcher('superpoint-lg', device=self.device)

        new_rot = self.quat_trasl_traj[-1]['cam_unnorm_rots'].clone().detach()
        new_tran = self.quat_trasl_traj[-1]['cam_trans'].clone().detach()

        if return_keypoints:
            i0, d0, p0 = last_frame['gt_im'], last_frame['gt_depth'], last_frame['r_w2c']
            i1 = current_frame['gt_im']

            p0 = self.w2c_traj[-1]

            pts_0, inlier_kpts1, _ = get_keypoints_3D_2D(self.matcher, i0, d0, p0, i1, self.k)

            return new_rot, new_tran, (pts_0, inlier_kpts1)
        else:
            return new_rot, new_tran, (None, None)

    def init_next_pose_feature_pnp(self, last_frame, current_frame, loader, last_view, depth_max=100.0):
        if self.matcher is None:
            self.matcher = get_matcher('superpoint-lg', device=self.device)
            # self.matcher = MyMatcher(self.device,0.90)

        i0, d0, p0, sil0 = last_frame['gt_im'], last_frame['gt_depth'], last_frame['r_w2c'], last_frame['r_sil']

        # avoid finding matches on low confidence / invalid pixels

        i0 = i0 * ((d0 > 0) & (d0 < depth_max)).to(int)

        i1 = current_frame['gt_im'] * ((current_frame['gt_depth']>0) & (current_frame['gt_depth']<depth_max)).to(int)

        # # debug
        # save_path = self.save_path / 'superpoint_matches'
        # os.makedirs(save_path, exist_ok=True)
        # save_path = save_path / '{:04d}.png'.format(i_curr)

        save_path = None
        success, w2c_init, keypoints = align_features_pnp((i0.to(self.device), d0.to(self.device), p0.to(self.device)), i1.to(self.device), self.k, self.matcher, save_path )#, debug_img=iii)

        if success:
            R = w2c_init[:3, :3]
            new_tran = torch.tensor(np.tile(w2c_init[:3, 3], (1, 1)))
            new_rot = torch.tensor(np.tile(rot2quat(R), (1,1)))
            my_logger.info(f"   - pose estimated with PnP")

        else:
            my_logger.info('   - failed to initialize with PnP')
            # new_rot, new_tran = self.init_next_pose_constant_speed()
            new_rot = self.quat_trasl_traj[-1]['cam_unnorm_rots'].clone().detach()
            new_tran = self.quat_trasl_traj[-1]['cam_trans'].clone().detach()
        return new_rot.detach(), new_tran.detach(), keypoints, success

    def init_next_frame_pose(self, last_frame, current_frame, mode='constant_speed', loader=None, last_view=None, gt_w2c=None,
                             return_keypoints=True, depth_max=100.0):
        # Initialize the camera pose for the current frame based on a constant velocity model
        with torch.no_grad():
            if mode == 'constant_speed':
                new_rot, new_tran = self.init_next_pose_constant_speed()
                keypoints = None
                success = True
            elif mode == 'features_pnp':
                new_rot, new_tran, keypoints, success = self.init_next_pose_feature_pnp(last_frame, current_frame, loader, last_view, depth_max=depth_max)
            elif mode == 'identity':
                new_rot, new_tran, keypoints = self.init_nest_pose_identity(last_frame, current_frame, loader, last_view, return_keypoints)
                success = True
            else:
                raise NotImplementedError

        self.quat_trasl_traj.append({'cam_unnorm_rots': new_rot.to(self.device).requires_grad_(True), 'cam_trans': new_tran.to(self.device).requires_grad_(True)})

        with torch.no_grad():
            init_w2c = torch.tensor(np.eye(4)).float().to(self.device)
            init_w2c[:3, :3] = quat2rot(F.normalize(new_rot)).clone().detach()
            init_w2c[:3, 3] = new_tran.clone().detach()

            if current_frame['gt_w2c'] is not None:
                ate = ((torch.linalg.inv(current_frame['gt_w2c'])[:3, 3].to(self.device) - torch.linalg.inv(init_w2c)[:3, 3]) ** 2).mean().detach()
                print(f'   - initialization ATE: {ate: .4f}')

        return new_rot, new_tran, init_w2c, keypoints, success

    @property
    def get_opengl_proj(self):
        fx, fy, cx, cy = self.k[0][0], self.k[1][1], self.k[0][2], self.k[1][2]
        opengl_proj = torch.tensor([[2 * fx / self.w, 0.0, -(self.w - 2 * cx) / self.w, 0.0],
                                    [0.0, 2 * fy / self.h, -(self.h - 2 * cy) / self.h, 0.0],
                                    [0.0, 0.0, self.far / (self.far - self.near), -(self.far * self.near) / (self.far - self.near)],
                                    [0.0, 0.0, 1.0, 0.0]]).to(self.device).float().unsqueeze(0).transpose(1, 2)
        return opengl_proj

    def get_rasterization_settings(self, w2c, sh_degree):

        cam_center = torch.inverse(w2c)[:3, 3]
        w2c =w2c.unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(self.opengl_proj)

        fx, fy = self.k[0][0], self.k[1][1]

        cam = GaussianRasterizationSettings(
            image_height=self.h,
            image_width=self.w,
            tanfovx=self.w / (2 * fx),
            tanfovy=self.h / (2 * fy),
            bg=torch.tensor(self.bg, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=sh_degree,
            campos=cam_center,
            prefiltered=False,
            debug=False # only for _depth
        )
        return cam

    @property
    def get_current_w2c(self):
        assert len(self.w2c_traj) > 0
        return self.w2c_traj[-1]

    @property
    def get_current_position(self):
        assert len(self.w2c_traj) > 0
        return self.w2c_traj[-1][:3, 3]

    @property
    def get_current_quat_transl(self):
        assert len(self.quat_trasl_traj) > 0
        return self.quat_trasl_traj[-1]

    def get_w2c_for_optimization(self, idx=-1):
        R = quat2rot(F.normalize(self.quat_trasl_traj[idx]['cam_unnorm_rots']))
        t = self.quat_trasl_traj[idx]['cam_trans']#.requires_grad_(True)

        w2c = torch.eye(4).to(self.device).float()
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        return w2c

    @property
    def get_current_max_dist(self):
        traj = torch.stack(self.w2c_traj, dim=0)
        centers = traj[:,:3,3]
        avg_cam_center = centers.mean(dim=0)
        dist = torch.linalg.norm(centers - avg_cam_center, dim=1, keepdims=True)
        return dist.max().item()

    # used in MonoGS
    def update_pose_lie(self):

        tau = torch.cat([self.delta_cam['cam_trans_delta'], self.delta_cam['cam_rot_delta']], axis=0)

        T_w2c = self.get_current_w2c

        new_w2c = SE3_exp(tau) @ T_w2c

        self.w2c_traj[-1] = new_w2c.detach().to(self.device)

        self.delta_cam['cam_rot_delta'].data.fill_(0)
        self.delta_cam['cam_trans_delta'].data.fill_(0)

    def reset_deltas(self):
        self.delta_cam = {'cam_rot_delta': torch.nn.Parameter(torch.zeros(3, requires_grad=True, device=self.device)),
                          'cam_trans_delta': torch.nn.Parameter(torch.zeros(3, requires_grad=True, device=self.device))}


class SceneRenderer:
    def __init__(self, device):
        self.device = device
        self.current_rendervar_im = None
        self.current_rendervar_depth_sil = None
        self.current_rendervar_norms = None


    def set_rendervar_im(self, params, retain_grad=True, pbr_color=None):
        """
        transforms params in EndoGSLAM format to dict suitable for GS rasterization of image

        args:
            -- params: EndoGSLAM format prams

        """

        # where gradients for densification are stored
        means2D = torch.zeros_like(params['means3D'], dtype=params['means3D'].dtype, requires_grad=True, device=self.device) + 0

        if retain_grad:
            means2D.retain_grad()

        rendervar = {
            'means3D': params['means3D'],
            'opacities': torch.sigmoid(params['logit_opacities']),
            'means2D': means2D,
        }

        if not isinstance(pbr_color, torch.Tensor):
            if "feature_rest" not in params.keys():
                rendervar['colors_precomp'] = params['rgb_colors']
            else:
                # concatenating all color params into sh components
                rendervar['shs'] = torch.cat((params['rgb_colors'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2),
                                              params['feature_rest'].transpose(1, 2)), dim=1)
                                              # params['feature_rest'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2)), dim=1)
        else:
            rendervar['colors_precomp'] = pbr_color

        if 'cov3D_precomp' in params.keys():
            rendervar['cov3D_precomp'] = params['cov3D_precomp']
        else:
            rendervar['rotations'] = F.normalize(params['unnorm_rotations'])
            if params['log_scales'].shape[1] == 1:
                # Isotropy
                rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
            else:
                # Anisotropy
                rendervar['scales'] = torch.exp(params['log_scales'])

        self.current_rendervar_im = rendervar

    def set_rendervar_depthplus_silhouette(self, params, w2c, cov_prerot=None):
        """
        transforms params in EndoGSLAM format to dict suitable for GS rasterization of Depth + Silhouette

        args:
            -- params: EndoGSLAM format prams
        """

        rendervar = {
            'means3D': params['means3D'],
            'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=self.device) + 0
        }

        if 'cov3D_precomp' in params.keys():
            rendervar['cov3D_precomp'] = params['cov3D_precomp']
        elif isinstance(cov_prerot, torch.Tensor):
            rendervar['cov3D_precomp'] = cov_prerot
        else:
            rendervar['rotations'] = F.normalize(params['unnorm_rotations'])
            if params['log_scales'].shape[1] == 1:
                # Isotropy
                rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
            else:
                # Anisotropy
                rendervar['scales'] = torch.exp(params['log_scales'])


        self.current_rendervar_depth_sil = rendervar


    def set_rendervar_normals(self, params, normals, w2c):
        """
        transforms params in EndoGSLAM format to dict suitable for GS rasterization of Surface normals

        args:
            -- params: EndoGSLAM format prams
            -- normals: Gaussians normals (smmeler eigen vector)
            -- w2c: world2camera matrix
        """

        rendervar = {
            'means3D': params['means3D'],
            'colors_precomp': F.normalize(vectors2frame(normals, w2c), dim=1),
            'rotations': F.normalize(params['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device=self.device) + 0,
        }

        # this is anyway treated as color
        if params['log_scales'].shape[1] == 1:
            rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
        else:
            rendervar['scales'] = torch.exp(params['log_scales'])

        self.current_rendervar_norms = rendervar

    def render_image(self, rasterization_settings, variables=None):
        # ENSURE TO SET RENDERVARS BEFORE !!!!
        im, radius, _ = LegacyRenderer(raster_settings=rasterization_settings)(**self.current_rendervar_im)
        if isinstance(variables, dict):
            variables['means2D'] =  self.current_rendervar_im['means2D']

        return im, radius

    def render_depth_silhouette(self, rasterization_settings):
        # ENSURE TO SET RENDERVARS BEFORE !!!!
        depth_sil, _, _ = LegacyRenderer(raster_settings=rasterization_settings)(**self.current_rendervar_depth_sil)
        depth = depth_sil[0, :, :].unsqueeze(0)
        depth_sq = depth_sil[2, :, :].unsqueeze(0)
        silhouette = depth_sil[1, :, :]

        return depth, silhouette, depth_sq

    def render_normals(self, rasterization_settings):
        # ENSURE TO SET RENDERVARS BEFORE !!!!
        # normals map of current frame
        r_norms, _, _ = LegacyRenderer(raster_settings=rasterization_settings)(**self.current_rendervar_norms)
        return r_norms

    def render_monogs(self, params, camera, w2c, variables=None, pbr_color=None, retain_grad=True):

        cam_center = torch.inverse(w2c)[:3, 3]
        w2c =w2c.unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(camera.opengl_proj)

        fx, fy = camera.k[0][0], camera.k[1][1]

        raster_settings = GaussianRasterizationSettings(
            image_height=camera.h,
            image_width=camera.w,
            tanfovx=camera.w / (2 * fx),
            tanfovy=camera.h / (2 * fy),
            bg=torch.tensor(camera.bg, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            projmatrix_raw=camera.opengl_proj,
            sh_degree=3 if 'feature_rest' in params.keys() else 0,
            campos=cam_center,
            prefiltered=False,
            debug=False,
        )

        means2D = torch.zeros_like(params['means3D'], dtype=params['means3D'].dtype, requires_grad=True,
                                   device=self.device) + 0

        if retain_grad:
            means2D.retain_grad()

        rendervar = {
            'means3D': params['means3D'],
            'opacities': torch.sigmoid(params['logit_opacities']),
            'means2D': means2D,
        }

        if not isinstance(pbr_color, torch.Tensor):
            if "feature_rest" not in params.keys():
                rendervar['colors_precomp'] = params['rgb_colors']
            else:
                # concatenating all color params into sh components
                rendervar['shs'] = torch.cat(
                    (params['rgb_colors'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2),
                     params['feature_rest'].transpose(1, 2)), dim=1)
                # params['feature_rest'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2)), dim=1)
        else:
            rendervar['colors_precomp'] = pbr_color

        if 'cov3D_precomp' in params.keys():
            rendervar['cov3D_precomp'] = params['cov3D_precomp']
        else:
            rendervar['rotations'] = F.normalize(params['unnorm_rotations'])
            if params['log_scales'].shape[1] == 1:
                # Isotropy
                rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
            else:
                # Anisotropy
                rendervar['scales'] = torch.exp(params['log_scales'])

        rendervar['theta'] = camera.delta_cam['cam_rot_delta']
        rendervar['rho'] = camera.delta_cam['cam_trans_delta']

        # rendered_image, radii, depth, opacity = LegacyRenderer(raster_settings=raster_settings)(**rendervar)
        rendered_image, radii, depth, opacity, _ = LegacyRenderer(raster_settings=raster_settings)(**rendervar)


        if isinstance(variables, dict):
            variables['means2D'] = rendervar['means2D']

        return rendered_image, radii, depth, opacity