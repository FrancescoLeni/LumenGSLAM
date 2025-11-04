import torch
import numpy as np
from tqdm import tqdm

import random
from pathlib import Path
import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

import torch.nn.functional as F

from scipy.ndimage import label as cc_label


from LumenGSLAM.utils.general import my_logger
from LumenGSLAM.utils.training.optimizer import GaussiansOptimizer
from LumenGSLAM.utils.training.losses import TrackingLoss, MappingLoss, MappingLossPBR, GeometricProjLoss, TrackingLossHybrid
from LumenGSLAM.utils.data_process.outlier_handling import energy_mask
from LumenGSLAM.utils.training.keyframe_selection import (keyframe_selection_overlap, keyframe_selection_distance, keyframe_selection_distance_loss,
                                                          keyframe_selection_loss, keyframe_selection_loss_time, keyframe_selection_uniform_count)
from LumenGSLAM.utils.metrics import get_psnr_ssim_lpips
from LumenGSLAM.utils.visualization.plot import save_img
from LumenGSLAM.utils.data_process.geometry import transform_to_frame
from LumenGSLAM.utils.data_process.sh_utils import compute_pbr_color
from LumenGSLAM.utils.training.timer import TrainingTimer
from LumenGSLAM.utils.data_process.loading import load_first_pose
from LumenGSLAM.utils.data_process.pose_handling import rot2quat, quat2rot, pose_from_rot_trasl
from LumenGSLAM.utils.data_process.lie_algebra import xi2rotm

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


class Trainer:
    def __init__(self, camera, gaussian_population, loader, renderer, logger, tracking, device, train_config, do_log):
        self.camera = camera
        self.gaussians = gaussian_population
        self.loader = loader
        self.renderer = renderer
        self.logger = logger
        self.tracking = tracking
        self.device = device
        self.train_config = train_config

        self.timer = TrainingTimer()

        self.log = do_log


        if self.train_config['tracking']['loss_fn'] == 'geometric':
            # this is WRONG
            self.tracking_loss_fn = GeometricProjLoss()
        elif self.train_config['tracking']['loss_fn'] == 'photometric':
            self.tracking_loss_fn = TrackingLoss(train_config['tracking']['loss'])
        elif self.train_config['tracking']['loss_fn'] == 'hybrid':
            # this is WRONG
            self.tracking_loss_fn = TrackingLossHybrid(train_config['tracking']['loss'])
        else:
            raise NotImplementedError

        if self.train_config['do_pbr']:
            self.mapping_loss_fn = MappingLossPBR(train_config['mapping']['loss'])
        else:
            self.mapping_loss_fn = MappingLoss(train_config['mapping']['loss'])

        if self.loader.data[0][2] is not None:
            self.first_w2c = self.loader.data[0][2].to(self.device)
        else:
            self.first_w2c = torch.tensor(np.eye(4)).to(self.device)

        self.keyframe_list = []
        self.keyframe_count = defaultdict(int)
        self.keyframes_last_loss = {'losses': {}, 'last_seen': {}}

        # if we use isotropic is nosense to do normal "pancakeing"
        self.use_normal_loss = not self.gaussians.data_config['isotropy']

        self.done_map_iters = 0
        self.last_ckpt = None
        self.added_pts_mask = None
        self.current_keypoints = None # those are the (match_3D_0, match_2D_1) from superpoint

        self.current_frame = {}
        self.last_frame = {}

        # self.edge_poly = Polygon()
        self.max_depth = 100.0 # HARD-CODED for C3VDv2 - should parametrize (also in gaussian-model)

    def train_loop(self):

        # pbar is inside tracking and mapping loops
        for i, (gt_im, gt_depth, gt_pose) in enumerate(self.loader):

            if i == 0 or (i + 1) % self.train_config['use_every'] == 0:

                self.timer.start()
                my_logger.info(f'({self.loader.data_path.name}) frame {i}')

                gt_im = gt_im.to(self.device)
                gt_depth = gt_depth.to(self.device)
                if gt_pose is not None:
                    gt_pose = gt_pose.to(self.device)
                self.current_frame['id'] = i
                self.current_frame['gt_im'] = gt_im.detach().cpu()
                self.current_frame['gt_depth'] = gt_depth.detach().cpu()
                self.current_frame['gt_w2c'] = gt_pose.detach().cpu() if gt_pose is not None else None

                # first frame initialization
                if i == 0:
                    self.gaussians.init_model(gt_im, gt_depth, self.camera.k, w2c=gt_pose)

                #  TRACKING
                if self.tracking:
                    self.timer.tracking_start()

                    if i == 0:
                        self.camera.init_quat_trasl()
                    else:
                        with torch.no_grad():
                            # print(len(self.camera.w2c_traj))
                            current_w2c = self.camera.get_current_w2c
                            gaussians_params = self.gaussians.get_params
                            detatched_gaussians_params = {k: gaussians_params[k].detach() for k in gaussians_params.keys()}

                            if self.train_config['do_pbr']:
                                pbr_color = self.gaussians.compute_pbr_color(self.camera.light.detach(),
                                                                             torch.inverse(current_w2c)[:3, 3].detach(),
                                                                             return_pbr_params=False).detach()
                            else:
                                pbr_color = None

                            self.renderer.set_rendervar_im(detatched_gaussians_params, pbr_color=pbr_color,
                                                           retain_grad=False)

                            self.renderer.set_rendervar_depthplus_silhouette(detatched_gaussians_params, current_w2c)
                            r_depth, r_sil, _ = self.renderer.render_depth_silhouette(self.camera.get_rasterization_settings(current_w2c, self.gaussians.active_sh_degree))
                            r_im, _ = self.renderer.render_image(self.camera.get_rasterization_settings(current_w2c, self.gaussians.active_sh_degree))

                            self.last_frame['r_im'] = r_im.detach().cpu().clip(0,1)
                            self.last_frame['r_depth'] = r_depth.detach().cpu()
                            self.last_frame['r_sil'] = r_sil.detach().cpu().clip(0,1)

                        # pose estimation
                        quat, tran, w2c_init, self.current_keypoints, success = self.camera.init_next_frame_pose(self.last_frame, self.current_frame, self.train_config['tracking']['init_pose_mode'],
                                                                                    self.loader, (r_im.detach(), r_depth.detach()), gt_w2c=gt_pose, depth_max=self.max_depth,
                                                                                        return_keypoints=((self.train_config['tracking']['loss_fn'] == 'geometric' )|(self.train_config['tracking']['loss_fn'] == 'hybrid')))

                        # checking if tracking has been succesfull
                        if not success:
                            self.camera.set_w2c_from_best_quat_transl(quat, tran)
                            my_logger.info('   - Pose initialised as last one and skipping mapping!')
                            continue

                        # print(current_w2c, w2c_init)
                        if self.train_config['do_phba'] not in ["after_tracking", "after_expansion", "after_mapping"]:
                            self.camera.set_w2c_from_best_quat_transl(quat, tran)

                        if self.train_config['do_phba'] == 'after_tracking':
                            if self.train_config['tracking']['loss_fn']  == 'geometric':
                                quat, tran = self.geometric_pose_refinement(gt_im, gt_depth, i, w2c_init=w2c_init, gt_w2c=gt_pose)
                            elif self.train_config['tracking']['loss_fn'] in ['photometric', 'hybrid']:
                                quat, tran = self.phba_loop(gt_im, gt_depth, i, gt_w2c=gt_pose)
                            else:
                                raise NotImplementedError

                            self.camera.set_w2c_from_best_quat_transl(quat, tran)


                    self.timer.tracking_end()
                    self.tracking_loss_fn.reset()

                else:
                    self.camera.set_w2c(gt_pose)

                self.current_frame['r_w2c'] = self.camera.get_current_w2c.detach().cpu()



                # MAPPING
                self.timer.mapping_start()
                if i == 0 or (i + 1) % self.train_config['map_every'] == 0:
                    self.done_map_iters += 1

                    current_w2c = self.camera.get_current_w2c


                    gaussians_params = self.gaussians.get_params
                    gaussians_vars = self.gaussians.get_variables

                    curr_data = {'im': gt_im, 'depth': gt_depth, 'w2c': current_w2c, 'id': i}

                    pre_pts = gaussians_params['means3D'].shape[0]
                    post_pts = self.gaussians.params['means3D'].shape[0]
                    # progressive growth
                    if i > 0:

                        detatched_gaussians_params = {k: gaussians_params[k].detach() for k in gaussians_params.keys()}

                        if self.train_config['do_pbr']:
                            pbr_color = self.gaussians.compute_pbr_color(self.camera.light.detach(), torch.inverse(current_w2c)[:3, 3].detach(), return_pbr_params=False).detach()
                        else:
                            pbr_color = None

                        self.renderer.set_rendervar_im(detatched_gaussians_params, pbr_color=pbr_color, retain_grad=False)


                        self.renderer.set_rendervar_depthplus_silhouette(detatched_gaussians_params, current_w2c)
                        depth_sil = self.renderer.render_depth_silhouette(self.camera.get_rasterization_settings(current_w2c, self.gaussians.active_sh_degree))

                        self.gaussians.add_new_gaussians(curr_data, self.camera.k, sil_thres=self.train_config['mapping']['sil_add_thresh'], depth_sil=depth_sil, frame_id=i, logger=self.logger)
                        post_pts = self.gaussians.params['means3D'].shape[0]

                        if post_pts-pre_pts >= 5000:
                            self.gaussians.reset_opacity_scheduler()

                    self.logger.after_growth(i, post_pts-pre_pts)

                    if self.train_config['do_phba'] == 'after_expansion' and self.tracking and i > 0:
                        quat, rot = self.phba_loop(gt_im, gt_depth, i)
                        self.camera.set_w2c_from_best_quat_transl(quat, rot)
                        self.tracking_loss_fn.reset()

                    # proper mapping
                    self.mapping_loop(curr_data, i, gaussians_params, gaussians_vars)


                    # mapping time
                    self.timer.mapping_end()

                    # logging
                    self.logger.mapping_end(self.gaussians.params['means3D'].shape[0])
                    self.mapping_loss_fn.reset()

                    if self.train_config['do_phba'] == 'after_mapping' and self.tracking and i>0:
                        quat, rot = self.phba_loop(gt_im, gt_depth, i)
                        self.camera.set_w2c_from_best_quat_transl(quat, rot)
                        self.tracking_loss_fn.reset()

                # CHECKPOINT
                if (i+1) % self.train_config['checkpoint_every'] == 0 and self.train_config['checkpoint_every'] != -1:
                    self.save_checkpoint(i)

                # adding keyframes for mapping
                if (i == 0) or ((self.done_map_iters+1) % self.train_config['keyframe_every'] == 0):
                    my_logger.info(f'   - frame {i} added to keyframe list')
                    self.keyframe_list.append({'id': i, 'w2c': self.camera.get_current_w2c, 'im': gt_im, 'depth': gt_depth})

                    if self.train_config['keyframes']['remove_count'] != -1:
                        filtered_list = [frame for frame in self.keyframe_list if self.keyframe_count.get(frame['id'], 0) <= self.train_config['keyframes']['remove_count']]
                        self.keyframe_list = filtered_list

                    if self.train_config['keyframes']['list_length'] != -1:
                        self.keyframe_list = self.keyframe_list[-self.train_config['keyframes']['list_length']:]

                self.timer.frame_end()

                self.last_frame = copy.deepcopy(self.current_frame)
                self.current_frame = {}
                torch.cuda.empty_cache()

    # this one works ONLY if self.tracking_loss_fn == GeometryProjLoss
    def geometric_pose_refinement(self, current_frame, current_depth, frame_id, w2c_init, gt_w2c):
        num_iter = self.train_config['tracking']['num_iter']

        gaussians_params = self.gaussians.get_params
        gaussians_vars = self.gaussians.get_variables
        current_min_loss = float(1e20)

        if not self.train_config['tracking']['use_lie_algebra']:
            optimizer = GaussiansOptimizer(self.train_config['tracking']['lrs'], self.camera.get_current_quat_transl, device=self.device)

        else:
            optimizer = GaussiansOptimizer(self.train_config['tracking']['lrs'], self.camera.delta_cam, device=self.device)


        pbar = tqdm(range(num_iter), total=num_iter, desc=f"   - POSE REFINEMENT (geometric)", bar_format=TQDM_BAR_FORMAT)

        for i in pbar:
            optimizer.opt.zero_grad()
            # iter_start_time = time.time()
            detatched_gaussians_params = {k: gaussians_params[k].detach() for k in gaussians_params.keys()}

            if not self.train_config['tracking']['use_lie_algebra']:
                w2c = self.camera.get_w2c_for_optimization()
            else:
                w2c = xi2rotm(self.camera.delta_cam['cam_trans_delta'], self.camera.delta_cam['cam_rot_delta'], w2c_init)

            rasterization_settings = self.camera.get_rasterization_settings(w2c.detach(), self.gaussians.active_sh_degree)

            with torch.no_grad():
                self.renderer.set_rendervar_im(detatched_gaussians_params, retain_grad=False)
                r_im, radius = self.renderer.render_image(rasterization_settings, gaussians_vars)
                self.renderer.set_rendervar_depthplus_silhouette(detatched_gaussians_params, self.first_w2c)
                r_depth, r_sil, r_depth_sq = self.renderer.render_depth_silhouette(rasterization_settings)

            loss = self.compute_loss({'im': current_frame, 'depth': current_depth, 'id': frame_id, 'w2c': w2c}, r_im, r_depth,
                                     r_sil, None, 0.99, mode='tracking')

            loss.backward()

            with torch.no_grad():
                # Save the best candidate rotation & translation

                candidate_cam_unnorm_rot, candidate_cam_tran = rot2quat(w2c[:3,:3].unsqueeze(0)), w2c[:3,3].unsqueeze(0)

                current_loss = self.tracking_loss_fn.current_dict

                ate = ((torch.linalg.inv(gt_w2c)[:3, 3] - torch.linalg.inv(w2c.detach())[:3, 3])**2).mean().detach()

                self.logger.tracking_iter_end(frame_id, current_loss, ate.detach().cpu().item())

                if loss < current_min_loss:
                    best_idx = i
                    current_min_loss = loss.item()
                    best_cam_tran = candidate_cam_tran.clone().detach()
                    best_cam_unnorm_rot = candidate_cam_unnorm_rot.clone().detach()

                    pbar.set_description(f"   - POSE REFINEMENT (geometric) -- loss: {loss.item(): .3f}; ATE: {ate: .4f}; best_iter: {best_idx}")

            optimizer.do_step()

            seen = radius > 0
            gaussians_vars['max_2D_radius'][seen] = torch.max(radius[seen], gaussians_vars['max_2D_radius'][seen])
            gaussians_vars['seen'] = seen

            self.timer.tracking_iter_end()

            self.camera.reset_deltas() # resets lie rho and phi

        return best_cam_unnorm_rot, best_cam_tran

    def phba_loop(self, current_frame, current_depth, frame_id, gt_w2c, sil_thresh=0.99):

        num_iter = self.train_config['tracking']['num_iter']

        gaussians_params = self.gaussians.get_params
        gaussians_vars = self.gaussians.get_variables
        current_min_loss = float(1e20)
        best_idx = 0


        optimizer = GaussiansOptimizer(self.train_config['tracking']['lrs'], self.camera.get_current_quat_transl, device=self.device)

        camera_params = self.camera.get_current_quat_transl

        # init vars for best pose
        best_cam_unnorm_rot = camera_params['cam_unnorm_rots'].detach().clone()
        best_cam_tran = camera_params['cam_trans'].detach().clone()


        pbar = tqdm(range(num_iter), total=num_iter, desc=f"   - POSE REFINEMENT", bar_format=TQDM_BAR_FORMAT)
        for i in pbar:
            optimizer.opt.zero_grad()
            # iter_start_time = time.time()

            # I transform everything to camera frame so to track grad for it
            detatched_gaussians_params = {k: gaussians_params[k].detach() for k in gaussians_params.keys()}


            if self.train_config['do_pbr']:
                pbr_color = self.gaussians.compute_pbr_color(self.camera.light.detach(), torch.linalg.inv(self.camera.get_w2c_for_optimization())[:3, 3].detach(), diffuse_only=self.train_config['tracking']['ignore_reflexes'])
                pbr_color = pbr_color.detach()
            else:
                pbr_color = None

            # copied_params = copy.deepcopy(gaussians_params)
            if self.train_config['tracking']['pre_rot_cov']:
                detatched_gaussians_params['means3D'], detatched_gaussians_params['cov3D_precomp'] = transform_to_frame(self.gaussians, self.camera.get_w2c_for_optimization() )
            else:
                detatched_gaussians_params['means3D'] = transform_to_frame(self.gaussians, self.camera.get_w2c_for_optimization(), False)

            # detatched_gaussians_params['means3D'] = transform_to_frame(self.gaussians, self.camera)
            # detatched_gaussians_params['means3D'] = transform_to_frame(self.gaussians, self.camera)

            self.renderer.set_rendervar_im(detatched_gaussians_params, pbr_color=pbr_color, retain_grad=False)

            rasterization_settings = self.camera.get_rasterization_settings(self.first_w2c, self.gaussians.active_sh_degree)


            # EndoGSLAM rendering
            r_im, radius = self.renderer.render_image(rasterization_settings, gaussians_vars)
            self.renderer.set_rendervar_depthplus_silhouette(detatched_gaussians_params, self.first_w2c)
            r_depth, r_sil, r_depth_sq = self.renderer.render_depth_silhouette(rasterization_settings)

            if self.use_normal_loss:
                self.renderer.set_rendervar_normals(detatched_gaussians_params, self.gaussians.get_unnorms_norms, self.first_w2c)
                r_normals = self.renderer.render_normals(rasterization_settings)
            else:
                r_normals = None


            loss = self.compute_loss({'im': current_frame, 'depth': current_depth, 'id': frame_id, 'w2c': self.camera.get_w2c_for_optimization()}, r_im, r_depth, r_sil, r_normals, sil_thresh, mode='tracking')

            loss.backward()

            with torch.no_grad():
                # Save the best candidate rotation & translation

                candidate_params = self.camera.get_current_quat_transl
                candidate_cam_unnorm_rot = candidate_params['cam_unnorm_rots'].detach().clone()
                candidate_cam_tran = candidate_params['cam_trans'].detach().clone()
                candidate_w2c = torch.tensor(np.eye(4))
                candidate_w2c[:3,:3] = quat2rot((F.normalize(candidate_cam_unnorm_rot)))
                candidate_w2c[:3, 3] = candidate_cam_tran

                candidate_w2c = candidate_w2c.to(self.device)

                current_loss = self.tracking_loss_fn.current_dict

                ate = ((torch.linalg.inv(gt_w2c)[:3, 3] - torch.linalg.inv(candidate_w2c)[:3, 3])**2).mean().detach()
                self.logger.tracking_iter_end(frame_id, current_loss, ate.detach().cpu().item())

                if loss < current_min_loss:
                    best_idx = i
                    current_min_loss = loss.item()
                    best_cam_tran = candidate_cam_tran.clone().detach()
                    best_cam_unnorm_rot = candidate_cam_unnorm_rot.clone().detach()

                pbar.set_description(f"   - POSE REFINEMENT ({self.train_config['tracking']['loss_fn']}) -- loss: {loss.item(): .3f}; ATE: {ate: .3f}; best_iter: {best_idx}")

            optimizer.do_step()

            seen = radius > 0
            gaussians_vars['max_2D_radius'][seen] = torch.max(radius[seen], gaussians_vars['max_2D_radius'][seen])
            gaussians_vars['seen'] = seen

            self.timer.tracking_iter_end()

        return best_cam_unnorm_rot, best_cam_tran

    def mapping_loop(self, curr_data, frame_id, gaussians_params, gaussians_vars, sil_thresh=0.99):

        # sil-thresh is actually unsued here

        num_iter = self.train_config['mapping']['num_iter']

        # ensures current framee is in here even if it is never seen....
        self.keyframes_last_loss['losses'][frame_id] = 0
        self.keyframes_last_loss['last_seen'][frame_id] = frame_id

        # sampling keyframes
        keyframe_idxs = self.sample_keyframes(frame_id, curr_data['depth'], mode=self.train_config['keyframes']['selection_mode'])

        used_frames = np.unique([self.keyframe_list[x]['id'] if x != 'curr' else curr_data['id'] for x in keyframe_idxs])

        self.logger.mapping_start([self.keyframe_list[x]['id'] for x in keyframe_idxs if x != 'curr' ])
        opt_params = {}
        opt_lrs = self.train_config['mapping']['lrs']
        if self.train_config['bundle_adjustment']  and frame_id>0:
            opt_lrs.update(self.train_config['tracking']['lrs'])
            cameras = self.camera.quat_trasl_traj
            opt_params.update(gaussians_params)
            for i, cam in enumerate(cameras):
                if i != 0:
                    opt_params.update({k+f'__{i}': cam[k] for k in cam.keys()})
        else:
            opt_params = gaussians_params

        if self.train_config['do_pbr']:
            optimizer = GaussiansOptimizer(opt_lrs, opt_params, device=self.device, light=self.camera.light)
        else:
            optimizer = GaussiansOptimizer(opt_lrs, opt_params, device=self.device)


        my_logger.info(f'   - selected keyframes (uniques):  [{" ".join(map(str, used_frames))}]')
        pbar = tqdm(range(num_iter), total=num_iter, desc=f"frame {frame_id} - MAPPING -- N° points: {gaussians_params['means3D'].shape[0]}", bar_format=TQDM_BAR_FORMAT)

        # debug_dict = {'silhouette': [], 'r_depth': [], 'r_im': []}

        for i in pbar:

            optimizer.opt.zero_grad()

            # keyframe selection
            curr_keyframe_id = keyframe_idxs[i]
            if curr_keyframe_id == 'curr':
                iter_data = curr_data
                trj_n = -1
            else:
                iter_data = self.keyframe_list[curr_keyframe_id]
                trj_n = curr_keyframe_id

            self.keyframe_count[iter_data['id']] += 1


            if self.train_config['tracking']['pre_rot_cov'] and self.train_config['bundle_adjustment']  and frame_id>0:
                w2c = self.camera.get_w2c_for_optimization(trj_n)
                gaussians_params['means3D'], cov_prerot = transform_to_frame(self.gaussians, w2c, detach=False)
                current_w2c = self.first_w2c
                loss_mode = 'bundle_adjustment'
                if self.train_config['do_pbr']:
                    pbr_color, pbr_params = self.gaussians.compute_pbr_color(self.camera.light, torch.inverse(w2c)[:3, 3].detach(),
                                                                             return_pbr_params=True)
                else:
                    pbr_color, pbr_params = None, None
            else:
                cov_prerot = None
                loss_mode = 'mapping'
                current_w2c = iter_data['w2c']
                if self.train_config['do_pbr']:
                    pbr_color, pbr_params = self.gaussians.compute_pbr_color(self.camera.light, torch.inverse(current_w2c)[:3, 3].detach(),
                                                                             return_pbr_params=True)
                else:
                    pbr_color, pbr_params = None, None


            rasterization_settings = self.camera.get_rasterization_settings(current_w2c, self.gaussians.active_sh_degree)
            self.renderer.set_rendervar_im(gaussians_params, pbr_color=pbr_color)

            if self.use_normal_loss:
                self.renderer.set_rendervar_normals(gaussians_params, self.gaussians.get_unnorms_norms, current_w2c)
                r_normals = self.renderer.render_normals(rasterization_settings)
            else:
                r_normals = None


            r_im, radius = self.renderer.render_image(rasterization_settings, gaussians_vars)
            self.renderer.set_rendervar_depthplus_silhouette(gaussians_params, current_w2c, cov_prerot)
            r_depth, r_sil, r_depth_sq = self.renderer.render_depth_silhouette(rasterization_settings)

            if self.train_config['plot_progress']:
                # print(self.done_map_iters)
                if (self.done_map_iters-1) % 25 == 0:
                    # print('loggg')
                    self.logger.progress_plot(frame_id, i, iter_data, r_im, r_depth, r_normals)

            loss = self.compute_loss(iter_data, r_im, r_depth, r_sil, r_normals, sil_thresh, mode=loss_mode, pbr_params=pbr_params, seen=radius > 0)
            loss.backward()

            with torch.no_grad():

                # storing which parameters the current view sees
                self.gaussians.set_seen(radius)

                # densification / pruning
                self.gaussians.densify_prune(optimizer, i, frame_id, current_w2c, self.camera, self.logger)

                # update
                optimizer.do_step()

                if self.train_config['cull_outside_depth']:

                    keep_mask = self.cull_outside_valid_depth(iter_data['depth'], current_w2c, self.camera.k, True)

                    self.gaussians.params, self.gaussians.variables = optimizer.prune((~keep_mask), gaussians_params, gaussians_vars)

                if self.train_config['bundle_adjustment'] and frame_id > 0:
                    self.camera.w2c_traj[trj_n] = pose_from_rot_trasl(
                        self.camera.quat_trasl_traj[trj_n]['cam_unnorm_rots'],
                                                                                self.camera.quat_trasl_traj[trj_n]['cam_trans']).detach()
                    gaussians_params['means3D'] = gaussians_params['means3D'].detach().requires_grad_(True)

                self.timer.mapping_iter_end()

                current_loss = self.mapping_loss_fn.current_dict
                self.keyframes_last_loss['losses'][iter_data['id']] = current_loss['loss']
                self.keyframes_last_loss['last_seen'][iter_data['id']] = frame_id

                self.logger.iter_end(iter_data['id'], r_im, iter_data['im'], r_depth, iter_data['depth'], current_loss, self.log)

                if self.log:
                    current_metrics = self.logger.get_current_metrics_map()

                    pbar.set_description(f"   - MAPPING -- N° points: {gaussians_params['means3D'].shape[0]}; iter frame: {iter_data['id']}; iter time {self.timer.get_mapping_single_iter_time(i+1): .2f}; "
                                         f"loss: {current_loss['loss']: .3f}; PSNR: {current_metrics['psnr']: .3f}, SSIM: {current_metrics['ssim']: .3f}, "
                                         f"M_SSIM: {current_metrics['mssim']: .3f}, LPIPS: {current_metrics['lpips']: .3f}")
                else:
                    pbar.set_description(f"   - MAPPING -- N° points: {gaussians_params['means3D'].shape[0]}; iter frame: {iter_data['id']}; iter time {self.timer.get_mapping_single_iter_time(i+1): .2f}; "
                                         f"loss: {current_loss['loss']: .3f}")


    def compute_loss(self, iter_data, r_im, r_depth, r_sil, r_norm, sil_thresh, mode, pbr_params=None, seen=None):
        # silhouette should just be the cumulated alpha at every pixel location (they have an option not to use it for loss)

        # uncertainty = r_depth_sq - r_depth ** 2
        # uncertainty = uncertainty.detach()

        gt_im, gt_depth = iter_data['im'], iter_data['depth']

        nan_mask = (~torch.isnan(r_depth)) # & (~torch.isnan(uncertainty))

        valid_depth_mask = (gt_depth > 0) & (gt_depth < self.max_depth)
        mask = valid_depth_mask & nan_mask
        mask.detach()

        if mode == 'tracking':

            bg_mask = energy_mask(gt_im) & energy_mask(r_im)

            presence_sil_mask = (r_sil > sil_thresh)

            mask = mask & presence_sil_mask & bg_mask
            mask.detach()


            if self.train_config['tracking']['loss_fn'] == 'geometric':
                match_3D_0, match_2D_1 = self.current_keypoints
                loss = self.tracking_loss_fn(match_3D_0, match_2D_1, r_depth, iter_data['w2c'], self.camera.k)
            elif self.train_config['tracking']['loss_fn'] == 'photometric':
                loss = self.tracking_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), mask=mask, use_weight=self.train_config['tracking']['weighted_loss'])
            elif self.train_config['tracking']['loss_fn'] == 'hybrid':
                match_3D_0, match_2D_1 = self.current_keypoints
                loss = self.tracking_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), (match_3D_0, match_2D_1, iter_data['w2c'], self.camera.k), mask=mask, use_weight=self.train_config['tracking']['weighted_loss'])

        elif mode == 'mapping':
            if not self.train_config['do_pbr']:
                loss = self.mapping_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), mask)
            else:

                loss = self.mapping_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), pbr_params, mask)
        elif mode == 'bundle_adjustment':
            if not self.train_config['do_pbr']:
                loss_map = self.mapping_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), mask)
            else:
                loss_map = self.mapping_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), pbr_params, mask)
            loss_track = self.tracking_loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), mask=mask, use_weight=self.train_config['tracking']['weighted_loss'])
            loss = loss_map + loss_track

        else:
            raise NotImplementedError

        return loss


    def sample_keyframes(self, time_idx, current_gt_depth, mode='distance'):

        # N.B. sampled_ids refers to keyframe list, the current frame has idx == 'curr'
        if len(self.keyframe_list) > 0:

            if mode == 'distance':
                sampled_ids = keyframe_selection_distance(time_idx, self.camera.get_current_position, self.keyframe_list, self.train_config['keyframes']['current_frame_prob'], self.train_config['mapping']['num_iter'])
            elif mode == 'loss':
                sampled_ids = keyframe_selection_loss(self.keyframe_list, self.keyframes_last_loss['losses'], self.train_config['keyframes']['current_frame_prob'], self.train_config['mapping']['num_iter'], tau=0.1)
            elif mode == 'distance_loss':
                # print(list(keyframes_last_loss.keys()))
                sampled_ids = keyframe_selection_distance_loss(time_idx, self.camera.get_current_position, self.keyframe_list, self.keyframes_last_loss['losses'],self.train_config['keyframes']['current_frame_prob'], self.train_config['mapping']['num_iter'])
            elif mode == 'time_loss':
                sampled_ids = keyframe_selection_loss_time(self.keyframe_list, self.keyframes_last_loss['losses'], self.train_config['keyframes']['current_frame_prob'], self.train_config['mapping']['num_iter'], time_idx, tau=0.1, loss_weight=0.5)
            elif mode == 'uniform':
                sampled_ids = keyframe_selection_uniform_count(self.keyframe_list, self.keyframe_count, self.train_config['keyframes']['current_frame_prob'], self.train_config['mapping']['num_iter'], tau=2)

            elif mode == 'overlap':
                overlapping_keyframes = keyframe_selection_overlap(current_gt_depth, self.camera.get_current_w2c, self.camera.k, self.keyframe_list, 100, min_percentage=0.15) # they used min_percentage=0.0
                # adding current frame
                overlapping_keyframes.append('curr')

                # setting higher prob to current frame and a constant one for others
                probabilities = [(1 - self.train_config['keyframes']['current_frame_prob']) / (len(overlapping_keyframes) - 1)] * (len(overlapping_keyframes) - 1) + [self.train_config['keyframes']['current_frame_prob']]

                # Sampling overlapping keyframes with probabilities
                sampled_ids = random.choices(overlapping_keyframes, weights=probabilities, k=self.train_config['mapping']['num_iter'])
            elif mode == 'random':
                curr_n = 5
                idxs = [i for i in range(len(self.keyframe_list))]
                sampled_ids = list(random.choices(idxs, k=self.train_config['mapping']['num_iter']-curr_n))
                sampled_ids += ['curr'] * curr_n
                random.shuffle(sampled_ids)
            else:
                raise NotImplementedError

        else:
            # if keyframe list is empty (first frames) just use the current
            sampled_ids = ['curr'] * self.train_config['mapping']['num_iter']

        return sampled_ids


    def cull_outside_valid_depth(self, depth_map, w2c, intrinsics, return_index=False):
        """
        considers depthmap as a connected component and retrieves indexes of points outside of it.
        To be used ONLY if depthmap are provided only for certain views and we want to cull non
        depth-regularized points

        Args:
        -- point_cloud (torch.Tensor): (N, 3) world points.
        -- depth_map (torch.Tensor): (H, W) depth map.
        -- w2c (torch.Tensor): (4, 4) world-to-camera transformation.
        -- intrinsics (torch.Tensor): (3, 3) camera intrinsics.
        -- return_index (bool): if True, return ONLY index of points outside of valid depth map.

        Returns:
        -- torch.Tensor: (M, 3) filtered points.
        """
        with torch.no_grad():
            params = self.gaussians.get_params#.clone()
            vars = self.gaussians.get_variables#.clone()

            point_cloud = params['means3D'].clone()

            _, H, W = depth_map.shape
            N = point_cloud.shape[0]

            # Compute depth mask on CPU
            depth_cpu = depth_map.cpu().squeeze().numpy()
            valid_mask = (depth_cpu > 0).astype(np.uint8)

            labeled, num = cc_label(valid_mask)

            # Pick largest component
            counts = np.bincount(labeled.flat)
            counts[0] = 0  # background
            largest = np.argmax(counts)
            main_region = (labeled == largest)  # bool (H, W)
            main_region_mask = torch.from_numpy(main_region).to(self.device)# (H,W), bool

            ones = torch.ones((N, 1), device=self.device)
            homo = torch.cat([point_cloud, ones], dim=1)
            cam = (w2c @ homo.T).T[:, :3]

            z = cam[:, 2]


            # Project to 2D
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
            u = torch.round((cam[:, 0] * fx) / z + cx).long()
            v = torch.round((cam[:, 1] * fy) / z + cy).long()

            # Inside image bounds
            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)  # shape N


            inside_main = torch.zeros(N, dtype=torch.bool, device=self.device)
            # Get indices of valid projections
            projectable_idx = torch.nonzero(in_bounds, as_tuple=False).squeeze(1).to(self.device)
            # Only these indices have valid u,v
            if projectable_idx.numel() > 0:
                v_in = v[projectable_idx]
                u_in = u[projectable_idx]
            inside_main[projectable_idx] = main_region_mask[v_in, u_in]

            keep_mask = (inside_main | (~in_bounds))

        if return_index:
            return keep_mask


        culled_params = {k: v[keep_mask].detach().requires_grad_(True) if v.shape[0] == N else v for k,v in params.items()}
        culled_vars = {k: v[keep_mask] if k not in ["seen", "means2D", "scene_radius"] else v for k, v in  vars.items()}

        self.gaussians.params = culled_params
        self.gaussians.variables = culled_vars



    def save_checkpoint(self, frame_id):
        # saving final parameters with ref frame
        params = copy.deepcopy(self.gaussians.get_params)

        data_path = self.loader.data_path
        dst = self.logger.dst_path
        train_config = self.train_config

        if train_config['do_pbr']:
            params['F0'] = params['F0'].clamp(min=0)
            params['light'] = self.camera.light.detach()

        params['loss'] = {'reconstruction_gain': train_config['mapping']['loss']['reconstruction_gain'],
                          'depth_gain': train_config['mapping']['loss']['depth_gain'],
                          'norm_gain': train_config['mapping']['loss']['norm_gain'],
                          'mask_photometric_loss': train_config['mapping']['loss']['mask_photometric_loss']}

        normals = torch.nn.functional.normalize(self.gaussians.get_unnorms_norms).detach()
        params['normals'] = normals

        my_logger.info(f'saving checkpoint!')
        if os.path.isfile(Path(data_path) / self.loader.pose_file_name):
            first_c2w = load_first_pose(Path(data_path) / self.loader.pose_file_name)
        else:
            first_c2w = torch.tensor(np.eye(4))

        saveing_dict = {'gaussians': params, 'first_c2w': first_c2w}

        if self.tracking:
            saveing_dict['tracked_trj'] = copy.deepcopy(self.camera.w2c_traj)
        else:
            saveing_dict['tracked_trj'] = None

        if self.last_ckpt is not None:
            os.remove(Path(dst) / self.last_ckpt)

        if frame_id != "final":
            torch.save(saveing_dict, Path(dst) / f'parameters_{frame_id}.pt')

            self.last_ckpt = f'parameters_{frame_id}.pt'
        else:
            torch.save(saveing_dict, Path(dst) / f'parameters.pt')

    def save_keyframe_stats(self):

        dst = self.logger.dst_path

        # saving the count with all time each keyframe has been seen
        keyframe_count = {'frame_id': list(self.keyframe_count.keys()),
                          'count': list(self.keyframe_count.values())}
        df = pd.DataFrame(keyframe_count)
        df.to_csv(dst / 'keyframe_count.csv', index=False)

        keyframes_last_loss = defaultdict(list)
        for i, ((k, l), n) in enumerate(
                zip(self.keyframes_last_loss['losses'].items(), self.keyframes_last_loss['last_seen'].values())):
            keyframes_last_loss['frame id'].append(k)
            keyframes_last_loss['loss'].append(l)
            keyframes_last_loss['last seen'].append(n)

        df = pd.DataFrame(keyframes_last_loss)
        df.to_csv(dst / 'keyframe_loss.csv', index=False)

class Evaluer:
    def __init__(self, camera, gaussian_params, loader, renderer, device, saving_path, save_images=True):
        self.camera = camera
        self.gaussians_params = gaussian_params
        self.loader = loader
        self.renderer = renderer
        self.device = device

        # if we use isotropic is nosense to do normal "pancakeing"
        self.save_normals = bool('normals' in gaussian_params)

        self.sh_degree = 3 if 'feature_rest' in gaussian_params.keys() else 0

        self.dst_path = Path(saving_path)

        self.save_imgs = save_images

        self.plot_dst, self.image_dst, self.depth_dst, self.normals_dst = self.create_dst_paths()

        self.loss_fn = MappingLoss(gaussian_params['loss'])

        my_logger.info(f'saving to {self.dst_path}')
        self.metrics_dict = dict(frame_id=[], psnr=[], depth_rmse=[], depth_l1=[], lpips=[], ssim=[], mssim=[],
                                 loss=[], reconstruction_loss=[], depth_loss=[], normal_loss=[])

        self.debug_pbr = None #DebugLoggerPBR(self.dst_path)


    def eval_save(self):

        my_logger.info("Evaluating Final Parameters...")
        pbar = tqdm(enumerate(self.loader), total=len(self.loader), desc="evaluation",
                    bar_format=TQDM_BAR_FORMAT)
        with torch.no_grad():
            for i, (color, depth, w2c) in pbar:
                color = color.to(self.device)
                depth = depth.to(self.device)
                w2c = w2c.to(self.device)

                self.metrics_dict['frame_id'].append(f'{i:04d}')

                if 'light' in self.gaussians_params.keys():
                    pbr_color = compute_pbr_color(self.gaussians_params, torch.inverse(w2c)[:3, 3], debug_logger=self.debug_pbr, frame_id=i)
                else:
                    pbr_color = None

                self.renderer.set_rendervar_im(self.gaussians_params, retain_grad=False, pbr_color=pbr_color)
                self.renderer.set_rendervar_depthplus_silhouette(self.gaussians_params, w2c)

                # # rendering
                rasterization_settings = self.camera.get_rasterization_settings(w2c, self.sh_degree)
                r_im, radius = self.renderer.render_image(rasterization_settings)

                r_depth, r_sil, r_depth_sq = self.renderer.render_depth_silhouette(rasterization_settings)


                if self.save_normals:

                    normal = self.gaussians_params['normals']

                    # # Normalize the input vectors
                    N = torch.nn.functional.normalize(normal, dim=1)

                    self.renderer.set_rendervar_normals(self.gaussians_params, N, w2c)
                    r_normals = self.renderer.render_normals(rasterization_settings)
                else:
                    r_normals = None

                r_im = torch.clamp(r_im, 0, 1)
                self.compute_metrics(r_im, color, r_depth, depth)
                self.compute_loss(color, depth, r_im, r_depth, r_normals)

                pbar.set_description(f"evaluation - psnr: {np.mean(self.metrics_dict['psnr'])}, ssim: {np.mean(self.metrics_dict['ssim'])}, "
                                     f"lpips: {np.mean(self.metrics_dict['lpips'])}, mssim: {np.mean(self.metrics_dict['mssim'])}, "
                                     f"loss: {np.mean(self.metrics_dict['loss'])}")

                if self.save_imgs:
                    self.save_renders(color, r_im, r_depth, r_normals, i)

        # saving metrics to csv and box plots
        # self.debug_pbr.save()
        self.save_csv()
        # self.save_boxplots()
        my_logger.info(f'results saved to {self.dst_path}')


    def compute_loss(self, gt_im, gt_depth, r_im, r_depth, r_norm):
        nan_mask = (~torch.isnan(r_depth))
        bg_mask = energy_mask(gt_im)
        valid_depth_mask = (gt_depth > 0)
        mask = valid_depth_mask & nan_mask & bg_mask
        mask.detach()

        _ = self.loss_fn((r_im, r_depth, r_norm), (gt_im, gt_depth), mask)

        loss =  self.loss_fn.current_dict

        self.metrics_dict['loss'].append(loss['loss'])
        self.metrics_dict['reconstruction_loss'].append(loss['reconstruction_loss'])
        self.metrics_dict['depth_loss'].append(loss['depth_loss'])
        self.metrics_dict['normal_loss'].append(loss['normals_loss'])

    def compute_metrics(self, r_im, gt_im, rastered_depth, depth):

        # Mask invalid depth in GT
        valid_depth_mask = (depth > 0) & (depth < 1e10)
        rastered_depth = rastered_depth * valid_depth_mask
        weighted_im = r_im * valid_depth_mask
        weighted_gt_im = gt_im * valid_depth_mask

        psnr, ssim, lpips, mssim = get_psnr_ssim_lpips(weighted_im, weighted_gt_im)

        self.metrics_dict['psnr'].append(psnr.cpu().item())
        self.metrics_dict['ssim'].append(ssim.cpu().item())
        self.metrics_dict['mssim'].append(mssim.cpu().item())
        self.metrics_dict['lpips'].append(lpips)

        # depth metrics
        diff_depth_rmse = torch.sqrt((((rastered_depth - depth)) ** 2))
        diff_depth_rmse = diff_depth_rmse * valid_depth_mask

        rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()

        diff_depth_l1 = torch.abs((rastered_depth - depth))
        diff_depth_l1 = diff_depth_l1 * valid_depth_mask
        depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        self.metrics_dict['depth_rmse'].append(rmse.cpu().numpy())
        self.metrics_dict['depth_l1'].append(depth_l1.cpu().numpy())

    def save_renders(self, gt_im, r_im, r_depth, r_normals, frame_id):
        # Save Rendered RGB and Depth
        viz_render_im = r_im.detach().cpu().permute(1, 2, 0).numpy()



        save_im = np.clip(viz_render_im * 255, 0, 255).astype(np.uint8)
        save_img(save_im, (os.path.join(self.image_dst, "color_{:04d}.png".format(frame_id))))


        # UNCOMMENT TO SAVE DEPTHS
        # viz_render_depth = r_depth[0].detach().cpu().numpy()
        # save_depth = np.uint16(np.clip(viz_render_depth * 655.35, 0, 65535))
        # save_img(save_depth, (os.path.join(self.depth_dst, "depth_{:04d}.tiff".format(frame_id))))
        # if self.save_normals:
        #     n = r_normals.detach().cpu().permute(1,2,0).numpy()
        #     n = (min_max(n)*255).astype(np.uint8)
        #     save_img(n, self.normals_dst / 'normal_{:04d}.png'.format(frame_id))


    def save_csv(self, name='metrics.csv'):
        # saving metrics to csv
        df = pd.DataFrame(self.metrics_dict)
        df.to_csv(self.dst_path / name, index=False)

    def save_boxplots(self):

        dst = self.dst_path / 'boxplots'
        os.makedirs(dst, exist_ok=True)

        for key, values in self.metrics_dict.items():
            if key != 'frame_id':

                vals = np.array(values, dtype=np.float32)
                vals = vals[~np.isnan(vals)]

                # Compute statistics
                mean = np.mean(vals)
                median = np.median(vals)
                std = np.std(vals)
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = q75 - q25

                # Plot boxplot
                plt.figure(figsize=(19.2, 10.8))
                plt.boxplot(vals, vert=True)
                plt.title(f'{key}')
                plt.ylabel(key)

                # Legend with stats
                stats_text = (
                    f"Mean: {mean:.2f}\n"
                    f"Median: {median:.2f}\n"
                    f"Std: {std:.2f}\n"
                    f"IQR: {iqr:.2f}"
                )
                plt.legend([stats_text], loc='upper right', fontsize=15, frameon=True)

                # Save plot
                plt.savefig(dst / f'{key}', bbox_inches='tight', dpi=100)
                plt.close()

        dst = self.dst_path / 'over_frame_plots'
        os.makedirs(dst, exist_ok=True)

        for key, values in self.metrics_dict.items():
            if key != 'frame_id':

                vals = np.array(values, dtype=np.float32)
                vals = vals[~np.isnan(vals)]

                plt.figure(figsize=(19.2, 10.8))
                plt.plot(vals)
                plt.title(f'{key}')
                plt.ylabel(key)

                plt.savefig(dst / f'{key}', bbox_inches='tight', dpi=100)
                plt.close()

    def create_dst_paths(self):

        # plot_dir = self.dst_path / "plots"
        # if self.save_couples:
        #     os.makedirs(plot_dir, exist_ok=True)

        render_rgb_dir = self.dst_path / "color"
        os.makedirs(render_rgb_dir, exist_ok=True)

        render_depth_dir = self.dst_path / "depth"
        # os.makedirs(render_depth_dir, exist_ok=True)

        render_norms_dir = self.dst_path / "norms"
        # if self.save_normals:
        #     os.makedirs(render_norms_dir, exist_ok=True)

        return None, render_rgb_dir, render_depth_dir, render_norms_dir



