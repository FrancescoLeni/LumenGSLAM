import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import math
import pandas as pd
from pathlib import Path
import copy
from collections import defaultdict

from utils.general import my_logger
from utils.metrics import get_psnr_ssim_lpips


# now only mapping logging
class TrainLogger:
    def __init__(self, dst_path, tracking):
        self.dst_path = dst_path
        self.debug_dict = defaultdict(list)
        self.dens_dict = defaultdict(list)
        self.mask_dict = defaultdict(list)
        self.logs = []
        self.tracking = tracking

        self.keyframes_ids = []

        self.iter_metrics_dict = dict(frame_id=[], psnr=[], depth_rmse=[], depth_l1=[], lpips=[], ssim=[], mssim=[])
        self.iter_track_metrics = dict(frame_id=[], ate=[])
        self.iter_loss_dict = {'frame_id': []}
        self.track_iter_loss_dict = {'frame_id': []}

    def get_masks_growth(self, masks, i):
        non_presence_sil_mask, non_presence_depth_mask, valid_depth_mask, valid_color_mask = masks

        self.mask_dict['frame_id'].append(i)
        self.mask_dict['non_presence_sil_mask'].append(non_presence_sil_mask.sum().item())
        self.mask_dict['non_presence_depth_mask'].append(non_presence_depth_mask.sum().item())
        self.mask_dict['valid_depth_mask'].append(valid_depth_mask.sum().item())
        self.mask_dict['valid_color_mask'].append(valid_color_mask.sum().item())


    def after_growth(self, i, added):
        self.debug_dict['frame_id'].append(i)
        self.debug_dict['added_pts'].append(added)

    def after_dens(self, i, pts, scene_radius):
        pruned, split, cloned = pts

        self.dens_dict['frame_id'].append(i)
        self.dens_dict['pruned'].append(pruned)
        self.dens_dict['split'].append(split)
        self.dens_dict['cloned'].append(cloned)
        self.dens_dict['scene radius'].append(scene_radius.cpu().item())

    def mapping_start(self, sampled_keyframes):
        self.keyframes_ids += [x for x in np.unique(sampled_keyframes) if x not in self.keyframes_ids]

    def iter_end(self, frame_id, r_im, gt_im, rastered_depth, depth, current_loss, do_metrics):


        self.iter_loss_dict['frame_id'].append(frame_id)

        if len(self.iter_loss_dict) == 1:
            for k in current_loss.keys():
                self.iter_loss_dict[k] = []

        if do_metrics:
            self.iter_metrics_dict['frame_id'].append(frame_id)
            self.compute_metrics(r_im, gt_im, rastered_depth, depth)

        for k in current_loss.keys():
            self.iter_loss_dict[k].append(current_loss[k])

    def tracking_iter_end(self, frame_id, current_loss, ate):

        self.track_iter_loss_dict['frame_id'].append(frame_id)
        self.iter_track_metrics['ate'].append(ate)
        self.iter_track_metrics['frame_id'].append(frame_id)

        if len(self.track_iter_loss_dict) == 1:
            for k in current_loss.keys():
                self.track_iter_loss_dict[k] = []

        for k in current_loss.keys():
            self.track_iter_loss_dict[k].append(current_loss[k])

    def mapping_end(self, pts):
        self.logs.append((copy.deepcopy(self.iter_metrics_dict), copy.deepcopy(self.iter_loss_dict),
                          copy.deepcopy(self.iter_track_metrics), copy.deepcopy(self.track_iter_loss_dict)))
        self.debug_dict['tot pts'].append(pts)

        self.reset_iter_dicts()

    def get_current_metrics_map(self):
        return {k: np.mean(self.iter_metrics_dict[k]) for k in self.iter_metrics_dict.keys()}


    def compute_metrics(self, r_im, gt_im, rastered_depth, depth):
        # Mask invalid depth in GT
        valid_depth_mask = (depth > 0) & (depth < 1e10)
        rastered_depth = rastered_depth * valid_depth_mask

        weighted_im = r_im * valid_depth_mask
        weighted_gt_im = gt_im * valid_depth_mask

        psnr, ssim, lpips, mssim = get_psnr_ssim_lpips(weighted_im, weighted_gt_im)

        self.iter_metrics_dict['psnr'].append(psnr.cpu().item())
        self.iter_metrics_dict['ssim'].append(ssim.cpu().item())
        self.iter_metrics_dict['mssim'].append(mssim.cpu().item())
        self.iter_metrics_dict['lpips'].append(lpips)

        # depth metrics
        diff_depth_rmse = torch.sqrt((((rastered_depth - depth)) ** 2))
        diff_depth_rmse = diff_depth_rmse * valid_depth_mask

        rmse = diff_depth_rmse.sum() / valid_depth_mask.sum()

        diff_depth_l1 = torch.abs((rastered_depth - depth))
        diff_depth_l1 = diff_depth_l1 * valid_depth_mask
        depth_l1 = diff_depth_l1.sum() / valid_depth_mask.sum()

        self.iter_metrics_dict['depth_rmse'].append(rmse.cpu().numpy())
        self.iter_metrics_dict['depth_l1'].append(depth_l1.cpu().numpy())

    def reset_iter_dicts(self):
        self.iter_metrics_dict = dict(frame_id=[], psnr=[], depth_rmse=[], depth_l1=[], lpips=[], ssim=[], mssim=[])
        iter_dict_reset = {k: [] for k in self.iter_loss_dict.keys()}
        self.iter_loss_dict = iter_dict_reset
        track_iter_dict_reset = {k: [] for k in self.track_iter_loss_dict.keys()}
        self.track_iter_loss_dict = track_iter_dict_reset
        self.iter_track_metrics = dict(frame_id=[], ate=[])

    def save_metrics_csv(self, plot_metrics=False, name='metrics.csv'):

        # single csv with metrics and losses per timestep
        csv_dict = {k: [] for k in np.unique(['map_'+x for x in list(self.iter_metrics_dict.keys()) if x != 'frame_id'] + ['map_'+x for x in list(self.iter_loss_dict.keys()) if x != 'frame_id'] +
                                             ['track_'+x for x in list(self.iter_track_metrics.keys()) if x != 'frame_id'] + ['track_'+x for x in list(self.track_iter_loss_dict.keys()) if x != 'frame_id'])}
        csv_dict['frame_id'] = list(range(len(self.logs)))

        for j, (m_m, l_m, m_t, l_t) in enumerate(self.logs):
            if plot_metrics:
                for k in m_m.keys():
                    if k != 'frame_id':
                        csv_dict['map_'+k].append(np.mean(m_m[k]))
            else:
                csv_dict_filt = {k: csv_dict[k] for k in csv_dict if not ('map_' in k and 'loss' not in k)}
                csv_dict = csv_dict_filt
            for k in l_m.keys():
                if k != 'frame_id':
                    csv_dict['map_'+k].append(np.mean(l_m[k]))
            if self.tracking:
                if j>0 and len(m_t['ate']) > 0:
                    for k in m_t.keys():
                        if k != 'frame_id':
                            csv_dict['track_'+k].append(np.min(m_t[k])) # this is definitely not aligned with the loss but hell yeah!
                    for k in l_t.keys():
                        if k != 'frame_id':
                            csv_dict['track_'+k].append(np.min(l_t[k])) # I always retrieve the min loss one in the pipe
                else:
                    for k in m_t.keys():
                        if k != 'frame_id':
                            csv_dict['track_'+k].append(0)
            else:
                csv_dict_filt = {k: csv_dict[k] for k in csv_dict if 'track_' not in k}
                csv_dict = csv_dict_filt

        if self.tracking:
            track_loss_keys = [k for k in csv_dict.keys() if ('track_' in k and 'loss' in k)]
            for k in track_loss_keys:
                csv_dict[k] = [0] + csv_dict[k]


        # saving metrics to csv
        df = pd.DataFrame(csv_dict)
        # setting frame id as first col
        df = df [['frame_id'] + [col for col in df.columns if col != 'frame_id']]
        df.to_csv(self.dst_path / name, index=False)

        return csv_dict

    def get_per_frame_dicts(self, plot_metrics=False):

        out = defaultdict(lambda: defaultdict(list))

        for j, (m_m, l_m, m_t, l_t) in enumerate(self.logs):
            for k in np.unique(l_m['frame_id']):
                if k in self.keyframes_ids:
                    idxs_m = np.where(np.array(l_m['frame_id']) == k)[0]
                    idxs_t = np.where(np.array(l_t['frame_id']) == k)[0]
                    # print(idxs_m, idxs_t)
                    if plot_metrics:
                        for key in m_m.keys():
                            if key != 'frame_id':
                                out[k]['map_'+key] += list(np.array(m_m[key])[idxs_m])
                    for key in l_m.keys():
                        if key != 'frame_id':
                            out[k]['map_'+key] += list(np.array(l_m[key])[idxs_m])
                    if self.tracking:
                        if j>0 and len(m_t['ate']) > 0:
                            for key in m_t.keys():
                                if key != 'frame_id':
                                    out[k]['track_'+key] += list(np.array(m_t[key])[idxs_t])
                            for key in l_t.keys():
                                if key != 'frame_id':
                                    out[k]['track_'+key] += list(np.array(l_t[key])[idxs_t])
                        else:
                            for key in m_t.keys():
                                if key != 'frame_id':
                                    out[k]['track_' + key] += list(np.zeros(15))
                            for key in l_t.keys():
                                if key != 'frame_id':
                                    out[k]['track_' + key] += list(np.zeros(15))
        if self.tracking:
            # tracking not done on frame 0
            track_loss_keys = [k for k in out[list(out.keys())[1]].keys() if ('track_' in k and 'loss' in k)]
            for k in track_loss_keys:
                out[0][k] += list(np.zeros(15))

        return dict(out)

    def save_info_csv(self, name='point_added.csv'):
        df = pd.DataFrame(self.debug_dict)
        df.to_csv(self.dst_path / name, index=False)

        df = pd.DataFrame(self.dens_dict)
        df.to_csv(self.dst_path / 'dens_stats.csv', index=False)

        # df = pd.DataFrame(self.mask_dict)
        # df.to_csv(self.dst_path / 'mask_growth.csv', index=False)

    def plot_save(self, plot_metrics=False):
        csv_dict = self.save_metrics_csv(plot_metrics)

        map_loss_dict = {k: csv_dict[k] for k in csv_dict.keys() if 'loss' in k and 'map_' in k and 'frame' not in k}
        track_loss_dict = {k: csv_dict[k] for k in csv_dict.keys() if 'loss' in k and 'track_' in k and 'frame' not in k}
        map_metrics_dict = {k: csv_dict[k] for k in csv_dict.keys() if 'loss' not in k and 'map_' in k and 'l1' not in k and 'frame' not in k}
        track_metrics_dict = {k: csv_dict[k] for k in csv_dict.keys() if 'loss' not in k and 'track_' in k and 'l1' not in k and 'frame' not in k}

        plt_dst = self.dst_path / 'plots'
        os.makedirs(plt_dst, exist_ok=True)

        my_logger.info(f'saving global stats to: {plt_dst}...')

        map_loss_dst = plt_dst / 'map_loss.png'
        track_loss_dst = plt_dst / 'track_loss.png'
        map_metrics_dst = plt_dst / 'map_metrics.png'
        track_metrics_dst = plt_dst / 'track_metrics.png'

        self.plot_from_dict(map_loss_dict, map_loss_dst)
        if plot_metrics:
            self.plot_from_dict(map_metrics_dict, map_metrics_dst)
        if self.tracking and len(track_metrics_dict['track_ate']) > 0:
            self.plot_from_dict(track_loss_dict, track_loss_dst)
            self.plot_from_dict(track_metrics_dict, track_metrics_dst)

        # plotting for every frame instance
        per_frame_dict = self.get_per_frame_dicts(plot_metrics)
        plt_dst_frames = plt_dst / 'per_frame'
        os.makedirs(plt_dst_frames, exist_ok=True)


        my_logger.info(f'saving per-frame stats to: {plt_dst_frames}...')

        # saving some plots
        for i in per_frame_dict.keys():

            os.makedirs(plt_dst_frames / f'frame_{i}', exist_ok=True)

            map_loss_dict = {k: per_frame_dict[i][k] for k in per_frame_dict[i].keys() if 'loss' in k and 'map_' in k and 'frame' not in k}
            track_loss_dict = {k: per_frame_dict[i][k] for k in per_frame_dict[i].keys() if 'loss' in k and 'track_' in k and 'frame' not in k}
            map_metrics_dict = {k: per_frame_dict[i][k] for k in per_frame_dict[i].keys() if 'loss' not in k and 'map_' in k and 'l1' not in k and 'frame' not in k}
            track_metrics_dict = {k: per_frame_dict[i][k] for k in per_frame_dict[i].keys() if 'loss' not in k and 'track_' in k and 'l1' not in k and 'frame' not in k}


            map_loss_dst = plt_dst_frames / f'frame_{i}' / 'map_loss.png'
            track_loss_dst = plt_dst_frames / f'frame_{i}' / 'track_loss.png'
            map_metrics_dst = plt_dst_frames / f'frame_{i}' / 'map_metrics.png'
            track_metrics_dst = plt_dst_frames / f'frame_{i}' / 'track_metrics.png'

            self.plot_from_dict(map_loss_dict, map_loss_dst, xlabel='n')
            if plot_metrics:
                self.plot_from_dict(map_metrics_dict, map_metrics_dst, xlabel='n')

            if self.tracking and len(track_metrics_dict['track_ate']) > 0:
                self.plot_from_dict(track_metrics_dict, track_metrics_dst, xlabel='n')
                self.plot_from_dict(track_loss_dict, track_loss_dst, xlabel='n')
        # saving the dictionary
        # torch.save(per_frame_dict, self.dst_path / 'per_frame_stats.pt')

        # # mapping iterations
        #
        # plt_mapping_dst = plt_dst / 'mapping'
        # os.makedirs(plt_dst, exist_ok=True)
        #
        # my_logger.info(f'saving mapping stats to: {plt_mapping_dst}...')
        #
        # for i in range(0, len(self.logs), plot_every):
        #     os.makedirs(plt_mapping_dst / f'time_{i}', exist_ok=True)
        #
        #     loss_dict = {k: self.logs[i][1][k] for k in self.logs[i][1].keys() if 'loss' in k and 'frame' not in k}
        #     metrics_dict = {k: self.logs[i][0][k] for k in self.logs[i][0].keys() if 'loss' not in k and 'l1' not in k and 'frame' not in k}
        #
        #     loss_dst = plt_mapping_dst / f'time_{i}' / 'loss.png'
        #     metrics_dst = plt_mapping_dst / f'time_{i}' / 'metrics.png'
        #
        #     self.plot_from_dict(loss_dict, loss_dst, xlabel='mapping iters')
        #     self.plot_from_dict(metrics_dict, metrics_dst, xlabel='mapping iters')

        # if self.tracking:
        #     plt_tracking_dst = plt_dst / 'tracking'
        #     os.makedirs(plt_dst, exist_ok=True)
        #
        #     my_logger.info(f'saving time_step stats...')
        #
        #     for i in range(len(self.logs)):
        #         if i != 0: # tracking not done on frame 0
        #             i = int(i)
        #             os.makedirs(plt_tracking_dst / f'time_{i}', exist_ok=True)
        #
        #             # map_loss_dict = {k: self.logs[i][1][k] for k in self.logs[i][1].keys() if 'loss' in k and 'frame' not in k}
        #             # map_metrics_dict = {k: self.logs[i][0][k] for k in self.logs[i][0].keys() if 'loss' not in k and 'l1' not in k and 'frame' not in k}
        #             track_loss_dict = {k: self.logs[i][3][k] for k in self.logs[i][3].keys() if 'loss' in k and 'frame' not in k}
        #             track_metrics_dict = {k: self.logs[i][2][k] for k in self.logs[i][2].keys() if 'loss' not in k and 'l1' not in k and 'frame' not in k}
        #
        #             track_loss_dst = plt_tracking_dst / f'time_{i}' / 'track_loss.png'
        #             track_metrics_dst = plt_tracking_dst / f'time_{i}' / 'track_metrics.png'
        #
        #             self.plot_from_dict(track_loss_dict, track_loss_dst, xlabel='tracking iters')
        #             self.plot_from_dict(track_metrics_dict, track_metrics_dst, xlabel='tracking iters')

    def progress_plot(self, frame_id, iter_id, iter_data, r_im, r_depth, r_normals):
        if iter_id % 5 == 0:
            # print(iter_id)
            debug_dst = self.dst_path / 'progress_plots' / f'frame_{frame_id}'
            os.makedirs(debug_dst, exist_ok=True)
            f, a = plt.subplots(2, 2)
            a = a.flatten()
            f.suptitle(f"frame {iter_data['id']}")
            a[0].imshow(r_im.clone().clamp(0,1).permute(1,2,0).detach().cpu().numpy())
            a[0].set_title('rendered')
            a[0].axis('off')
            a[1].imshow(iter_data['im'].clone().clamp(0,1).permute(1,2,0).detach().cpu().numpy())
            a[1].set_title('gt')
            a[1].axis('off')
            a[2].imshow(r_depth.clone().permute(1,2,0).detach().cpu().numpy())
            a[2].set_title('rendered')
            a[2].axis('off')
            a[3].imshow(iter_data['depth'].clone().permute(1,2,0).detach().cpu().numpy())
            a[3].set_title('gt depth')
            a[3].axis('off')
            # if isinstance(r_normals, torch.Tensor):
            #     a[3].imshow(min_max(r_normals.clone().permute(1,2,0).detach().cpu().numpy()))
            #     a[3].set_title('rendered norms')
            #     a[3].axis('off')
            # plt.tight_layout()

            plt.savefig(debug_dst / f"iter_{iter_id}.png")
            plt.close()

    @staticmethod
    def plot_from_dict(data, dst, xlabel=None):
        # Calculate the number of columns and rows
        cols = math.floor(math.sqrt(len(data) * 1.5))  # Increase columns for rectangle
        rows = math.ceil(len(data) / cols)  # Rows as a function of columns

        f, axs = plt.subplots(rows, cols, figsize=(19.2, 10.8))

        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        else:
            axs = [axs]

        for i, m in enumerate(data.keys()):
            axs[i].plot(data[m])
            axs[i].set_title(m)
            if xlabel is not None:
                axs[i].set_xlabel(xlabel)

        plt.savefig(dst, dpi=100)
        plt.close()


    def save_logs_dict(self, name='logs.pt'):
        torch.save(self.logs_dict, self.dst_path / name)


class DebugLoggerPBR:
    def __init__(self, dst_path):
        self.dst_path = Path(dst_path)
        self.F_dict = defaultdict(list)
        self.G_dict = defaultdict(list)
        self.D_dict = defaultdict(list)
        self.pbr_dict = defaultdict(list)

    def forward(self, F, D, G, pbr, i):
        fresnel, H_V_sum = F
        d, N_dot_H = D
        g, N_dot_V_clamp, N_dot_L_clamp = G
        I_diff, I_spec = pbr

        self.F_dict['frame id'].append(i)
        self.F_dict['F'].append(fresnel.detach().cpu().numpy())
        self.F_dict['H_V_sum'].append(H_V_sum.detach().cpu().numpy())

        self.D_dict['frame id'].append(i)
        self.D_dict['D'].append(d.detach().cpu().numpy())
        self.D_dict['N_dot_H'].append(N_dot_H.detach().cpu().numpy())

        self.G_dict['frame id'].append(i)
        self.G_dict['G'].append(g.detach().cpu().numpy())
        self.G_dict['N_dot_V_clamp'].append(N_dot_V_clamp.detach().cpu().numpy())
        self.G_dict['N_dot_L_clamp'].append(N_dot_L_clamp.detach().cpu().numpy())

        self.pbr_dict['frame id'].append(i)
        self.pbr_dict['I diff'].append(I_diff.detach().cpu().numpy())
        self.pbr_dict['I spec'].append(I_spec.detach().cpu().numpy())

    def save(self):
        torch.save({'F': self.F_dict, 'G': self.G_dict, 'D': self.D_dict, 'pbr': self.pbr_dict}, self.dst_path / 'pbr_debug.pt')

    def __call__(self, **kwargs):
        self.forward(**kwargs)


