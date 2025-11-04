import torch
from pathlib import Path
from PIL import Image
import numpy as np
import json
import os

from LumenGSLAM.utils.general import my_logger

from ..data_process.loading import reshape_normalize_channel_first, preprocess_depth, load_poses
from ..data_process.pose_handling import align
from ..data_process.preprocessing import scale_intrinsics

def load_config_json(data_path):
    data_path = Path(data_path)
    assert os.path.isfile(data_path), f'{data_path} does not exist'

    with open(data_path, "r") as f:
        config = json.load(f)

    my_logger.info(f"loaded train config from: {data_path}")

    return config


def load_data_config(data_path, do_not_resize=False):
    data_path = Path(data_path)
    assert os.path.isfile(data_path), f'{data_path} does not exist'

    with open(data_path, "r") as f:
        config = json.load(f)

    my_logger.info(f"loaded dataset config from: {data_path}")

    curr_h, curr_w, target_h, target_w = config['image_height'], config['image_width'], config['desired_height'], config['desired_width']
    ratio_w = target_w / curr_w
    ratio_h = target_h / curr_h

    K = torch.eye(3)
    K[0, 0] = config['fx']
    K[1, 1] = config['fy']
    K[0, 2] = config['cx']
    K[1, 2] = config['cy']

    if not do_not_resize:
        K = scale_intrinsics(K, ratio_h, ratio_w)
        return K, (target_h, target_w), config['depth_scale'], Path(config['data_path'])
    else:
        return K, (curr_h, curr_w), config['depth_scale'], Path(config['data_path'])

class BaseDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, data_shape: tuple, depth_scale: float, first_c2w=None, pose_file_name='pose.txt',
                 split=None):
        super().__init__()

        self.split = split

        self.data_path = Path(data_path) # C3VD structure (colore, depth, pose.txt)
        self.data_shape = data_shape
        self.depth_scale = depth_scale
        self.pose_file_name = pose_file_name

        self.first_c2w = first_c2w

        self.data = self.load_all()

    def __getitem__(self, index):
        return self.data[index] # image, depth, pose

    def __len__(self):
        return len(self.data)

    def load_all(self):
        images = self.load_images()
        depths = self.load_depths()
        poses = self.load_poses(len(images))

        if self.split == 'train':
            # keep out 1 every 8
            data = [(i,d,p) for j, (i, d, p) in enumerate(zip(images, depths, poses)) if j not in range(7, len(os.listdir(self.data_path / 'color')), 8)]
        elif self.split == 'val':
            # only 1 every 8
            data = [(i, d, p) for j, (i, d, p) in enumerate(zip(images, depths, poses)) if j in range(7, len(os.listdir(self.data_path / 'color')), 8)]
        else:
            # whole seq
            data = [(i, d, p) for i, d, p in zip(images, depths, poses)]
        # data = data + data[700:]

        my_logger.info(f'loaded {len(data)} image/depth/poses triplets')

        return data

    def load_images(self):
        imgs_path = self.data_path / 'color'

        my_logger.info(f'loading images from {imgs_path}')

        # images = [torch.tensor(reshape_normalize_channel_first(np.asarray(imageio.imread(imgs_path / x), dtype=float), self.data_shape,
        #                            normalize=True, channel_first=True), dtype=torch.float32) for x in sorted(os.listdir(imgs_path))]


        images = [torch.tensor(reshape_normalize_channel_first(np.array(Image.open(imgs_path / x).convert('RGB')), self.data_shape,
                               normalize=True, channel_first=True), dtype=torch.float32) for x in sorted(os.listdir(imgs_path))]
        return images

    def load_depths(self):
        depths_path = self.data_path / 'depth'

        my_logger.info(f'loading depths from {depths_path}')

        # if 'SCARED' in str(depths_path):
        #     depths = [torch.tensor(preprocess_depth(cv2.imread(str(depths_path / x), -1).astype(np.float32), self.data_shape,
        #                                       depth_scale=self.depth_scale, channel_first=True), dtype=torch.float32) for x in sorted(os.listdir(depths_path))]
        #
        # else:
        depths = [torch.tensor(preprocess_depth(np.array(Image.open(depths_path / x), dtype=np.float64), self.data_shape,
                               depth_scale=self.depth_scale, channel_first=True), dtype=torch.float32) for x in sorted(os.listdir(depths_path))]

        return depths

    def load_poses(self, N):
        poses_path = self.data_path / self.pose_file_name

        if os.path.isfile(poses_path):
            my_logger.info(f'loading poses from {poses_path}')
            # poses are treated to be
            poses = load_poses(poses_path, world_frame=self.first_c2w, return_torch=False)

            if "Chiara_data/" in str(poses_path):
                my_logger.info('Mirroring X axis!')
                F = torch.tensor(np.diag([-1,1,1,1])).float()

                # F = torch.tensor(np.array([[-1,  0,  0, 0],  # X: left
                #                            [ 0,  0,  1, 0],  # Y: up
                #                            [ 0, -1,  0, 0],  # Z: into screen
                #                            [ 0,  0,  0, 1]])).float()

                poses_corr = [F @ x @ F for x in poses]
                poses = poses_corr
        else:
            my_logger.info(f'pose file not found, assuming no gt pose')
            poses = [None for _ in range(N)]

        return poses


class EvalLoader(BaseDataLoader):
    def __init__(self, data_path, data_shape: tuple, depth_scale: float, first_c2w=None, tracked_trj=None, pose_file_name='pose.txt',
                 split=None, use_every=1):
        super().__init__(data_path, data_shape, depth_scale, first_c2w, pose_file_name, split=split if tracked_trj is None else None)
        self.use_every = use_every

        data = [x for i, x in enumerate(self.data) if ((i == 0) or ((i + 1) % self.use_every) == 0)]
        self.data = data
        if tracked_trj and split is None:
            data = [(i,d,p) for (i,d,_), p in zip(data, tracked_trj)]
            self.data = data
        elif tracked_trj and split == 'val':
            # estimating alignment (Horn method)

            gt_poses_train = [p for i, (_,_,p) in enumerate(self.data) if i not in range(7, len(self.data), 8)]
            rot, trans, _ = align(torch.stack(gt_poses_train)[:,:3,3].cpu().numpy().T,
                                  torch.stack(tracked_trj)[:,:3,3].cpu().numpy().T)
            data = []
            for i, (im, d, p) in enumerate(self.data):
                # val data only
                if i in range(7, len(self.data), 8):
                    # aligning
                    horn_t = rot @ p[:3,3].numpy() + trans.squeeze()
                    p[:3,3] = torch.tensor(horn_t)
                    # tracked_trj.insert(i, horn_p)
                    data.append((im, d, p))
            # data = [(i, d, p) for (i, d, _), p in zip(self.data, tracked_trj)]
            self.data = data
            my_logger.info(f'retained only {len(data)} image/depth/poses triplets aligning with tracked pose')
        elif tracked_trj and split == 'train':
            train_data = [(im, d, p) for i, (im, d, p) in enumerate(self.data) if i not in range(7, len(self.data), 8)]
            data = [(i, d, p) for (i, d, _), p in zip(train_data, tracked_trj)]
            self.data = data
            my_logger.info(f'retained only {len(data)} image/depth/poses triplets using tracked pose')

