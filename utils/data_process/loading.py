from pathlib import Path
import os
from PIL import Image
import numpy as np
import torch

from .preprocessing import reshape_normalize_channel_first, preprocess_depth, preprocess_poses
from ..general import my_logger


def load_C3VD_poses(file_path):

    poses = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        pose = list(map(float, line.split(sep=',')))

        pose = torch.Tensor(pose).reshape(4, 4).float().transpose(0, 1)

        poses.append(pose)

    return poses


def load_poses(file_path, world_frame=None, return_torch=True, inv=True, relative=True):
    """
    loads poses in C3VD format and returns a stack of torch.Tensor w2c poses (this assumes c2w poses are stored in the .txt)

    args:
        -- file_path: path to C3VD poses
        -- world_frame: world frame to set the relative poses, if None, the first pose in the sequence is used
        -- return_torch: whether to return torch tensors or a list of torch tensors
        -- inv: whether to invert poses (to pass from c2w to w2c) (True == w2c)
    """


    poses = load_C3VD_poses(file_path) # c2w

    # this set all poses relative to the first (the first becomes identity and the other follows)

    if relative:
        poses = preprocess_poses(torch.stack(poses), world_frame=world_frame)

    if inv:
        poses = [torch.linalg.inv(x) for x in poses] # w2c
    else:
        poses = [x for x in poses]

    return poses if not return_torch else torch.stack(poses, dim=0)


def load_imgs_depths_poses(src_path, desired_shape: tuple[int, int], depth_scale=2.55, channel_first=False, world_frame=None):

    """
    !!!!!!!!!!!!!!!!!!   I'M ASSUMING C3VD DATA    !!!!!!!!!!!!!!!!!!!!
                     (both for pose and depth_scale)

    args:
        -- src_path: path to the source images
        -- desired_shape: desired image shape for optional reshaping
        -- depth_scale: depth scale for conversion to meters (default is C3VD)
        -- channel_first: whether to return channel first tensor
        -- world_frame: first pose to set world frame, if None uses the first of the sequence
                       (it has to be set if the model was trained with a specific sequence and then it)
                        is tested on a different set of poses
    """

    src_path = Path(src_path)

    imgs_path = src_path / 'color'
    depths_path = src_path / 'depth'
    poses_path = src_path / 'pose.txt'

    # debug lines
    # i = reshape_normalize_channel_first(np.array(Image.open(imgs_path / sorted(os.listdir(imgs_path))[0]).convert('RGB')), desired_shape,  normalize=True, channel_first=False)
    # d = preprocess_depth(np.array(Image.open(depths_path / sorted(os.listdir(depths_path))[0]), dtype=np.float64), desired_shape, depth_scale=depth_scale)
    # from matplotlib import pyplot as plt
    # f, a = plt.subplots(1,2)
    # a[0].imshow(i)
    # a[1].imshow(d)
    # plt.savefig('debug_fig')

    print(f'loading images from {imgs_path}')
    images = [torch.tensor(reshape_normalize_channel_first(np.array(Image.open(imgs_path  / x).convert('RGB')), desired_shape,  normalize=True, channel_first=channel_first)) for x in sorted(os.listdir(imgs_path))]
    print(f'loading depths from {depths_path}')
    depths = [torch.tensor(preprocess_depth(np.array(Image.open(depths_path / x), dtype=np.float64), desired_shape, depth_scale=depth_scale, channel_first=channel_first), dtype=torch.float64) for x in sorted(os.listdir(depths_path))]
    print(f'loading poses from {poses_path}')
    poses = load_poses(poses_path, world_frame)

    return images, depths, poses


def load_params(params_path, device='cuda:0'):

    my_logger.info(f'loading params from: {params_path}')

    if Path(params_path).suffix == '.npz':
        params = np.load(params_path)
        params = {k: torch.tensor(params[k]).to(torch.device(device)) for k in params.keys()}
        return params

    elif Path(params_path).suffix == '.pt':
        params = torch.load(params_path, map_location=torch.device(device))
        gaussians = params['gaussians']
        first_c2w = params['first_c2w']
        tracked_trj = params['tracked_trj']
        return gaussians, first_c2w, tracked_trj

    else:
        raise NotImplementedError


def load_first_pose(src):

    with open(src, "r") as f:
        line = f.readline()

    pose = list(map(float, line.split(sep=',')))
    pose = torch.Tensor(pose).reshape(4, 4).float().transpose(0, 1)

    return pose # c2w (c3vd)


