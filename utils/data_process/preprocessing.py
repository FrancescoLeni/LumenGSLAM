import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
import warnings
from kornia.geometry.linalg import compose_transformations, inverse_transformation

from .outlier_handling import energy_mask


def normalize_image(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Normalizes RGB image values from :math:`[0, 255]` range to :math:`[0, 1]` range.

    Args:
        rgb (torch.Tensor or numpy.ndarray): RGB image in range :math:`[0, 255]`

    Returns:
        torch.Tensor or numpy.ndarray: Normalized RGB image in range :math:`[0, 1]`

    Shape:
        - rgb: :math:`(*)` (any shape)
        - Output: Same shape as input :math:`(*)`
    """
    if torch.is_tensor(rgb):
        return rgb.float() / 255
    elif isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255
    else:
        raise TypeError("Unsupported input rgb type: %r" % type(rgb))


def channels_first(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Converts from channels last representation :math:`(*, H, W, C)` to channels first representation
    :math:`(*, C, H, W)`

    Args:
        rgb (torch.Tensor or numpy.ndarray): :math:`(*, H, W, C)` ordering `(*, height, width, channels)`

    Returns:
        torch.Tensor or numpy.ndarray: :math:`(*, C, H, W)` ordering

    Shape:
        - rgb: :math:`(*, H, W, C)`
        - Output: :math:`(*, C, H, W)`
    """
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):
        raise TypeError("Unsupported input rgb type {}".format(type(rgb)))

    if rgb.ndim < 3:
        raise ValueError(
            "Input rgb must contain atleast 3 dims, but had {} dims.".format(rgb.ndim)
        )
    if rgb.shape[-3] < rgb.shape[-1]:
        msg = "Are you sure that the input is correct? Number of channels exceeds height of image: %r > %r"
        warnings.warn(msg % (rgb.shape[-1], rgb.shape[-3]))
    ordering = list(range(rgb.ndim))
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]

    if isinstance(rgb, np.ndarray):
        return np.ascontiguousarray(rgb.transpose(*ordering))
    elif torch.is_tensor(rgb):
        return rgb.permute(*ordering).contiguous()

def reshape_normalize_channel_first(img, desired_shape: tuple[int, int], normalize=True, channel_first=False):
    """
    takes an image nd reshapes it to desired size, normalizes it and set channels

    """

    original_h, original_w = img.shape[:2]
    desired_h, desired_w = desired_shape

    assert original_h / desired_h == original_w / desired_w, "don't distort!"

    if img.shape[0] != desired_h and img.shape[1] != desired_w:
        img = cv2.resize(img, (desired_w, desired_h), interpolation=cv2.INTER_AREA)
        img = img[..., np.newaxis] if img.ndim < 3 else img
    if normalize:
        img = normalize_image(img)
    if channel_first:
        img = channels_first(img)
    return img


def preprocess_depth(depth: np.ndarray, desired_shape: tuple[int, int], channel_first=False, depth_scale=None):
    """
    Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters.

    Args:
        -- depth: Raw depth image (just loaded)
        -- desired_shape: Desired shape for the depth image (so it optionally resize)
        -- channel_first: Whether to converto to channel_first
        -- depth_scale: Scaling factor from raw depth to meters

    Returns:
        np.ndarray: Preprocessed depth
    """

    depth = np.expand_dims(depth, -1)
    depth = reshape_normalize_channel_first(depth, desired_shape, False, channel_first)
    return depth / depth_scale
    # return depth * (200 - 0.1) + 0.1

def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

    return depth_silhouette

def preprocess_poses(poses: torch.Tensor, world_frame: torch.Tensor = None):
    r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
    homogenous transformation for all other poses.

    Args:
        poses (torch.Tensor): Pose matrices to be preprocessed (c2w)
        world_frame (torch.Tensor): if not None, it uses this as reference frame of the sequence, and all the poses are
                                    computed respect of it

    Returns:
        Output (torch.Tensor): Preprocessed poses (c2w)

    Shape:
        - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
    """

    if isinstance(world_frame, torch.Tensor):
        frame0 = world_frame.unsqueeze(0).repeat(poses.shape[0], 1, 1).to(poses.device)
    else:
        frame0 = poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1)

    return relative_transformation(
        frame0,
        poses,
        orthogonal_rotations=False,
    )

def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    r"""Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    .. note:: Works with imperfect (non-orthogonal) rotation matrices as well.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        orthogonal_rotations (bool): If True, will invert `trans_01` assuming `trans_01[:, :3, :3]` are
            orthogonal rotation matrices (more efficient). Default: False

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = gradslam.geometry.geometryutils.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_01.shape)
        )
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )
    trans_10: torch.Tensor = (
        inverse_transformation(trans_01)
        if orthogonal_rotations
        else torch.inverse(trans_01)
    )
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12


def get_pointcloud(color, depth, intrinsics, w2c=None, mask: torch.Tensor = None):
    """
    extracts a point cloud from a depth maps given the camera intrinsics

    args:
        -- color: the image frame (CxHxW)
        -- depth: the depth map (CxHxW)
        -- intrinsics: the camera intrinsics
        -- w2c: the world to camera transformation matrix, if not None it uses it to transform the point from camera coord into world frame
        -- mask: mask tensor for valid pixels

    returns:
        -- point_cld: a torch.tensor shape (N_points, 9) where first 3 features are x,y,z coords, while the last 3 are rgb color, last 3 normals
    """

    width, height = color.shape[2], color.shape[1]
    CX, CY, FX, FY  = intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).float(),
                                    torch.arange(height).float(),
                                    indexing='xy')
    x_grid, y_grid = x_grid.to(color.device), y_grid.to(color.device)

    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)  # camera coord

    N = normals_from_depth(depth)
    N = N.view(3, -1).T  # (C, H, W) -> (C, H*W) -> (H * W, C)
    N = torch.nn.functional.normalize(N, dim=1)

    if isinstance(w2c, torch.Tensor):

        pix_ones = torch.ones(height * width, 1).float().to(color.device)
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.linalg.inv(w2c)#.double()
        pts = (c2w @ pts4.T).T[:, :3]  # world coord

        # N4 = torch.cat((N, pix_ones), dim=1)

        # vectors DO NOT TRANSLATE!!!!!!!
        N = (c2w[:3,:3] @ N.T).T  # world coord
    else:
        pts = pts_cam

    # debug
    # torch.save({'pts_cam': pts_cam, 'pts_w': pts, 'x_grid': x_grid, 'y_grid': y_grid, 'w2c': w2c, 'c2w': c2w}, 'point_cloud.pt')
    # raise AssertionError()

    # Colorize point cloud
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1)

    point_cld = torch.cat((point_cld, N), -1)

    # debug line
    # Image.fromarray(np.uint8((torch.permute(color, (1, 2, 0)) * mask.reshape(320, 320, 1)).detach().cpu().numpy()*255), 'RGB').save('gaussian.png')

    # Select points based on mask
    if isinstance(mask, torch.Tensor):
        point_cld = point_cld[mask]

    return point_cld


def scale_intrinsics(
    intrinsics: Union[np.ndarray, torch.Tensor],
    h_ratio: Union[float, int],
    w_ratio: Union[float, int],
):
    r"""Scales the intrinsics appropriately for resized frames where
    :math:`h_\text{ratio} = h_\text{new} / h_\text{old}` and :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Args:
        intrinsics (numpy.ndarray or torch.Tensor): Intrinsics matrix of original frame
        h_ratio (float or int): Ratio of new frame's height to old frame's height
            :math:`h_\text{ratio} = h_\text{new} / h_\text{old}`
        w_ratio (float or int): Ratio of new frame's width to old frame's width
            :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

    Returns:
        numpy.ndarray or torch.Tensor: Intrinsics matrix scaled approprately for new frame size

    Shape:
        - intrinsics: :math:`(*, 3, 3)` or :math:`(*, 4, 4)`
        - Output: Matches `intrinsics` shape, :math:`(*, 3, 3)` or :math:`(*, 4, 4)`

    """
    if isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = intrinsics.astype(np.float32).copy()
    elif torch.is_tensor(intrinsics):
        scaled_intrinsics = intrinsics.to(torch.float).clone()
    else:
        raise TypeError("Unsupported input intrinsics type {}".format(type(intrinsics)))
    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
        raise ValueError(
            "intrinsics must have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(
                intrinsics.shape
            )
        )
    if (intrinsics[..., -1, -1] != 1).any() or (intrinsics[..., 2, 2] != 1).any():
        warnings.warn(
            "Incorrect intrinsics: intrinsics[..., -1, -1] and intrinsics[..., 2, 2] should be 1."
        )

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
    return scaled_intrinsics


def normals_from_depth(depth, mode='central'):
    # ensures everything works
    if len(depth.shape) == 3:
        depth = depth.unsqueeze(0)


    _, _, H, W = depth.shape

    if mode == 'forward':
        # Compute gradients along x and y axes using finite differences
        diff_x = depth[..., 1:, 1:] - depth[..., 1:, :-1]
        diff_y = depth[..., 1:, 1:] - depth[..., :-1, 1:]
    elif mode == 'central':

        # Padding to allow central differences at borders
        depth_padded = F.pad(depth, (1, 1, 1, 1), mode='replicate')

        # Central differences
        diff_x = (depth_padded[:, :, 1:-1, 2:] - depth_padded[:, :, 1:-1, :-2]) / 2.0
        diff_y = (depth_padded[:, :, 2:, 1:-1] - depth_padded[:, :, :-2, 1:-1]) / 2.0
    else:
        raise NotImplementedError()

    # Compute the z component of the normal based on the gradients
    z = torch.sqrt(1 + diff_x ** 2 + diff_y ** 2)

    # Stack the x, y, and z components to form the normal vector
    normal = torch.cat([-diff_x, -diff_y, z], dim=1)

    # Normalize the normal vector
    normal = F.normalize(normal, dim=1)

    return F.interpolate(normal, (H,W)).squeeze()  # as we are not batching -> 3,H,W


def compute_image_gradients(img: torch.Tensor):

    if img.shape == 3:
        img = img.unsqueeze(0)

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)

    grads_x = F.conv2d(img, sobel_x, padding=1, groups=1)
    grads_y = F.conv2d(img, sobel_y, padding=1, groups=1)

    return grads_x, grads_y


def depth_filter(depth, intrinsics, c2w, z_threshold):
    """
    extracts a point cloud from a depth maps given the camera intrinsics

    args:
        -- color: the image frame (CxHxW)
        -- depth: the depth map (CxHxW)
        -- intrinsics: the camera intrinsics
        -- c2w: the camera to world transformation matrix, if not None it uses it to transform the point from camera coord into world frame
        -- z_threshold: the depth threshold for culling

    returns:
        -- point_cld: a torch.tensor shape (N_points, 9) where first 3 features are x,y,z coords, while the last 3 are rgb color, last 3 normals
    """
    c2w = c2w.to(depth.device)
    width, height = depth.shape[2], depth.shape[1]
    CX, CY, FX, FY  = intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1]

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).float(),
                                    torch.arange(height).float(),
                                    indexing='xy')
    x_grid, y_grid = x_grid.to(depth.device), y_grid.to(depth.device)

    xx = (x_grid - CX) / FX
    yy = (y_grid - CY) / FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1).to(depth.device)
    # print('cam')
    # print(pts_cam[:, 2].min())
    pix_ones = torch.ones(height * width, 1).float().to(depth.device)
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    pts = (c2w @ pts4.T).T[:, :3]  # world coord

    mask_flat = (pts[:, 2] < z_threshold)  # True if point is too close
    # print('world')
    # print(pts[:, 2].min())
    return mask_flat.reshape(height, width)
