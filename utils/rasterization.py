import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from .data_process.preprocessing import get_depth_and_silhouette


def setup_camera(w, h, k, w2c, near=0.01, far=100, bg=[0,0,0], use_simplification=True):
    """
    Sets up a camera with intrinsic and extrinsic parameters for rendering or projection.

    args:
        -- w : int; The width of the image (in pixels).
        -- h : int; The height of the image (in pixels).
        -- k : np.array; The 3x3 camera intrinsic matrix, where:
              - k[0][0] = fx (focal length in x)
              - k[1][1] = fy (focal length in y)
              - k[0][2] = cx (principal point x-coordinate)
              - k[1][2] = cy (principal point y-coordinate)
        -- w2c : np.array; A 4x4 world-to-camera transformation matrix (extrinsics),
        -- near : float; The near clipping plane distance. Objects closer than this will be clipped.
        -- far : float; The far clipping plane distance. Objects farther than this will be clipped.
        -- bg : list[float]; Background color as an RGB list.
        -- use_simplification : bool; If True, uses sh_degree=0 (color). If False, uses sh_degree=3.

    Returns:
        -- cam : A Camera object initialized with the given parameters (GS rasterizer settings)
    """
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0 if use_simplification else 3,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def params2rendervar(params, pts):
    """
    transforms params in EndoGSLAM format to dict suitable for GS rasterization of image

    args:
        -- params: EndoGSLAM format prams
        -- pts: point cloud in world coordinates

    """

    rendervar = {
        'means3D': pts,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=False, device="cuda") + 0
    }
    if params['log_scales'].shape[1] == 1:
        rendervar['colors_precomp'] = params['rgb_colors']
        rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
    else:
        rendervar['shs'] = torch.cat((params['rgb_colors'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2), params['feature_rest'].reshape(params['rgb_colors'].shape[0], 3, -1).transpose(1, 2)), dim=1)
        rendervar['scales'] = torch.exp(params['log_scales'])
    return rendervar


def params2depth_silhouette(params, w2c, pts):
    """
    transforms params in EndoGSLAM format to dict suitable for GS rasterization of DEPTH and Silhouette ('objectness')

    args:
        -- params: EndoGSLAM format prams
        -- w2c: current frame w2c matrix (used to initialize depth for rendering)
        -- pts: point cloud in world coordinates

    """
    rendervar = {
        'means3D': pts,
        'colors_precomp': get_depth_and_silhouette(pts, w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        # 'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=False, device="cuda") + 0
    }
    if params['log_scales'].shape[1] == 1:
        rendervar['scales'] = torch.exp(torch.tile(params['log_scales'], (1, 3)))
    else:
        rendervar['scales'] = torch.exp(params['log_scales'])
    return rendervar