
import torch
import numpy as np

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def compute_projection_jacobians(pts, w2c, fx, fy):
    """
    Compute the Jacobian J for projecting 3D Gaussians into the image space.

    args:
        pts: (N, 3) tensor of Gaussian centers in world coordinates
        w2c: (4, 4) tensor, world-to-camera transformation matrix
        fx, fy: focal lengths (scalars)

    returns:
        Js: (N, 2, 3) tensor of Jacobian matrices
    """
    N = pts.shape[0]

    # Convert to homogeneous coordinates
    means_h = torch.cat([pts, torch.ones((N, 1), device=pts.device)], dim=-1)  # (N, 4)

    # Transform to camera space
    means_cam = (w2c @ means_h.T).T[:, :3]  # (N, 3)
    X_c, Y_c, Z_c = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]

    # Avoid division by zero
    eps = 1e-8
    Z_c_safe = Z_c + eps

    # Compute entries of J
    J = torch.zeros((N, 2, 3), device=pts.device)
    J[:, 0, 0] = fx / Z_c_safe
    J[:, 0, 2] = -fx * X_c / (Z_c_safe ** 2)
    J[:, 1, 1] = fy / Z_c_safe
    J[:, 1, 2] = -fy * Y_c / (Z_c_safe ** 2)

    return J  # shape: (N, 2, 3)

def transform_pts_to_frame(pts, rel_w2c, detach=True):
    """
    Function to transform points from world frame to camera frame.

    Args:
        pts: points in world frame
        rel_w2c: relative pose of current frame
        detach: whether to detach params

    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    if detach:
        pts = pts.detach()

    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).to(rel_w2c.device).float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts


def transform_to_frame(gaussians, current_w2c, pre_rot_cov=True, detach=True):

    params = gaussians.get_params

    transformed_pts = transform_pts_to_frame(params['means3D'], current_w2c, detach=detach)

    # J = compute_projection_jacobians(transformed_pts, current_w2c,camera.k[0,0], camera.K[1,1])  # shape: (N, 2, 3)

    if pre_rot_cov:
        cov_w = gaussians.get_cov  # (N, 3, 3)
        # #
        W = current_w2c[:3, :3].expand(cov_w.shape[0], -1, -1)
        #
        cov_cam = W @ cov_w.detach() @ W.transpose(1, 2)
        #
        # cov_proj = (J @ cov_cam @ J.transpose(1, 2))

        return transformed_pts, strip_lowerdiag(cov_cam)
    else:
        return transformed_pts



def vectors2frame(vec, w2c):
    """
    Takes vectors in world coordinates and returns in camera frame
    """

    return (w2c[:3,:3] @ vec.T).T


def nth_closest_index(queries: torch.Tensor, keys: torch.Tensor, n: int=1) -> torch.Tensor:
    """
    Computes the index of the Nth closest key point for each query point using torch.topk.

    args:
        -- queries (torch.Tensor): Tensor of shape (M, 3) with M query points.
        -- keys (torch.Tensor): Tensor of shape (N, 3) with N key points.
        -- n (int): Which closest point to retrieve (1 = closest, 2 = second closest, etc.).

    Returns:
        -- indices (torch.Tensor): Tensor of shape (M,) with the index of the Nth closest key point for each query.
    """
    assert n >= 1 and n <= keys.size(0), "n must be between 1 and the number of key points"

    # Compute pairwise squared Euclidean distances
    diff = queries.unsqueeze(1) - keys.unsqueeze(0)  # (M, N, 3)
    dists = (diff ** 2).sum(dim=2)  # (M, N)

    # Get the indices of the n smallest distances
    _, topk_indices = torch.topk(dists, k=n, dim=1, largest=False, sorted=True)

    # Extract the index of the Nth closest (n-1 due to 0-based indexing)
    nth_indices = topk_indices[:, n - 1]

    return nth_indices

# those are taken from MonoGS

def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    # MonoGS
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    # MonoGS
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    # MonoGS
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):
    # MonoGS
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def slerp(q0, q1, alpha):
    """
    SLERP between two unit quaternions q0, q1 (N x 4), with scalar alpha
    Returns interpolated quaternion.
    """
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)

    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # Flip for shortest path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # Compute SLERP
    theta = torch.acos(dot.clamp(-1, 1))  # angle
    sin_theta = torch.sin(theta)

    eps = 1e-6
    s0 = torch.sin((1 - alpha) * theta) / (sin_theta + eps)
    s1 = torch.sin(alpha * theta) / (sin_theta + eps)

    return s0 * q0 + s1 * q1

def project_to_screen(pcd, w2c, k, near=0.01, far=100, return_culled=False):
    pts_cam = (w2c @ pcd.T).T[:,:3]

    culling_mask = ((pts_cam[:, 2] > near) & (pts_cam[:, 2] < far)).unsqueeze(-1)
    if return_culled:
        pts_cam = pts_cam[culling_mask]

    pts_2D = (k @ pts_cam.T).T
    pts_2D = pts_2D[:,:2]/pts_2D[:,2:3]

    return pts_2D, culling_mask

def backproject_selected_points(points_2d, depth_map, intrinsics, w2c=None):
    """
    Backprojects 2D points into 3D using depth map and intrinsics.

    Args:
        points_2d: (N, 2) tensor of [x, y] pixel coordinates
        depth_map: (H, W) tensor of depth values
        intrinsics: (3, 3) camera intrinsics matrix
        w2c: (4, 4) world to cam transformation matrix to bring points in world coordinates

    Returns:
        points_3d: (N, 3) tensor of 3D points in camera coordinates
    """
    assert points_2d.shape[1] == 2, "points_2d must be Nx2"

    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze()

    # Split pixel coordinates
    u = points_2d[:, 0].long()
    v = points_2d[:, 1].long()

    # Get depth at those pixels (same order as input)
    z = depth_map[v, u]  # (N,) — use y = v, x = u indexing

    # Backproject to 3D
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x = (u.float() - cx) * z / fx
    y = (v.float() - cy) * z / fy

    points_3d = torch.stack([x, y, z], dim=-1)  # shape (N, 3)

    if isinstance(w2c, torch.Tensor):
        pix_ones = torch.ones(points_3d.shape[0], 1).float().to(points_3d.device)
        pts4 = torch.cat((points_3d, pix_ones), dim=1)
        c2w = torch.linalg.inv(w2c)#.double()
        points_3d = (c2w @ pts4.T).T[:, :3]  # world coord

    return points_3d