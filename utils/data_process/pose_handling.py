import numpy as np
import torch
import torch.nn.functional as F
import cv2
from matching.viz import plot_matches

from LumenGSLAM.models.feature_matcher import sample_sparse_matches_3d2d_fuzzy, get_keypoints_3D_2D

def align(model, data):
    """
    Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    # print(model.shape, data.shape)

    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def quat2rot(q):

    """
    builds a rotation matrix from unit quaternions
    args:
        -- q: normalized quaternions ( F.normalize(GS['unorm_rots']) ) batched
    return:
        --rot: torch.tensor 3x3 rotation matrix
    """

    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device=q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def pose_from_rot_trasl(quat_rot, trasl):
    """
    builds pose (w2c assuming points optimized in world coord) from quaternions and translation
    args:
        -- quat_rot: unnormalized quaternions
        -- trasl: translation vector
    return:
        -- w2c: torch.tensor 4x4 camera pose (w2c)
    """

    quat_rot = F.normalize(quat_rot)
    w2c = torch.eye(4).to(quat_rot.device).float()
    w2c[:3, :3] = quat2rot(quat_rot)
    w2c[:3, 3] = trasl

    return w2c


def rot2quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quat2eul(q: torch.Tensor, degrees: bool = True):
    """
    Converts a quaternion to Euler angles (roll, pitch, yaw) using intrinsic XYZ convention.

    Args:
        q (torch.Tensor): Quaternion tensor of shape (..., 4) in (x, y, z, w) format.
        degrees (bool): If True, return angles in degrees. Otherwise, radians.

    Returns:
        roll, pitch, yaw (torch.Tensor): Euler angles in radians or degrees.
    """
    assert q.shape[-1] == 4, "Quaternion must have shape (..., 4)"

    x, y, z, w = q.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))  # Clamp for numerical stability

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    if degrees:
        roll = torch.rad2deg(roll)
        pitch = torch.rad2deg(pitch)
        yaw = torch.rad2deg(yaw)

    return roll, pitch, yaw

def rotation_matrices(angle_x, angle_y, angle_z, mode="deg", device="cpu"):
    """
    Generate 3D rotation matrices around the X, Y, and Z axes.

    Args:
    -- angle_x (float or torch.Tensor): Rotation angle around the X-axis.
    -- angle_y (float or torch.Tensor): Rotation angle around the Y-axis.
    -- angle_z (float or torch.Tensor): Rotation angle around the Z-axis.
    -- mode (str, optional): "rad" for radians or "deg" for degrees.
    -- device (torch.device, optional): Device to use.

    Returns:
    -- Rx (torch.Tensor): 3x3 rotation matrix for the X-axis.
    -- Ry (torch.Tensor): 3x3 rotation matrix for the Y-axis.
    -- Rz (torch.Tensor): 3x3 rotation matrix for the Z-axis.
    """

    angle_x = torch.tensor(angle_x)
    angle_y = torch.tensor(angle_y)
    angle_z = torch.tensor(angle_z)

    if mode == "deg":
        angle_x = torch.deg2rad(angle_x)
        angle_y = torch.deg2rad(angle_y)
        angle_z = torch.deg2rad(angle_z)
    elif mode != "rad":
        raise ValueError("Invalid mode. Use 'rad' for radians or 'deg' for degrees.")

    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)

    Rx = torch.tensor([
        [1, 0,      0     ],
        [0, cos_x, -sin_x ],
        [0, sin_x,  cos_x ]
    ], dtype=torch.float32)

    Ry = torch.tensor([
        [ cos_y, 0, sin_y ],
        [ 0,     1, 0     ],
        [-sin_y, 0, cos_y ]
    ], dtype=torch.float32)

    Rz = torch.tensor([
        [cos_z, -sin_z, 0 ],
        [sin_z,  cos_z, 0 ],
        [0,      0,     1 ]
    ], dtype=torch.float32)

    return Rx.to(device), Ry.to(device), Rz.to(device)


def rotation_aligned_with_normal(z: torch.Tensor, return_quat=True) -> torch.Tensor:
    """
    Build an orthonormal basis from a unit vector z using cross product with [1,0,0].

    args:
    -- z: [N, 3] unit vector to use as the z-axis

    Returns:
    -- [N, 4] quaternions if return_quat=True
    -- [N, 3, 3] rotation matrix (x, y, z as columns) if return_quat==False
    """
    num_pts = z.shape[0]

    a = torch.tensor([1.0, 0.0, 0.0], device=z.device).expand(num_pts, 3)

    # Find entries where z is too close to a — swap those to [0, 1, 0]
    close = torch.abs((z * a).sum(dim=1)) > 0.99
    a[close] = torch.tensor([0.0, 1.0, 0.0], device=z.device)

    # x = normalize(cross(a, z))
    x = torch.cross(a, z, dim=1)
    x = F.normalize(x, dim=1)

    # y = cross(z, x)
    y = torch.cross(z, x, dim=1)

    rots = torch.stack([x, y, z], dim=2) # [N, 3, 3]

    # Combine as columns
    return rot2quat(rots) if return_quat else rots

#
# def rotation_aligned_with_normal(normals: torch.Tensor, return_quaternions=True) -> torch.Tensor:
#     """
#     Computes rotation matrices that align z-axis to normals using Rodrigues' formula.
#
#     args:
#     -- normals: [N, 3] tensor of unit normals.
#     -- return_quaternions: If True, return quaternion rotation.
#
#     Returns:
#     -- [N, 4] quaternions rotation if return_quaternions is True.
#     -- [N, 3, 3] rotation if return_quaternions is False.
#     """
#     N = normals.shape[0]
#     print(normals.shape)
#     z_axis = torch.tensor([0, 0, 1], device=normals.device).expand(N, 3).float()
#
#     v = F.normalize(torch.cross(z_axis, normals), dim=-1)
#     c = (z_axis * normals).sum(dim=-1, keepdim=True)
#     s = v.norm(dim=-1, keepdim=True)
#
#     # Rodrigues' formula components
#     vx = torch.zeros(N, 3, 3, device=normals.device)
#     vx[:, 0, 1] = -v[:, 2]
#     vx[:, 0, 2] =  v[:, 1]
#     vx[:, 1, 0] =  v[:, 2]
#     vx[:, 1, 2] = -v[:, 0]
#     vx[:, 2, 0] = -v[:, 1]
#     vx[:, 2, 1] =  v[:, 0]
#
#     eye = torch.eye(3, device=normals.device).unsqueeze(0).expand(N, 3, 3)
#
#     vx2 = torch.bmm(vx, vx)
#     rot_mats = eye + vx + vx2 * ((1 - c) / (s ** 2 + 1e-8))
#
#     # Handle cases where normal is exactly z or -z
#     aligned = (1 - c.squeeze()).abs() < 1e-4
#     rot_mats[aligned] = torch.eye(3, device=normals.device)
#
#     return rot2quat(rot_mats) if return_quaternions else rot_mats


def align_features_pnp(frame0: tuple, frame1, K, matcher, save_path=None, sample_spread=False, debug_img=None):
    """
    compute the pose of a frame respect the world frame given a reference view.
      -- first sparse points are matched between i0 and i1 using a feature matcher provided
      -- then the matched points are backprojected from i0,d0 and mapped to world coordinates using p0
      -- PnP registration is then performed to get the pose of frame1 in world coordinates

    args:
        - frame0: (image1, depth1, pose1) frame, depth and pose relative to the ref view
        - frame1: frame of the view to register
        - K: intrinsic matrix of camera
        - matcher: matching algorithm to use
        - save_path: (optional) path to save the matching results
        - sample_spread: whether to sample matches maximizing spread around the centroid

    """
    i0, d0, p0 = frame0
    i1 = frame1

    pts_0, inlier_kpts1, result = get_keypoints_3D_2D(matcher, i0, d0, p0, i1, K)

    if sample_spread:
        _, _, indices = sample_sparse_matches_3d2d_fuzzy(pts_0, inlier_kpts1, num_samples=17, fuzziness=0.3)
    else:
        indices = torch.ones(pts_0.shape[0]).to(bool)

    if save_path:
        result['inlier_kpts1'] = result['inlier_kpts1'][indices.cpu()]
        result['inlier_kpts0'] = result['inlier_kpts0'][indices.cpu()]
        if debug_img is not None:
            i0 = debug_img
        plot_matches(i0, i1, result, save_path=save_path)

    w2c_np = np.eye(4)

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_0[indices, :3].float().cpu().numpy(),
            inlier_kpts1[indices].float().cpu().numpy(),
            K.float().cpu().numpy(),
            None,
            reprojectionError=2.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
            # flags=cv2.SOLVEPNP_EPNP
        )
        # success, rvec, tvec = cv2.solvePnP(
        #     pts_0[indices, :3].float().cpu().numpy(),
        #     inlier_kpts1[indices].float().cpu().numpy(),
        #     K.float().cpu().numpy(),
        #     None,
        #     # reprojectionError=4.0,
        #     flags=cv2.SOLVEPNP_ITERATIVE,
        #     # flags=cv2.SOLVEPNP_EPNP
        # )
        # success, rvec, tvec, inliers = ransac_pnp(
        #     pts_0[:, :3].float().cpu().numpy(),
        #     inlier_kpts1,
        #     K.float().cpu().numpy(),
        #     None,
        #     reproj_threshold=2.0,
        #
        #     )

        R_np, _ = cv2.Rodrigues(rvec)
        t_np = tvec

        w2c_np[:3, :3] = R_np
        w2c_np[:3, 3] = np.squeeze(t_np)
    except cv2.error:
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_0[indices, :3].float().cpu().numpy(),
                inlier_kpts1[indices].float().cpu().numpy(),
                K.float().cpu().numpy(),
                None,
                reprojectionError=4.0,
                flags=cv2.SOLVEPNP_ITERATIVE,
                # flags=cv2.SOLVEPNP_EPNP
            )
            # success, rvec, tvec = cv2.solvePnP(
            #     pts_0[indices, :3].float().cpu().numpy(),
            #     inlier_kpts1[indices].float().cpu().numpy(),
            #     K.float().cpu().numpy(),
            #     None,
            #     # reprojectionError=4.0,
            #     flags=cv2.SOLVEPNP_ITERATIVE,
            #     # flags=cv2.SOLVEPNP_EPNP
            # )
            # success, rvec, tvec, inliers = ransac_pnp(
            #     pts_0[:, :3].float().cpu().numpy(),
            #     inlier_kpts1,
            #     K.float().cpu().numpy(),
            #     None,
            #     reproj_threshold=4.0,
            #
            # )

            R_np, _ = cv2.Rodrigues(rvec)
            t_np = tvec

            w2c_np[:3, :3] = R_np
            w2c_np[:3, 3] = np.squeeze(t_np)
        except cv2.error:
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_0[indices, :3].float().cpu().numpy(),
                    inlier_kpts1[indices].float().cpu().numpy(),
                    K.float().cpu().numpy(),
                    None,
                    reprojectionError=8.0,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    # flags=cv2.SOLVEPNP_EPNP
                )
                # success, rvec, tvec = cv2.solvePnP(
                #     pts_0[indices, :3].float().cpu().numpy(),
                #     inlier_kpts1[indices].float().cpu().numpy(),
                #     K.float().cpu().numpy(),
                #     None,
                #     # reprojectionError=4.0,
                #     flags=cv2.SOLVEPNP_ITERATIVE,
                #     # flags=cv2.SOLVEPNP_EPNP
                # )
                # success, rvec, tvec, inliers = ransac_pnp(
                #     pts_0[:, :3].float().cpu().numpy(),
                #     inlier_kpts1,
                #     K.float().cpu().numpy(),
                #     None,
                #     reproj_threshold=4.0,
                #
                # )

                R_np, _ = cv2.Rodrigues(rvec)
                t_np = tvec

                w2c_np[:3, :3] = R_np
                w2c_np[:3, 3] = np.squeeze(t_np)
            except cv2.error:
                success = False

    # F = torch.tensor(np.diag([-1, 1, 1, 1])).float()

    # w2c_np[0,3] = -w2c_np[0,3]

    return success, torch.tensor(w2c_np).float(), (pts_0[:,:3].to(i0.device), inlier_kpts1.to(i0.device))


# def align_features_pnp_project(pcd, frame0: tuple, frame1, K, matcher, near=0.01, far=100):
#     """
#     compute the pose of a frame respect the world frame given a reference view.
#       -- first sparse points are matched between i0 and i1 using a feature matcher provided
#       -- then pcd is projected toward the screen and using knn to find corrispondance with the matched
#       -- PnP registration is then performed to get the pose of frame1 in world coordinates
#
#     args:
#         - pcd: global point_cloud
#         - frame0: (image1, depth1, pose1) frame, depth and pose relative to the ref view
#         - frame1: frame of the view to register
#         - K: intrinsic matrix of camera
#         - matcher: matching algorithm to use
#         - near: near plane
#         - far: far plane
#
#     """
#     i0, d0, p0 = frame0
#     i1 = frame1
#
#     result = matcher(i0, i1)
#
#     inlier_kpts0, inlier_kpts1 = result['inlier_kpts0'], result['inlier_kpts1']
#
#     match_0 = torch.tensor(np.round(inlier_kpts0).astype(int))
#     # match_1 = torch.tensor(np.round(inlier_kpts1).astype(int))
#
#     mask_0 = points_to_mask_torch(match_0, i0.shape[1:])
#     # mask_1 = points_to_mask_torch(match_1, i0.shape[1:])
#
#     # project pcd to screen
#     pts_2D = project_to_screen(pcd, p0, K, near, far)
#
#     #TODO: knn to find corrispondance (idk if it's acutally meaningfull) and get the 3D pcd
#
#
#
#     success, rvec, tvec, inliers = cv2.solvePnPRansac(
#         pts_0[:, :3].float().numpy(),
#         inlier_kpts1.float().numpy(),
#         K.float().numpy(),
#         np.zeros(5),
#         # reprojectionError=100.0,
#         flags=cv2.SOLVEPNP_ITERATIVE,
#         # flags=cv2.SOLVEPNP_EPNP
#     )
#
#     R_np, _ = cv2.Rodrigues(rvec)
#     t_np = tvec
#
#     w2c_np = np.eye(4)
#     w2c_np[:3, :3] = R_np
#     w2c_np[:3, 3] = np.squeeze(t_np)
#
#     return success, torch.tensor(w2c_np).float()

def ransac_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray = None,
    iterations: int = 100,
    reproj_threshold: float = 5.0,
    min_inliers: int = 6,
    seed: int = 0
):
    """
    Estimate camera pose using PnP with a deterministic RANSAC.

    Args:
        -- object_points (np.ndarray): Nx3 array of 3D world points.
        -- image_points (np.ndarray): Nx2 array of corresponding 2D image points.
        -- camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        -- dist_coeffs (np.ndarray): Optional distortion coefficients (default: None).
        -- iterations (int): Number of RANSAC iterations (default: 100).
        -- reproj_threshold (float): Reprojection error threshold to count as inlier (default: 5.0).
        -- min_inliers (int): Minimum number of inliers to accept a pose (default: 6).
        -- seed (int): Random seed for deterministic behavior (default: 42).

    Returns:
        -- best_rvec (np.ndarray): Rotation vector (3x1).
        -- best_tvec (np.ndarray): Translation vector (3x1).
        -- best_inliers (np.ndarray): Boolean mask of inliers.
    """
    assert object_points.shape[0] == image_points.shape[0], "Mismatched number of points."
    assert object_points.shape[1] == 3 and image_points.shape[1] == 2, "Wrong shape."

    num_points = object_points.shape[0]
    best_inlier_count = 0
    best_rvec, best_tvec = None, None
    best_inliers = None

    rng = np.random.default_rng(seed)

    for _ in range(iterations):
        # Minimal set for PnP (usually 6 for DLT, 4 for EPNP)
        sample_indices = rng.choice(num_points, 6, replace=False)
        obj_sample = object_points[sample_indices]
        img_sample = image_points[sample_indices]

        success, rvec, tvec = cv2.solvePnP(
            obj_sample, img_sample, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            continue

        # Project all points
        projected_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.squeeze()

        # Compute reprojection error
        errors = np.linalg.norm(image_points - projected_points, axis=1)
        inliers = errors < reproj_threshold
        inlier_count = np.sum(inliers)

        # Update best result
        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_rvec, best_tvec = rvec, tvec
            best_inliers = inliers

    if best_inliers is None:
        success = False
    else:
        success = True

    return success, best_rvec, best_tvec, best_inliers