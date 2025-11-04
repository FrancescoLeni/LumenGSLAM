import torch
import torch.nn.functional as F


def skew(w):
    """Skew-symmetric matrix for vector w (3,)"""
    zero = torch.zeros_like(w[..., 0])
    Wx = torch.stack([
        zero,    -w[..., 2], w[..., 1],
        w[..., 2], zero,    -w[..., 0],
        -w[..., 1], w[..., 0], zero
    ], dim=-1).reshape(*w.shape[:-1], 3, 3)
    return Wx

def se3_exp(xi):
    """
    Exponential map from se(3) vector to SE(3) matrix.
    xi: (..., 6) tensor with [rho(3), phi(3)]
    Returns: (..., 4, 4) transformation matrix
    """
    rho, phi = xi[..., :3], xi[..., 3:]  # translation and rotation vectors

    angle = torch.norm(phi, dim=-1, keepdim=True)
    axis = phi / (angle + 1e-8)

    # Rodrigues formula for SO(3)
    # When angle is near zero, use Taylor expansions to avoid division by zero
    skew_phi = skew(phi)
    I = torch.eye(3, device=xi.device).expand_as(skew_phi)
    angle = angle[..., 0]

    def rodrigues(angle, axis):
        # Rodrigues' rotation formula
        K = skew(axis)
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        R = cos.unsqueeze(-1).unsqueeze(-1) * I + (1 - cos).unsqueeze(-1).unsqueeze(-1) * axis.unsqueeze(-1) @ axis.unsqueeze(-2) + sin.unsqueeze(-1).unsqueeze(-1) * K
        return R

    # For small angles, approximate R ~ I + skew_phi
    small_angle = angle < 1e-5
    R = torch.where(
        small_angle.unsqueeze(-1).unsqueeze(-1),
        I + skew_phi,
        rodrigues(angle, axis)
    )

    # Left Jacobian of SO(3) for translation part
    def left_jacobian_SO3(phi):
        angle = torch.norm(phi, dim=-1, keepdim=True)
        axis = phi / (angle + 1e-8)
        K = skew(axis)
        I = torch.eye(3, device=phi.device).expand_as(K)
        angle = angle[..., 0]

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        J = torch.where(
            angle.unsqueeze(-1).unsqueeze(-1) < 1e-5,
            I + 0.5 * K,
            I + (1 - cos) / (angle ** 2 + 1e-8) * K + (angle - sin) / (angle ** 3 + 1e-8) * (K @ K)
        )
        return J

    J = left_jacobian_SO3(phi)
    t = (J @ rho.unsqueeze(-1)).squeeze(-1)

    # Compose SE(3) matrix
    bottom = torch.tensor([0, 0, 0, 1], device=xi.device, dtype=xi.dtype).expand(*xi.shape[:-1], 1, 4)
    T = torch.cat([torch.cat([R, t.unsqueeze(-1)], dim=-1), bottom], dim=-2)
    return T


def skew_inv(W):
    """Inverse of skew-symmetric matrix: R^3 from 3x3 skew matrix"""
    return torch.stack([W[..., 2, 1] - W[..., 1, 2],
                        W[..., 0, 2] - W[..., 2, 0],
                        W[..., 1, 0] - W[..., 0, 1]], dim=-1) * 0.5

def so3_log(R):
    """
    Logarithm map for SO(3) rotation matrix to rotation vector.
    R: (..., 3, 3)
    Returns: (..., 3) rotation vector phi
    """
    cos_angle = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) * 0.5
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    def small_angle_case():
        # For angle ~ 0, use approximation
        return skew_inv(R - R.transpose(-2, -1))

    def normal_case():
        return angle / (2 * torch.sin(angle)) * skew_inv(R - R.transpose(-2, -1))

    # Use small angle approximation when angle < 1e-5
    phi = torch.where(angle.unsqueeze(-1) < 1e-5, small_angle_case(), normal_case())
    return phi

def left_jacobian_SO3_inv(phi):
    """
    Inverse left Jacobian of SO(3)
    phi: (..., 3)
    Returns: (..., 3, 3)
    """
    angle = torch.norm(phi, dim=-1, keepdim=True)
    axis = phi / (angle + 1e-8)

    half_angle = 0.5 * angle
    cot_half_angle = 1.0 / torch.tan(half_angle + 1e-8)

    I = torch.eye(3, device=phi.device).expand(*phi.shape[:-1], 3, 3)
    K = skew(axis)

    small_angle = angle < 1e-5
    # For small angle approx
    J_inv = torch.where(
        small_angle.unsqueeze(-1).unsqueeze(-1),
        I - 0.5 * K + (1/12) * (K @ K),
        half_angle * cot_half_angle * I + (1 - half_angle * cot_half_angle) * (axis.unsqueeze(-1) @ axis.unsqueeze(-2)) - half_angle * K
    )
    return J_inv

def se3_log(T):
    """
    Logarithm map from SE(3) matrix to se(3) vector (6,)
    T: (..., 4, 4) transformation matrix
    Returns: (..., 6) xi = [rho(3), phi(3)]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    phi = so3_log(R)  # rotation vector

    J_inv = left_jacobian_SO3_inv(phi)  # inverse left Jacobian

    rho = torch.matmul(J_inv, t.unsqueeze(-1)).squeeze(-1)

    xi = torch.cat([rho, phi], dim=-1)
    return xi


def xi2rotm(rho, phi, w2c_init):
    """
    args:
        -- rho: translational component of se(3) vector (shape: [3])
        -- phi: rotational component of se(3) vector (shape: [3])
        -- w2c_init: initial 4x4 pose matrix (world-to-camera)

    returns:
        -- new_w2c: updated 4x4 pose matrix
    """
    tau = torch.cat([rho, phi], dim=0).unsqueeze(0)
    new_w2c = se3_exp(tau).squeeze() @ w2c_init # world perturbation
    # print(se3_exp(tau).squeeze())
    # new_w2c = w2c_init @ se3_exp(tau).squeeze() # camera perturbation
    return new_w2c