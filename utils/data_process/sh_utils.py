#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """

    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def compute_gaussian_rgb(params, camera_center, active_sh_degree=3):
    n = params['rgb_colors'].shape[0]
    shs_view = torch.cat((params['rgb_colors'].reshape(n, 3, -1).transpose(1, 2), params['feature_rest'].transpose(1, 2)), dim=1).transpose(1, 2)
    dir_pp = (params['means3D'] - camera_center.repeat(n, 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
    return sh2rgb +0.5 #torch.clamp_min(sh2rgb + 0.5, 0.0)

def compute_pbr_color(params, camera_center, light_center_sep=None, debug_logger=None, frame_id=None):
    """

    Compute color according to Physical Based Rendering as in PR-Endo (N.B. sh_degree is set to 3)

    args:
        -- params: gaussian parameters
        -- camera_center: camera center
        -- light_center_sep: tensor with center of light-source, if None it is assumed coincident with camera
    """

    light = params['light']

    if torch.is_tensor(light_center_sep):
        light_center_raw = light_center_sep
    else:
        light_center_raw = camera_center

    # PR-Endo uses SH to compute albedo
    if 'feature_rest' in params.keys():
        base_color = compute_gaussian_rgb(params, camera_center, 3)
    else:
        base_color = params['rgb_colors']

    # Light color is [1,1,1,]
    light_intensity, attenuation_k, attenuation_power, light_adjustment = light[0], light[1], light[2], light[3:12]

    # For debug: set manually params
    # light_intensity, attenuation_k, attenuation_power = 2, 0.3, 3

    # ------ HOW TO SET LIGHT DIRECTION - either basic version, or with d optimized
    # light dir == view dir - the basic option
    light_center = light_center_raw
    dir_pp_light = (params['means3D'] - light_center.repeat(params['means3D'].shape[0], 1))
    dir_pp_light = torch.nn.functional.normalize(dir_pp_light, dim=1)

    # --------------
    dir_gauss_lightcenter = (params['means3D'] - light_center.repeat(params['means3D'].shape[0], 1))
    light_gauss_dist = dir_gauss_lightcenter.norm(dim=1, keepdim=True) #/ 3
    # --------------

    dir_pp_camera = (params['means3D'] - camera_center.repeat(params['means3D'].shape[0], 1))

    normal = params['normals']

    # Normalize the input vectors
    N = torch.nn.functional.normalize(normal, dim=1)
    L = -torch.nn.functional.normalize(dir_pp_light, dim=1)
    V = -torch.nn.functional.normalize(dir_pp_camera, dim=1)

    # normals always towards camera
    N_dot_V = torch.sum(N * V, dim=1, keepdim=True)  # [N, 1]
    N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0

    # cosine
    N_dot_L = torch.clamp(torch.sum(N * L, dim=1, keepdim=True), min=0.0)

    # Compute distance attenuation (light intensity fades with distance)
    attenuation_raw_coeffs = 1.0 / (1.0 + attenuation_k * light_gauss_dist ** attenuation_power)
    attenuation_coeffs = torch.clamp(attenuation_raw_coeffs, 0, 1)

    # Diffuse component - from coefficients
    I_diffuse_color_coeffs = light_intensity * N_dot_L * attenuation_coeffs * base_color

    # ======== PBR reflections

    # Compute halfway vector for specular component
    H = F.normalize(L + V, dim=1)

    # Fresnel Effect (Schlick's approximation) for non-metals (F0 = 0.04) -value from gpt
    fresnel = params['F0'] + (1 - params['F0']) * (1 - torch.sum(H * V, dim=1, keepdim=True)) ** 5

    # Specular component using Cook-Torrance model
    N_dot_H = torch.clamp(torch.sum(N * H, dim=1, keepdim=True), min=0.0)
    alpha = (params['roughness'].clamp(0, 0.7)) ** 2
    # for debug set manually alpha
    # alpha = (0.3) ** 2

    # Microfacet distribution function (Trowbridge-Reitz GGX)
    D = (alpha ** 2) / (torch.pi * ((N_dot_H ** 2) * (alpha ** 2 - 1) + 1) ** 2)

    # Geometric attenuation (Smith GGX)
    k = (params['roughness'] + 1) ** 2 / 8
    N_dot_V_clamped = torch.clamp(torch.sum(N * V, dim=1, keepdim=True), min=0.0)
    N_dot_L_clamped = torch.clamp(N_dot_L, min=0.0)
    G1_V = N_dot_V_clamped / (N_dot_V_clamped * (1 - k) + k)
    G1_L = N_dot_L_clamped / (N_dot_L_clamped * (1 - k) + k)
    G = G1_V * G1_L

    # Final Cook-Torrance specular term
    specular_component_coeffs = (fresnel * D * G) / (4 * N_dot_V_clamped * N_dot_L_clamped + 1e-5)
    I_specular_coeffs = light_intensity * attenuation_coeffs * specular_component_coeffs * N_dot_L

    I_diffuse_rough, I_specular_rough = I_diffuse_color_coeffs * (1 - fresnel), I_specular_coeffs

    if debug_logger:
        debug_logger(F=(fresnel, torch.sum(H * V, dim=1, keepdim=True)), D=(D, N_dot_H),
                     G=(G, N_dot_V_clamped, N_dot_L_clamped), pbr=(I_diffuse_rough, I_specular_rough), i=frame_id)

    I_diffuse_final, I_specular_final = torch.clamp_min(I_diffuse_rough, 0), torch.clamp_min(I_specular_rough, 0)
    reflected_rgb = I_diffuse_final + I_specular_final

    return reflected_rgb


def get_pbr_params(params, camera_center, light_center_sep=None, debug_logger=None, frame_id=None):
    """

    Compute color according to Physical Based Rendering as in PR-Endo (N.B. sh_degree is set to 3)

    args:
        -- params: gaussian parameters
        -- camera_center: camera center
        -- light_center_sep: tensor with center of light-source, if None it is assumed coincident with camera
    returns:
        -- base_color_render: per gaussian albedo of rendered img
        -- diffuse_coeffs: diffusion coefficients in a shape suitable for rendering
        -- specular_coefs: in a form suitable for rendering
    """

    light = params['light']

    if torch.is_tensor(light_center_sep):
        light_center_raw = light_center_sep
    else:
        light_center_raw = camera_center

    # PR-Endo uses SH to compute albedo
    if 'feature_rest' in params.keys():
        base_color_render = compute_gaussian_rgb(params, camera_center, 3)
    else:
        base_color_render = params['rgb_colors']

    # Light color is [1,1,1,]
    light_intensity, attenuation_k, attenuation_power, light_adjustment = light[0], light[1], light[2], light[3:12]

    # For debug: set manually params
    # light_intensity, attenuation_k, attenuation_power = 2, 0.3, 3

    # ------ HOW TO SET LIGHT DIRECTION - either basic version, or with d optimized
    # light dir == view dir - the basic option
    light_center = light_center_raw
    dir_pp_light = (params['means3D'] - light_center.repeat(params['means3D'].shape[0], 1))
    dir_pp_light = torch.nn.functional.normalize(dir_pp_light, dim=1)

    # --------------
    dir_gauss_lightcenter = (params['means3D'] - light_center.repeat(params['means3D'].shape[0], 1))
    light_gauss_dist = dir_gauss_lightcenter.norm(dim=1, keepdim=True) #/ 3
    # --------------

    dir_pp_camera = (params['means3D'] - camera_center.repeat(params['means3D'].shape[0], 1))

    normal = params['normals']

    # Normalize the input vectors
    N = torch.nn.functional.normalize(normal, dim=1)
    L = -torch.nn.functional.normalize(dir_pp_light, dim=1)
    V = -torch.nn.functional.normalize(dir_pp_camera, dim=1)

    # normals always towards camera
    N_dot_V = torch.sum(N * V, dim=1, keepdim=True)  # [N, 1]
    N = torch.where(N_dot_V < 0, -N, N)  # Flip N if N_dot_V < 0

    # cosine
    N_dot_L = torch.clamp(torch.sum(N * L, dim=1, keepdim=True), min=0.0)

    # Compute distance attenuation (light intensity fades with distance)
    attenuation_raw_coeffs = 1.0 / (1.0 + attenuation_k * light_gauss_dist ** attenuation_power)
    attenuation_coeffs = torch.clamp(attenuation_raw_coeffs, 0, 1)

    # # Diffuse component - from coefficients
    # I_diffuse_color_coeffs = light_intensity * N_dot_L * attenuation_coeffs * base_color

    # ======== PBR reflections

    # Compute halfway vector for specular component
    H = F.normalize(L + V, dim=1)

    # Fresnel Effect (Schlick's approximation) for non-metals (F0 = 0.04) -value from gpt
    fresnel = params['F0'] + (1 - params['F0']) * (1 - torch.sum(H * V, dim=1, keepdim=True)) ** 5

    # Specular component using Cook-Torrance model
    N_dot_H = torch.clamp(torch.sum(N * H, dim=1, keepdim=True), min=0.0)
    alpha = (params['roughness'].clamp(0, 0.7)) ** 2
    # for debug set manually alpha
    # alpha = (0.3) ** 2

    # Microfacet distribution function (Trowbridge-Reitz GGX)
    D = (alpha ** 2) / (torch.pi * ((N_dot_H ** 2) * (alpha ** 2 - 1) + 1) ** 2)

    # Geometric attenuation (Smith GGX)
    k = (params['roughness'] + 1) ** 2 / 8
    N_dot_V_clamped = torch.clamp(torch.sum(N * V, dim=1, keepdim=True), min=0.0)
    N_dot_L_clamped = torch.clamp(N_dot_L, min=0.0)
    G1_V = N_dot_V_clamped / (N_dot_V_clamped * (1 - k) + k)
    G1_L = N_dot_L_clamped / (N_dot_L_clamped * (1 - k) + k)
    G = G1_V * G1_L

    # Final Cook-Torrance specular term
    specular_component_coeffs = (fresnel * D * G) / (4 * N_dot_V_clamped * N_dot_L_clamped + 1e-5)
    I_specular_coeffs = light_intensity * attenuation_coeffs * specular_component_coeffs * N_dot_L

    I_specular_final = torch.clamp_min(I_specular_coeffs, 0)

    coeffs = torch.cat((N_dot_L * light_intensity*(1-fresnel), attenuation_coeffs, I_specular_final), dim=1) # Nx3



    return base_color_render, coeffs


