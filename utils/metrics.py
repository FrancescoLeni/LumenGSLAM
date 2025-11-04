import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import lpips
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.models import AlexNet_Weights

from .data_process.pose_handling import align


LPIPS_AlexNet = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def calculate_ssim(img1, img2, window_size=11, size_average=True):

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    """
    Calculates the PSNR of two NORMALIZED images.
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()

    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def calculate_mssim(img1, img2):
    """
    Calculates the MS-SSIM between two images using PyTorch (NORMALIZED).
    """

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    ms_ssim_value = ms_ssim(img1, img2, data_range=1.0)
    return ms_ssim_value

lpips_model = lpips.LPIPS(net='vgg').cuda()
def calculate_lpips(img1, img2):
    """
    Calculates the LPIPS between two NORMALIZED images.
    """

    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()

    lpips_distance = lpips_model(img1, img2)
    return lpips_distance.item()


def get_psnr_ssim_lpips(img1: torch.tensor, img2: torch.tensor):
    """
    input images are expected to be Channel first
    """



    psnr = calc_psnr(img1, img2).mean()
    ssim = calculate_ssim(img1.unsqueeze(0).cpu(), img2.unsqueeze(0).cpu())
    lpips_score = calculate_lpips(torch.clamp(img1.unsqueeze(0), 0.0, 1.0).to(torch.float32),
                                  torch.clamp(img2.unsqueeze(0), 0.0, 1.0).to(torch.float32))
    mssim = calculate_mssim(img1.unsqueeze(0).cpu().to(torch.float64), img2.unsqueeze(0).cpu().to(torch.float64)).mean()


    return psnr, ssim, lpips_score, mssim


def evaluate_ate(gt_traj, est_traj):
    """
    Input :
        gt_traj: list of 4x4 matrices
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.stack(gt_traj_pts).detach().cpu().numpy().T
    est_traj_pts = torch.stack(est_traj_pts).detach().cpu().numpy().T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error