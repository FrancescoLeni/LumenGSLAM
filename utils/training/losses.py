
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable

from utils.data_process.image_handling import rgb2gray
from utils.data_process.preprocessing import normals_from_depth, compute_image_gradients
from utils.data_process.geometry import project_to_screen, backproject_selected_points


class BaseLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.running_dict = {'loss': 0.}

    def forward(self, **kwargs):
        raise NotImplementedError('implement it in child class')

    def accumulate(self, batch_loss, **kwargs):
        self.running_dict['loss'] += batch_loss.item()

    def get_current_value(self, batch_n, **kwargs):
        n = batch_n + 1  # batch_n is [0, N-1]
        return self.running_dict['loss'] / n

    def reset(self, **kwargs):
        self.running_dict['loss'] = 0.

class MAELoss(BaseLoss):
    def __init__(self):
        super().__init__()

        self.running_dict = {'loss': 0.}

    def forward(self, x, y, mask=None, reduction='mean'):
        loss = ((x - y) ** 2)

        if isinstance(mask, torch.Tensor):
            loss = loss[mask]

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError

        self.accumulate(loss)
        return loss


class l1_Loss(BaseLoss):
    def __init__(self):
        super().__init__()

        self.running_dict = {'loss': 0.}

    def forward(self, x, y, mask=None, reduction='mean', weights=None):
        """
        Args:
            x: rendered data
            y: gt data
            mask: bool tensor to mask out invalid pixels
            reduction: 'mean' or 'sum' (intended pixelwise for 1 image)
            weights: weights for pixels
        Returns:
            Loss: loss averaged
        """

        loss = torch.abs((x - y))

        if weights is not None:
            loss = loss * weights

        if isinstance(mask, torch.Tensor):
            loss = loss[mask]

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError

        self.accumulate(loss)
        return loss


class ReprojectionLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.running_dict = {'loss': 0.}

    def forward(self, match_3D_0, match_2D_1, w2c_1, K):

        assert match_3D_0.shape[0] == match_2D_1.shape[0], 'same number of points is needed!'

        match_3D_0_homo = torch.cat((match_3D_0, torch.ones((match_3D_0.shape[0], 1)).to(match_3D_0.device)), dim=-1)
        pts_0_2D, culling_mask = project_to_screen(match_3D_0_homo, w2c_1, K, return_culled=False)

        # loss = torch.mean(torch.sum((pts_0_2D[culling_mask.tile(1,2)] - match_2D_1[culling_mask.tile(1,2)]) ** 2, dim=-1)) # mse
        # loss = torch.mean(torch.sum((pts_0_2D[culling_mask.tile(1, 2)] - match_2D_1[culling_mask.tile(1, 2)]).abs(), dim=-1))  # mae
        # loss = torch.sum((pts_0_2D[culling_mask.tile(1, 2)] - match_2D_1[culling_mask.tile(1, 2)]).abs()) # l1
        loss = torch.sum(torch.norm((pts_0_2D[culling_mask.tile(1,2)] - match_2D_1[culling_mask.tile(1,2)]), p=2, dim=-1))  # l2

        self.accumulate(loss)
        return loss

class BackprojectionLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.running_dict = {'loss': 0.}

    def forward(self, match_3D_0, match_2D_1, depth_1, w2c_1, K):
        assert match_3D_0.shape[0] == match_2D_1.shape[0], 'same number of points is needed!'

        pts_3D_1 = backproject_selected_points(match_2D_1, depth_1, K, w2c_1)

        # loss = torch.mean(torch.sum((pts_3D_1 - match_3D_0) ** 2, dim=-1))  # mse
        # loss = torch.mean(torch.sum((pts_3D_1 - match_3D_0).abs(), dim=-1))  # mae
        # loss = torch.sum((pts_3D_1 - match_3D_0).abs())  # l1
        loss = torch.mean(torch.norm((pts_3D_1 - match_3D_0), p=2, dim=-1))  # l2

        self.accumulate(loss)
        return loss


class GeometricProjLoss(BaseLoss):
    def __init__(self, gain_bkprj=0.2, gain_reproj=1.0):
        super().__init__()
        self.running_dict = {'loss': 0., 'backprojection_loss': 0., 'reprojection_loss': 0.}
        self.current_dict = {'loss': 0., 'backprojection_loss': 0., 'reprojection_loss': 0.}

        self.bkprj_loss = BackprojectionLoss()
        self.reprj_loss = ReprojectionLoss()

        self.gain_bkprj = gain_bkprj
        self.gain_reproj = gain_reproj

    def forward(self, match_3D_0, match_2D_1, depth_1, w2c_1, K):
        assert match_3D_0.shape[0] == match_2D_1.shape[0], 'same number of points is needed!'

        reprj_loss = self.gain_reproj * self.reprj_loss(match_3D_0, match_2D_1, w2c_1, K)
        bkprj_loss = self.gain_bkprj * self.bkprj_loss(match_3D_0, match_2D_1, depth_1, w2c_1, K)

        loss = reprj_loss + bkprj_loss

        self.current_dict = {'loss': loss.item(), 'backprojection_loss': bkprj_loss.item(), 'reprojection_loss': reprj_loss.item()}
        self.accumulate(loss, bkprj_loss, reprj_loss)

        return loss

    def accumulate(self, batch_loss, batch_bkprj, batch_reprj):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['backprojection_loss'] += batch_bkprj.item()
        self.running_dict['reprojection_loss'] += batch_reprj.item()

    def reset(self, **kwargs):
        self.running_dict = {'loss': 0., 'backprojection_loss': 0., 'reprojection_loss': 0.}

class l2_Loss(BaseLoss):
    def __init__(self):
        super().__init__()

        self.running_dict = {'loss': 0.}

    def forward(self, x, y, mask=None, reduction='mean', weights=None):
        """
        Args:
            x: rendered data
            y: gt data
            mask: bool tensor to mask out invalid pixels
            reduction: 'mean' or 'sum' (intended pixelwise for 1 image)
            weights: weights for pixels
        Returns:
            Loss: loss averaged
        """

        loss = ((x - y)**2)

        if weights is not None:
            loss = loss * weights

        if isinstance(mask, torch.Tensor):
            loss = loss[mask]

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError

        self.accumulate(loss)
        return loss


class HuberLoss(BaseLoss):
    def __init__(self):
        super().__init__()

        self.running_dict = {'loss': 0.}

    def forward(self, x, y, mask=None, reduction='mean'):
        """
        Args:
            x: rendered data
            y: gt data
            mask: bool tensor to mask out invalid pixels
            reduction: 'mean' or 'sum' (intended pixelwise for 1 image)
        Returns:
            Loss: loss averaged
        """

        # delta of GS pancakes
        loss = F.huber_loss(x,y, reduction='none', delta=0.2)

        if isinstance(mask, torch.Tensor):
            loss = loss[mask]

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError

        self.accumulate(loss)
        return loss


class DSSIM_Loss(BaseLoss):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.running_dict = {'loss': 0.}

    def forward(self, x, y, mask=None):

        DSSIM = 1. - self.calc_ssim(x, y, mask=mask)
        self.accumulate(DSSIM)
        return DSSIM

    def calc_ssim(self, img1, img2, size_average=True, mask=None):
        channel = img1.size(-3)
        window = self.create_window(channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, channel, size_average, mask)

    def gaussian(self, sigma=1.5):
        gauss = torch.Tensor([math.exp(-(x - self.window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(self.window_size)])
        return gauss / gauss.sum()

    def create_window(self, channel):
        _1D_window = self.gaussian().unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, channel, size_average=True, mask=None):

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if isinstance(mask, torch.Tensor):
            ssim_map = ssim_map[mask]

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class RenderedSurfaceLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.running_dict = {'loss': 0.}

    def forward(self, x, y, mask=None, reduction='mean'):
        """
        surface loss estimating surface's normals from gt depth (as Endo-4DGS) with cosine similarity (as pancakes)

        args:
            -- x: rendered norms (3,H,W)
            -- y: gt depth (1,H,W)
        """

        _, h, w = x.shape

        x = F.normalize(x, dim=0)

        gt_norms = normals_from_depth(y, mode='central')  # (3,H,W)

        cosine_sim = (x.view(-1, h*w).T * gt_norms.view(-1,h*w).T).sum(dim=1).T.view(-1, h, w)

        loss = 1 - cosine_sim.abs() #(pancakes has abs) why??????

        # if isinstance(mask, torch.Tensor):
        #     loss = loss[mask]

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        else:
            raise NotImplementedError

        self.accumulate(loss)
        return loss

class TrackingLoss(torch.nn.Module):
    def __init__(self, gains_dict):
        super().__init__()
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normal_loss': 0.}
        self.current_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0.}

        self.reconstruction_gain = gains_dict['reconstruction_gain']
        self.depth_gain = gains_dict['depth_gain']
        self.norm_gains = gains_dict['norm_gain']
        self.use_normal_loss = gains_dict['use_normal_loss']
        self.mask_photo = gains_dict['mask_photometric_loss'] if "mask_photometric_loss" in gains_dict else False


        self.depth_l1 = HuberLoss()
        self.img_l1 = l1_Loss()

        self.normal_loss = RenderedSurfaceLoss()

    def forward(self, x: tuple, y:tuple, mask=None, use_weight=False):
        im_x, depth_x, norm_x = x
        im_y, depth_y = y

        im_x = im_x.clip(0,1)

        if not isinstance(mask, torch.Tensor):
            mask = torch.ones_like(depth_x)

        # plt.imshow(mask.detach().squeeze().cpu().numpy().astype(np.uint8))
        # plt.savefig('mask.png')

        if use_weight:
            im_gray = rgb2gray(im_y)
            g_x, g_y = compute_image_gradients(im_gray)
            gradient_magnitude_sq = g_x ** 2 + g_y ** 2  # [B, 1, H, W]
            weights = gradient_magnitude_sq / (gradient_magnitude_sq.max() + 1e-6)
            weights = weights.squeeze(0) # [1,H,W]
        else:
            weights = None

        if self.mask_photo:
            reconstruction_loss = self.reconstruction_gain * self.img_l1(im_x, im_y, reduction='sum', mask=torch.tile(mask, (3, 1, 1)), weights=weights)
        else:
            reconstruction_loss = self.reconstruction_gain * self.img_l1(im_x, im_y, reduction='sum', weights=weights)

        depth_loss =  self.depth_gain * self.depth_l1(depth_x, depth_y, reduction='sum', mask=mask)
        loss = reconstruction_loss + depth_loss

        # normal loss
        if isinstance(norm_x, torch.Tensor) and  self.use_normal_loss:
            normal_loss = self.norm_gains * self.normals_loss(norm_x, depth_y, reduction='mean', mask=mask)
            loss += normal_loss
        else:
            normal_loss = torch.tensor([0.])


        self.current_dict = {'loss': loss.item(), 'reconstruction_loss': reconstruction_loss.item(), 'depth_loss': depth_loss.item(),
                             'normals_loss': normal_loss.item()}
        self.accumulate(loss, reconstruction_loss, depth_loss, normal_loss)
        return loss

    def accumulate(self, batch_loss, batch_rec, batch_depth, normal_loss):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['reconstruction_loss'] += batch_rec.item()
        self.running_dict['depth_loss'] += batch_depth.item()
        self.running_dict['normal_loss'] += normal_loss.item()


    def get_current_value(self, batch_n, **kwargs):
        n = batch_n + 1  # batch_n is [0, N-1]
        out = {k: self.running_dict[k] / n for k in self.running_dict.keys()}
        return out

    def reset(self, **kwargs):
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normal_loss': 0.}


class TrackingLossHybrid(torch.nn.Module):
    def __init__(self, gains_dict):
        super().__init__()
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normal_loss': 0., 'proj_loss': 0.}
        self.current_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0., 'proj_loss': 0.}

        self.reconstruction_gain = gains_dict['reconstruction_gain']
        self.depth_gain = gains_dict['depth_gain']
        self.norm_gains = gains_dict['norm_gain']
        self.proj_gain = gains_dict['proj_gain']
        self.use_normal_loss = gains_dict['use_normal_loss']
        self.mask_photo = gains_dict['mask_photometric_loss'] if "mask_photometric_loss" in gains_dict else False


        self.depth_l1 = HuberLoss()
        self.img_l1 = l1_Loss()
        self.proj_loss = GeometricProjLoss(gain_bkprj=0.2, gain_reproj=1.0)

        self.normal_loss = RenderedSurfaceLoss()

    def forward(self, x: tuple, y:tuple, geom_pack=(None, None, None, None), mask=None, use_weight=False):
        im_x, depth_x, norm_x = x
        im_y, depth_y = y

        match_3D_0, match_2D_1, w2c_1, K = geom_pack

        im_x = im_x.clip(0,1)

        if not isinstance(mask, torch.Tensor):
            mask = torch.ones_like(depth_x)

        # plt.imshow(mask.detach().squeeze().cpu().numpy().astype(np.uint8))
        # plt.savefig('mask.png')

        if use_weight:
            im_gray = rgb2gray(im_y)
            g_x, g_y = compute_image_gradients(im_gray)
            gradient_magnitude_sq = g_x ** 2 + g_y ** 2  # [B, 1, H, W]
            weights = gradient_magnitude_sq / (gradient_magnitude_sq.max() + 1e-6)
            weights = weights.squeeze(0) # [1,H,W]
        else:
            weights = None

        if self.mask_photo:
            reconstruction_loss = self.reconstruction_gain * self.img_l1(im_x, im_y, reduction='mean', mask=torch.tile(mask, (3, 1, 1)), weights=weights)
        else:
            reconstruction_loss = self.reconstruction_gain * self.img_l1(im_x, im_y, reduction='mean', weights=weights)

        depth_loss =  self.depth_gain * self.depth_l1(depth_x, depth_y, reduction='mean', mask=mask)
        loss = reconstruction_loss + depth_loss

        # normal loss
        if isinstance(norm_x, torch.Tensor) and  self.use_normal_loss:
            normal_loss = self.norm_gains * self.normals_loss(norm_x, depth_y, reduction='mean', mask=mask)
            loss += normal_loss
        else:
            normal_loss = torch.tensor([0.])

        # geometric projection loss
        if self.proj_gain > 0.:
            proj_loss = self.proj_gain * self.proj_loss(match_3D_0, match_2D_1, depth_x, w2c_1, K)
            loss += proj_loss
        else:
            proj_loss = torch.tensor([0.])

        self.current_dict = {'loss': loss.item(), 'reconstruction_loss': reconstruction_loss.item(), 'depth_loss': depth_loss.item(),
                             'normals_loss': normal_loss.item(), 'proj_loss': proj_loss.item()}
        self.accumulate(loss, reconstruction_loss, depth_loss, normal_loss, proj_loss)
        return loss

    def accumulate(self, batch_loss, batch_rec, batch_depth, normal_loss, proj_loss):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['reconstruction_loss'] += batch_rec.item()
        self.running_dict['depth_loss'] += batch_depth.item()
        self.running_dict['normal_loss'] += normal_loss.item()
        self.running_dict['proj_loss'] += proj_loss.item()


    def get_current_value(self, batch_n, **kwargs):
        n = batch_n + 1  # batch_n is [0, N-1]
        out = {k: self.running_dict[k] / n for k in self.running_dict.keys()}
        return out

    def reset(self, **kwargs):
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normal_loss': 0., 'proj_loss': 0.}


class MappingLoss(torch.nn.Module):
    def __init__(self, gains_dict):
        super().__init__()
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0.}
        self.current_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0.}

        self.reconstruction_gain = gains_dict['reconstruction_gain']
        self.depth_gain = gains_dict['depth_gain']
        self.norm_gains = gains_dict['norm_gain']
        self.mask_photo = gains_dict['mask_photometric_loss'] if "mask_photometric_loss" in gains_dict else False

        self.depth_l1 = HuberLoss()
        self.img_l1 = l1_Loss()
        self.DSSIM = DSSIM_Loss()
        self.normals_loss = RenderedSurfaceLoss()

    def forward(self, x: tuple, y: tuple, mask=None):
        # x -> rendered; y -> gt

        im_x, depth_x, norm_x = x
        im_y, depth_y = y

        if not isinstance(mask, torch.Tensor):
            mask = torch.ones_like(depth_x).requires_grad_(False)

        if self.mask_photo:
            reconstruction_loss = self.reconstruction_gain * (0.8 * self.img_l1(im_x, im_y, mask.tile((3,1,1))) + 0.2 * self.DSSIM(im_x, im_y, mask.tile((3,1,1))))
        else:
            reconstruction_loss = self.reconstruction_gain * (0.8 * self.img_l1(im_x, im_y) + 0.2 * self.DSSIM(im_x, im_y))

        depth_loss =  self.depth_gain * self.depth_l1(depth_x, depth_y, reduction='mean', mask=mask)

        # not using mask for img as EndoGSLAM
        loss = reconstruction_loss + depth_loss

        if isinstance(norm_x, torch.Tensor):
            normal_loss = self.norm_gains * self.normals_loss(norm_x, depth_y, reduction='mean', mask=mask)
            # loss += normal_loss
        else:
            normal_loss = torch.tensor([0.])

        # debug
        # print(im_x.dtype, im_y.dtype, depth_x.dtype, depth_y.dtype)
        # print(loss.dtype, reconstruction_loss.dtype, depth_loss.dtype)

        self.current_dict = {'loss': loss.item(), 'reconstruction_loss': reconstruction_loss.item(),
                             'depth_loss': depth_loss.item(), 'normals_loss': normal_loss.item()}

        self.accumulate(loss, reconstruction_loss, depth_loss, normal_loss)
        return loss

    def accumulate(self, batch_loss, batch_rec, batch_depth, batch_norm):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['reconstruction_loss'] += batch_rec.item()
        self.running_dict['depth_loss'] += batch_depth.item()
        self.running_dict['normals_loss'] += batch_norm.item()

    def get_current_value(self, batch_n, **kwargs):
        n = batch_n + 1  # batch_n is [0, N-1]
        out = {k: self.running_dict[k] / n for k in self.running_dict.keys()}
        return out

    def reset(self, **kwargs):
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0.}

class MappingLossPBR(torch.nn.Module):
    def __init__(self, gains_dict):
        super().__init__()
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0., 'light_loss': 0.}
        self.current_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0., 'light_loss': 0.}

        self.reconstruction_gain = gains_dict['reconstruction_gain']
        self.depth_gain = gains_dict['depth_gain']
        self.norm_gain = gains_dict['norm_gain']
        self.light_gain = gains_dict['light_gain']
        self.use_normal_loss = gains_dict['use_normal_loss']
        self.mask_photo = gains_dict['mask_photometric_loss'] if "mask_photometric_loss" in gains_dict else False


        self.depth_l1 = HuberLoss()
        self.img_l1 = l1_Loss()
        self.DSSIM = DSSIM_Loss()
        self.normals_loss = RenderedSurfaceLoss()

        self.albedo_loss = MAELoss()
        self.roughness_loss = MAELoss()
        self.reflectivity_loss = MAELoss()

    def forward(self, x: tuple, y: tuple, pbr_params: tuple, mask=None):
        # x -> rendered; y -> gt

        im_x, depth_x, norm_x = x
        im_y, depth_y = y

        # ((albedo, roughness, F0), surf_pack) = pbr_params
        albedo, roughness, F0 = pbr_params

        if not isinstance(mask, torch.Tensor):
            mask = torch.ones_like(depth_x).requires_grad_(False)

        if self.mask_photo:
            reconstruction_loss = self.reconstruction_gain * (0.8 * self.img_l1(im_x, im_y, mask.tile((3,1,1))) + 0.2 * self.DSSIM(im_x, im_y, mask.tile((3,1,1))))
        else:
            reconstruction_loss = self.reconstruction_gain * (0.8 * self.img_l1(im_x, im_y) + 0.2 * self.DSSIM(im_x, im_y))

        depth_loss =  self.depth_gain * self.depth_l1(depth_x, depth_y, reduction='mean', mask=mask)

        # not using mask for img as EndoGSLAM
        loss = reconstruction_loss + depth_loss

        if isinstance(norm_x, torch.Tensor) and self.use_normal_loss:
            # normal_loss = self.norm_gain * self.normals_loss(surf_pack, depth_y, reduction='mean', mask=mask)
            normal_loss = self.norm_gain * self.normals_loss(norm_x, depth_y, reduction='mean', mask=mask)
            loss += normal_loss
        else:
            normal_loss = torch.tensor([0.])

        albedo_loss = self.albedo_loss(albedo, albedo.mean(dim=0))
        roughness_loss = self.roughness_loss(roughness, roughness.mean(dim=0))
        reflectivity_loss = self.reflectivity_loss(F0, F0.mean(dim=0))

        light_loss =  self.light_gain * (albedo_loss + roughness_loss + reflectivity_loss)
        loss += light_loss

        # print(reconstruction_loss.item(), depth_loss.item(), normal_loss.item(),
        #       albedo_loss.item(), roughness_loss.item(), reflectivity_loss.item())

        # debug
        # print(im_x.dtype, im_y.dtype, depth_x.dtype, depth_y.dtype)
        # print(loss.dtype, reconstruction_loss.dtype, depth_loss.dtype)

        self.current_dict = {'loss': loss.item(), 'reconstruction_loss': reconstruction_loss.item(),
                             'depth_loss': depth_loss.item(), 'normals_loss': normal_loss.item(),
                             'light_loss': light_loss.item()}

        self.accumulate(loss, reconstruction_loss, depth_loss, normal_loss, light_loss)
        return loss

    def accumulate(self, batch_loss, batch_rec, batch_depth, batch_norm, batch_light):
        self.running_dict['loss'] += batch_loss.item()
        self.running_dict['reconstruction_loss'] += batch_rec.item()
        self.running_dict['depth_loss'] += batch_depth.item()
        self.running_dict['normals_loss'] += batch_norm.item()
        self.running_dict['light_loss'] += batch_light.item()

    def get_current_value(self, batch_n, **kwargs):
        n = batch_n + 1  # batch_n is [0, N-1]
        out = {k: self.running_dict[k] / n for k in self.running_dict.keys()}
        return out

    def reset(self, **kwargs):
        self.running_dict = {'loss': 0., 'reconstruction_loss': 0., 'depth_loss': 0., 'normals_loss': 0., 'light_loss': 0.}