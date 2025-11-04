import numpy as np

def alpha_blend_img_mask(im, mask, alpha):
    im = np.concatenate((im, np.ones((*im.shape[:2], 1))), axis=2)
    mask_rgb = np.zeros_like(im)
    mask_rgb[mask] = [1, 0, 0, 0.8]

    alpha_mask = mask_rgb[:, :, 3:4]  # (N, 1), keep dims for broadcasting
    alpha_image = im[:, :, 3:4]  # (N, 1) → usually all 1s

    # Compute output alpha
    out_alpha = alpha_mask + alpha_image * (1 - alpha_mask)

    # Blend RGB channels
    blended_rgb = (mask_rgb[:, :, :3] * alpha_mask + im[:, :, :3] * alpha_image * (1 - alpha_mask)) / out_alpha

    # # Combine RGB and alpha
    # blended = np.concatenate([blended_rgb, out_alpha], axis=1)

    return blended_rgb

def rgb2gray(img):
    # channel first
    return 0.2989 * img[0:1, :, :] + 0.5870 * img[1:2, :, :] + 0.1140 * img[2:3, :, :]

