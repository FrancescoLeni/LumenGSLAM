import torch


def energy_mask(color: torch.Tensor, th_1=0.1, th_2=0.9):
    """
    creates a mask for pixels with brightness out of [th_1, th_2] range
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=color.device).view(3, 1, 1)
    gray = torch.sum(color * weights, dim=0).detach() # mask should not have grad
    # print(gray.max())
    zero_mask = torch.where((gray >= th_1) & (gray <= th_2), torch.tensor([True], device=color.device), torch.tensor([False], device=color.device))[None]
    # Image.fromarray(np.uint8(zero_mask[0].detach().cpu().numpy()*255), 'L').save('mask.png')

    return zero_mask
    # return torch.ones_like(zero_mask).to(color.device)