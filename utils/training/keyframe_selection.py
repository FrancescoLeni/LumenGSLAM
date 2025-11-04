import torch
import torch.nn.functional as F
import numpy as np


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX) / FX
    yy = (sampled_indices[:, 0] - CY) / FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, num_keyframes, min_percentage=0.0, pixels=1600):
    """
    Select overlapping keyframes to the current camera observation.

    Args:
        gt_depth (tensor): ground truth depth image of the current frame.
        w2c (tensor): world to camera matrix (4 x 4).
        keyframe_list (list): a list containing info for each keyframe.
        num_keyframes (int): number of overlapping keyframes to select.
        min_percentage (float): minimum percentage of points in a keyframe that are visible from current frame.
        pixels (int, optional): number of pixels to sparsely sample
            from the image of the current camera. Defaults to 1600.
    Returns:
        selected_keyframe_list (list): list of selected keyframe id.
    """
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where((gt_depth[0] > 0) & (gt_depth[0] < 1e10))  # Remove invalid depth values
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    sampled_indices = valid_depth_indices[indices]

    # Back Project the selected pixels to 3D Pointcloud
    pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

    list_keyframe = []
    for keyframeid, keyframe in enumerate(keyframe_list):
        # Get the world2cam of the keyframe
        est_w2c = keyframe['w2c']
        # Transform the 3D pointcloud to the keyframe's camera space
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (est_w2c @ pts4.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]

        # Filter out the points that are outside the image
        edge = 20
        mask = (projected_pts[:, 0] < width - edge) * (projected_pts[:, 0] > edge) * \
               (projected_pts[:, 1] < height - edge) * (projected_pts[:, 1] > edge)
        mask = mask & (points_z[:, 0] > 0)

        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum() / projected_pts.shape[0]
        list_keyframe.append({'id': keyframeid, 'percent_inside': percent_inside})

    # Sort the keyframes based on the percentage of points that are inside the image
    list_keyframe = sorted(
        list_keyframe, key=lambda i: i['percent_inside'], reverse=True)

    # Select the keyframes with percentage of points inside the image > min_percentage
    selected_keyframe_list = [keyframe_dict['id'] for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > min_percentage]
    selected_keyframe_list = list(np.random.permutation(
        np.array(selected_keyframe_list))[:num_keyframes])

    return selected_keyframe_list


def keyframe_selection_distance(time_idx, curr_position, keyframe_list, distance_current_frame_prob, n_samples):
    """
    Performs sampling based on a probability distribution and returns
    the indices of `n_samples` selected keyframes.

    Args:
        -- time_idx: current timestep (frame_id)
        -- curr_position: current position (frame's pose translation)
        -- keyframe_list (list): a list containing all available keyframes
        -- distance_current_frame_prob (float): probability of sampling current frame
        -- n_samples: The number of keyframes to be sampled.

    Returns:
        A list of `n_samples` indices of the selected keyframes.
    """
    distances = []
    time_laps = []
    curr_position = curr_position.detach().cpu().numpy()
    curr_shift = np.linalg.norm(curr_position)
    for keyframe in keyframe_list:
        est_w2c = keyframe['w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = np.linalg.norm(camera_position - curr_position)
        time_lap = time_idx - keyframe['id']
        distances.append(distance)
        time_laps.append(time_lap)

    dis2prob = lambda x, scaler: np.log2(1 + scaler / (x + scaler / 5))
    dis_prob = [dis2prob(d, curr_shift) + dis2prob(t, time_idx) for d, t in zip(distances, time_laps)]
    sum_prob = sum(dis_prob) / (1 - distance_current_frame_prob)  # distance_current_frame_prob： p_c
    norm_dis_prob = [p / sum_prob for p in dis_prob]
    norm_dis_prob.append(distance_current_frame_prob)  # index 'len(keyframe_list)' indicate the current frame
    # Compute the cumulative distribution function (CDF).
    cdf = np.cumsum(norm_dis_prob)
    # Generate random samples.
    samples = np.random.rand(n_samples)
    # Select indices by comparing random numbers with CDF.
    sample_indices = np.searchsorted(cdf, samples)

    return [x if x != len(keyframe_list) else 'curr' for x in sample_indices]


def keyframe_selection_distance_loss(time_idx, curr_position, keyframe_list, keyframe_losses, distance_current_frame_prob,
                                     n_samples, loss_weight=0.7):
    """
    Performs sampling based on a probability distribution and returns
    the indices of `n_samples` selected keyframes.

    Args:
        -- time_idx: current timestep (frame_id)
        -- curr_position: current position (frame's pose translation)
        -- keyframe_list (list): a list containing all available keyframes
        -- keyframe_losses (list): a list containing keyframes last observed loss
        -- distance_current_frame_prob (float): probability of sampling current frame
        -- n_samples: The number of keyframes to be sampled.

    Returns:
        A list of `n_samples` indices of the selected keyframes.
    """

    loss_values = torch.tensor([keyframe_losses[keyframe['id']] for keyframe in keyframe_list])

    # min max alternative
    # if np.ptp(loss_values) == 0:
    #     norm_losses = np.ones_like(loss_values)
    # else:
    #     norm_losses = (loss_values - np.min(loss_values)) / (np.ptp(loss_values) + 1e-8)  # min-max scaling

    norm_losses = F.softmax(loss_values).detach().cpu().numpy()

    # Compute prob based on distance
    distances = []
    time_laps = []
    curr_position = curr_position.detach().cpu().numpy()
    curr_shift = np.linalg.norm(curr_position)
    for keyframe in keyframe_list:
        est_w2c = keyframe['w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3]
        distance = np.linalg.norm(camera_position - curr_position)
        time_lap = time_idx - keyframe['id']
        distances.append(distance)
        time_laps.append(time_lap)

    dis2prob = lambda x, scaler: np.log2(1 + scaler / (x + scaler / 5))
    dis_prob = [dis2prob(d, curr_shift) + dis2prob(t, time_idx) for d, t in zip(distances, time_laps)]

    # Incorporate loss into dis_prob
    combined_prob = [(1 - loss_weight) * p + loss_weight * l for p, l in zip(dis_prob, norm_losses)]

    # Normalize and sample
    sum_prob = sum(combined_prob) / (1 - distance_current_frame_prob)
    norm_dis_prob = [p / sum_prob for p in combined_prob]
    norm_dis_prob.append(distance_current_frame_prob)  # current frame
    cdf = np.cumsum(norm_dis_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)


    return [x if x != len(keyframe_list) else 'curr' for x in sample_indices]


def keyframe_selection_loss(keyframe_list, keyframe_losses, current_frame_prob, n_samples, tau=0.1):
    """
    Performs sampling prioritizing higher loss values returning the indices of `n_samples` selected keyframes.

    Args:
        -- keyframe_list (list): a list containing all available keyframes
        -- keyframe_losses (list): a list containing keyframes last observed loss
        -- current_frame_prob (float): probability of sampling current frame
        -- n_samples: The number of keyframes to be sampled.
        -- tau: temperature coefficient for smoothing the distribution

    Returns:
        A list of `n_samples` indices of the selected keyframes.
    """

    # keyframe losses actually stores all frame losses, so need to filter through keyframe list
    loss_values = torch.tensor([keyframe_losses[keyframe['id']] for keyframe in keyframe_list])

    losses_prob = F.softmax(loss_values / tau).cpu().numpy()

    # Normalize and sample
    sum_prob = sum(losses_prob) / (1 - current_frame_prob)
    norm_prob = [p / sum_prob for p in losses_prob]
    norm_prob.append(current_frame_prob)  # current frame
    cdf = np.cumsum(norm_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    return [x if x != len(keyframe_list) else 'curr' for x in sample_indices]


def keyframe_selection_loss_time(keyframe_list, keyframe_losses, current_frame_prob, n_samples, time_idx, tau=0.1, loss_weight=0.5):
    """
    Performs sampling prioritizing higher loss values returning the indices of `n_samples` selected keyframes.

    Args:
        -- keyframe_list (list): a list containing all available keyframes
        -- keyframe_losses (list): a list containing keyframes last observed loss
        -- current_frame_prob (float): probability of sampling current frame
        -- n_samples: The number of keyframes to be sampled.
        -- time_idx: current timestep (frame_id)
        -- tau: temperature coefficient for smoothing the distribution
        -- loss_weight: loss weight in final probability distribution

    Returns:
        A list of `n_samples` indices of the selected keyframes.
    """

    time_laps = []
    for keyframe in keyframe_list:
        time_lap = time_idx - keyframe['id']
        time_laps.append(time_lap)

    dis2prob = lambda x, scaler: np.log2(1 + scaler / (x + scaler / 5))
    dis_prob = [dis2prob(t, time_idx) for t in time_laps]

    # keyframe losses actually stores all frame losses, so need to filter through keyframe list
    loss_values = torch.tensor([keyframe_losses[keyframe['id']] for keyframe in keyframe_list])
    losses_prob = F.softmax(loss_values / tau).cpu().numpy()


    # Incorporate loss into dis_prob
    combined_prob = [(1 - loss_weight) * p + loss_weight * l for p, l in zip(dis_prob, losses_prob)]

    # Normalize and sample
    sum_prob = sum(combined_prob) / (1 - current_frame_prob)
    norm_prob = [p / sum_prob for p in combined_prob]
    norm_prob.append(current_frame_prob)  # current frame
    cdf = np.cumsum(norm_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    return [x if x != len(keyframe_list) else 'curr' for x in sample_indices]

def keyframe_selection_uniform_count(keyframe_list, keyframe_count, current_frame_prob, n_samples, tau=1, return_prob=False):

    counts = np.array([keyframe_count[keyframe['id']] for keyframe in keyframe_list])

    epsilon = 1e-6  # small number to avoid division by zero
    inv_counts = 1 / (counts + epsilon) ** tau
    prob = inv_counts / np.sum(inv_counts)

    sum_prob = np.sum(prob) / (1 - current_frame_prob)
    norm_dis_prob = [p / sum_prob for p in prob]
    norm_dis_prob.append(current_frame_prob)  # append the current frame probability

    # Step 4: Sample
    cdf = np.cumsum(norm_dis_prob)
    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    if return_prob:
        return [x if x != len(keyframe_list) else 'curr' for x in sample_indices], norm_dis_prob
    else:
        return [x if x != len(keyframe_list) else 'curr' for x in sample_indices]

def keyframe_selection_distance_spread(curr_position, keyframe_list, distance_current_frame_prob, n_samples, tau=1, return_prob=False):
    """
    Probabilistically samples keyframes to favor spatial diversity.

    Args:
        -- time_idx: current timestep (frame_id)
        -- curr_position: current position (frame's pose translation)
        -- keyframe_list (list): list of dicts, each with 'id' and 'w2c' keys
        -- distance_current_frame_prob (float): probability of sampling the current frame
        -- n_samples: number of keyframes to sample

    Returns:
        A list of n_samples indices into keyframe_list, or 'curr' for current frame.
    """

    def softmax(x, temp=1.0):
        x = np.array(x) / temp
        e_x = np.exp(x - np.max(x))  # for numerical stability
        return e_x / e_x.sum()

    curr_position = curr_position.detach().cpu().numpy()

    distances = []
    for i, keyframe in enumerate(keyframe_list):
        est_w2c = keyframe['w2c'].detach().cpu()
        camera_position = est_w2c[:3, 3].numpy()
        distance = np.linalg.norm(camera_position - curr_position)
        distances.append(distance)

    distances = np.array(distances)
    # # max_distance = np.max(distances) + 1e-6  # Prevent division by zero
    #
    # # Compute probability favoring more spaced keyframes
    # # Function rises with distance but caps out (log-saturation)
    # dis_prob = np.log1p(distances)  # log(1 + d)
    # dis_prob /= dis_prob.sum()  # Normalize (before adding current frame prob)



    dis_prob = softmax(distances, tau)

    # Rescale to account for distance_current_frame_prob
    dis_prob *= (1 - distance_current_frame_prob)

    # Add current frame as a sampling option
    dis_prob = np.append(dis_prob, distance_current_frame_prob)

    # Compute CDF for sampling
    cdf = np.cumsum(dis_prob)

    samples = np.random.rand(n_samples)
    sample_indices = np.searchsorted(cdf, samples)

    if return_prob:
        return [x if x != len(keyframe_list) else 'curr' for x in sample_indices], dis_prob
    else:
        return [x if x != len(keyframe_list) else 'curr' for x in sample_indices]