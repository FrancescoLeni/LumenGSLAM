import torch
import numpy as np

from lightglue import LightGlue, SuperPoint

from LumenGSLAM.utils.data_process.geometry import backproject_selected_points



class MyMatcher(torch.nn.Module):
    def __init__(self, device, score_th=0.9):
        super().__init__()

        def rbd(data: dict) -> dict:
            """Remove batch dimension from elements in data"""
            return {
                k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
                for k, v in data.items()
            }

        self.extractor = SuperPoint(max_num_keypoints=1000).eval().to(device)  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().to(device)  # load the matcher

        self.score_th = score_th

        self.rbd_fn = rbd

    def forward(self, i0, i1):
        feats0 = self.extractor.extract(i0)  # auto-resize the image, disable with resize=None
        feats1 = self.extractor.extract(i1)

        matches01 = self.matcher({'image0': feats0, 'image1': feats1})

        feats0, feats1, matches01 = [self.rbd_fn(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]
        scores = matches01['scores']

        mask = scores >= self.score_th
        sorted_indices = torch.argsort(scores[mask], descending=True)
        points0_sorted = points0[mask][sorted_indices]
        points1_sorted = points1[mask][sorted_indices]
        scores_sorted = scores[mask][sorted_indices]

        results = {'matched_kpts0': points0, 'matched_kpts1': points1, 'inlier_kpts0': points0_sorted,
                   'inlier_kpts1': points1_sorted, 'matches_scores': scores, 'inlier_scores': scores_sorted}


        return results


def get_keypoints_3D_2D(matcher, i0, d0, p0, i1, K):
    result = matcher(i0, i1)

    inlier_kpts0, inlier_kpts1 = result['inlier_kpts0'], result['inlier_kpts1']

    if not isinstance(inlier_kpts0, torch.Tensor):
        inlier_kpts0 = torch.tensor(inlier_kpts0, dtype=torch.float32).to(i0.device)
        inlier_kpts1 = torch.tensor(inlier_kpts1, dtype=torch.float32).to(i0.device)

    match_0 = torch.round(inlier_kpts0)

    # matched points in i0 in world coord
    pts_0 = backproject_selected_points(match_0, d0, K, p0)

    return pts_0[:,:3].to(i0.device), inlier_kpts1.to(i0.device), result


def sample_sparse_matches_3d2d_fuzzy(points3D, points2D, num_samples=17, fuzziness=0.3):
    """
    Samples spatially sparse and slightly randomized 3D-2D matches for PnP.

    Args:
        points3D: (N, 3) torch.Tensor — 3D points.
        points2D: (N, 2) torch.Tensor — 2D keypoints corresponding to points3D.
        num_samples: int — number of matches to sample.
        fuzziness: float in [0, 1] — adds randomness to selection (higher = more random).

    Returns:
        sel_points3D: (num_samples, 3)
        sel_points2D: (num_samples, 2)
    """
    assert points3D.shape[0] == points2D.shape[0]
    N = points3D.shape[0]
    if N <= num_samples:
        return points3D, points2D

    selected_indices = []
    remaining_indices = torch.arange(N)

    # Start from 3D point closest to centroid
    centroid = points3D.mean(dim=0)
    dists_to_centroid = torch.norm(points3D - centroid, dim=1)
    first_idx = torch.argmin(dists_to_centroid)
    selected_indices.append(first_idx.item())

    selected_pts = points3D[first_idx].unsqueeze(0)

    for _ in range(num_samples - 1):
        # Compute distance of remaining points to the selected set
        dists = torch.cdist(points3D[remaining_indices], selected_pts)  # (N_remain, k)
        min_dists = dists.min(dim=1).values  # (N_remain,)

        # Add fuzziness (small noise to distances)
        #noise = fuzziness * torch.rand_like(min_dists)
        scores = min_dists #+ noise

        # Select the index with highest (noisy) distance
        idx_in_remain = torch.argmax(scores)
        sel_idx = remaining_indices[idx_in_remain]
        selected_indices.append(sel_idx.item())

        # Update
        selected_pts = torch.cat([selected_pts, points3D[sel_idx].unsqueeze(0)], dim=0)
        remaining_indices = remaining_indices[remaining_indices != sel_idx]

    sel_indices = torch.tensor(selected_indices, device=points3D.device)
    return points3D[sel_indices], points2D[sel_indices], sel_indices