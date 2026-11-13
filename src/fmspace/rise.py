"""RISE (Petsiuk et al., BMVC 2018): estimate each pixel's contribution to a score by randomly masking the image.

The only difference from the paper: scores are centered (mean subtracted) before the weighted sum,
so that 0 in the saliency map means "no effect" when scores can be negative.
"""

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


def generate_masks(n, grid, height, width, p, rng):
    """Section 3.2 of the paper. Returns n masks [n, height, width] with values in [0, 1]; 1 = keep."""
    cell_h = int(np.ceil(height / grid))
    cell_w = int(np.ceil(width / grid))

    # 1. Binary grid of size grid x grid; each cell is 1 with probability p
    small = (rng.random((n, 1, grid, grid)) < p).astype(np.float32)

    # 2. Bilinear upsampling to one cell larger than the image
    large = torch.nn.functional.interpolate(
        torch.from_numpy(small), size=((grid + 1) * cell_h, (grid + 1) * cell_w), mode="bilinear", align_corners=False
    ).numpy()[:, 0]

    # 3. Crop the image size at a random offset
    masks = np.empty((n, height, width), dtype=np.float32)
    for i in range(n):
        top = rng.integers(0, cell_h)
        left = rng.integers(0, cell_w)
        masks[i] = large[i, top : top + height, left : left + width]
    return masks


def apply_masks(image, masks):
    """I ⊙ M: masked regions become black. image [H, W, 3], masks [n, H, W]; returns n PIL images."""
    masked = image[None] * masks[..., None]
    return [Image.fromarray(a) for a in masked.round().astype(np.uint8)]


def rise(score, image, n_masks=4000, grid=7, p=0.5, batch_size=32, seed=0):
    """Saliency map [K, H, W].

    score: list[PIL.Image] -> ndarray [B, K], K scores per image.
    saliency = Σ (s_i − mean score) · M_i / (N · p), i.e. Eq. (6) of the paper with centered scores.
    """
    rng = np.random.default_rng(seed)
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape[:2]

    sum_score_mask = 0  # Σ s_i · M_i, [K, H, W]
    sum_score = 0  # Σ s_i, [K]
    sum_mask = 0  # Σ M_i, [H, W]

    for start in tqdm(range(0, n_masks, batch_size)):
        n = min(batch_size, n_masks - start)
        masks = generate_masks(n, grid, height, width, p, rng)  # M_i
        scores = score(apply_masks(image, masks))  # s_i = f(I ⊙ M_i)

        sum_score_mask = sum_score_mask + np.einsum("nk,nhw->khw", scores, masks)
        sum_score = sum_score + scores.sum(axis=0)
        sum_mask = sum_mask + masks.sum(axis=0)

    mean_score = sum_score / n_masks
    # Σ (s_i − mean) · M_i = Σ s_i · M_i − mean · Σ M_i
    centered = sum_score_mask - mean_score[:, None, None] * sum_mask
    return centered / (n_masks * p)
