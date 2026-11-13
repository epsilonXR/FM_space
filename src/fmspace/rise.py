"""RISE（Petsiuk et al., BMVC 2018）：随机遮挡图像，由分数的变化估计每个像素对分数的贡献。

与论文的唯一区别：分数先减去平均值再加权求和。分数有正有负时，这样显著图的 0 才表示"没有影响"。
"""

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


def generate_masks(n, grid, height, width, p, rng):
    """论文 3.2 节。返回 n 张掩膜 [n, height, width]，取值 0–1，1 = 保留。"""
    cell_h = int(np.ceil(height / grid))
    cell_w = int(np.ceil(width / grid))

    # 1. grid × grid 的二值网格，每格以概率 p 取 1
    small = (rng.random((n, 1, grid, grid)) < p).astype(np.float32)

    # 2. 双线性放大，比图像多出一格
    large = torch.nn.functional.interpolate(
        torch.from_numpy(small), size=((grid + 1) * cell_h, (grid + 1) * cell_w), mode="bilinear", align_corners=False
    ).numpy()[:, 0]

    # 3. 在随机位置裁出图像大小
    masks = np.empty((n, height, width), dtype=np.float32)
    for i in range(n):
        top = rng.integers(0, cell_h)
        left = rng.integers(0, cell_w)
        masks[i] = large[i, top : top + height, left : left + width]
    return masks


def apply_masks(image, masks):
    """I ⊙ M：遮挡处变黑。image [H, W, 3]，masks [n, H, W]；返回 n 张 PIL 图像。"""
    masked = image[None] * masks[..., None]
    return [Image.fromarray(a) for a in masked.round().astype(np.uint8)]


def rise(score, image, n_masks=4000, grid=7, p=0.5, batch_size=32, seed=0):
    """显著图 [K, H, W]。

    score：list[PIL.Image] → ndarray [B, K]，每张图 K 个分数。
    显著图 = Σ (s_i − 平均分数) · M_i ÷ (N · p)，即论文式 (6)，分数先减去平均值。
    """
    rng = np.random.default_rng(seed)
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape[:2]

    sum_score_mask = 0  # Σ s_i · M_i，[K, H, W]
    sum_score = 0  # Σ s_i，[K]
    sum_mask = 0  # Σ M_i，[H, W]

    for start in tqdm(range(0, n_masks, batch_size)):
        n = min(batch_size, n_masks - start)
        masks = generate_masks(n, grid, height, width, p, rng)  # M_i
        scores = score(apply_masks(image, masks))  # s_i = f(I ⊙ M_i)

        sum_score_mask = sum_score_mask + np.einsum("nk,nhw->khw", scores, masks)
        sum_score = sum_score + scores.sum(axis=0)
        sum_mask = sum_mask + masks.sum(axis=0)

    mean_score = sum_score / n_masks
    # Σ (s_i − 平均) · M_i = Σ s_i · M_i − 平均 · Σ M_i
    centered = sum_score_mask - mean_score[:, None, None] * sum_mask
    return centered / (n_masks * p)
