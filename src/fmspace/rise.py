"""RISE（Petsiuk et al., BMVC 2018）。score(images: list[PIL.Image]) -> ndarray[B, K]。"""

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm.auto import tqdm


def load_image(path, size=512):
    image = Image.open(path).convert("RGB")
    scale = size / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def fov_mask(image, threshold=10):
    return np.asarray(image).max(axis=2) > threshold


def blurred(image, radius=40):
    return image.filter(ImageFilter.GaussianBlur(radius))


def random_masks(n, grid, shape, p, rng):
    """[n, H, W]，取值 0–1，1 = 保留。"""
    h, w = shape
    ch, cw = -(-h // grid), -(-w // grid)
    cells = torch.from_numpy((rng.random((n, 1, grid, grid)) < p).astype(np.float32))
    big = torch.nn.functional.interpolate(cells, size=((grid + 1) * ch, (grid + 1) * cw), mode="bilinear", align_corners=False)
    dy, dx = rng.integers(0, ch, n), rng.integers(0, cw, n)
    return np.stack([big[i, 0, y : y + h, x : x + w].numpy() for i, (y, x) in enumerate(zip(dy, dx))])


def mask_batches(n_masks, grid, shape, p, batch_size, seed):
    """按批生成掩膜；参数相同则每次生成的掩膜相同。"""
    rng = np.random.default_rng(seed)
    for start in range(0, n_masks, batch_size):
        yield random_masks(min(batch_size, n_masks - start), grid, shape, p, rng)


def occlude(image, fill, masks):
    """image × 掩膜 + fill × (1 − 掩膜) → 遮挡图的列表。"""
    x, f, m = np.asarray(image, dtype=np.float32), np.asarray(fill, dtype=np.float32), masks[..., None]
    return [Image.fromarray(a) for a in (x * m + f * (1 - m)).round().astype(np.uint8)]


def occluded_scores(score, image, fill, batches):
    """各批遮挡图的得分 [B, K]。只有这一步过模型。"""
    return [score(occlude(image, fill, masks)) for masks in batches]


def saliency(scores, batches):
    """显著图 [K, H, W] = E[分数 | 像素保留] − E[分数 | 像素遮挡]。"""
    kept = hidden = weight_kept = weight_hidden = 0
    for s, masks in zip(scores, batches):
        kept = kept + np.einsum("bk,bhw->khw", s, masks)
        hidden = hidden + np.einsum("bk,bhw->khw", s, 1 - masks)
        weight_kept = weight_kept + masks.sum(axis=0)
        weight_hidden = weight_hidden + (1 - masks).sum(axis=0)
    return (kept / weight_kept - hidden / weight_hidden).astype(np.float32)


def rise(score, image, fill, n_masks=4000, grid=8, p=0.5, batch_size=32, seed=0):
    """掩膜不保存：打分和算显著图时各生成一遍（同一个 seed）。"""
    batches = lambda: mask_batches(n_masks, grid, image.size[::-1], p, batch_size, seed)
    scores = occluded_scores(score, image, fill, tqdm(batches(), total=-(-n_masks // batch_size)))
    return saliency(scores, batches())
