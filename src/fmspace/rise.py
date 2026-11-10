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


def rise(score, image, fill, n_masks=4000, grid=8, p=0.5, batch_size=32, seed=0):
    """显著图 [K, H, W] = E[分数 | 像素保留] − E[分数 | 像素遮挡]；fill 是遮挡处显示的图像。"""
    rng = np.random.default_rng(seed)
    x, f = np.asarray(image, dtype=np.float32), np.asarray(fill, dtype=np.float32)
    kept = masked = sum_kept = sum_masked = 0
    for i in tqdm(range(0, n_masks, batch_size)):
        m = random_masks(min(batch_size, n_masks - i), grid, x.shape[:2], p, rng)
        mixed = x * m[..., None] + f * (1 - m[..., None])
        s = score([Image.fromarray(a) for a in mixed.round().astype(np.uint8)])
        kept = kept + np.einsum("bk,bhw->khw", s, m)
        masked = masked + np.einsum("bk,bhw->khw", s, 1 - m)
        sum_kept, sum_masked = sum_kept + m.sum(axis=0), sum_masked + (1 - m).sum(axis=0)
    return (kept / sum_kept - masked / sum_masked).astype(np.float32)
