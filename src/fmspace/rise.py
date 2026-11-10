"""RISE（Petsiuk et al., BMVC 2018）：随机遮挡图像，由分数的变化估计各像素对分数的贡献。

score(images: list[PIL.Image]) -> ndarray[B, K]，K 个分数共用同一批前向。
分数有正负，显著图取 E[分数 | 像素保留] − E[分数 | 像素遮挡]。
"""

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm.auto import tqdm


def load_image(path, size=512):
    """读图并把长边缩到 size（保持纵横比）；掩膜加在这个尺寸上，之后交给各模型自己的预处理。"""
    image = Image.open(path).convert("RGB")
    scale = size / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def fov_mask(image, threshold=10):
    """眼底的视野圆（True = 圆内）。"""
    return np.asarray(image).max(axis=2) > threshold


def blurred(image, radius=40):
    """遮挡处的填充：模糊后的原图，保留亮度和颜色，去掉病灶等细节。"""
    return image.filter(ImageFilter.GaussianBlur(radius))


def random_masks(n, grid, shape, p, rng):
    """n 张掩膜 [n, H, W]，取值 0–1（1 = 保留）：grid × grid 的随机 0/1 网格双线性放大后随机平移。"""
    h, w = shape
    ch, cw = -(-h // grid), -(-w // grid)  # 每格的像素数
    cells = torch.from_numpy((rng.random((n, 1, grid, grid)) < p).astype(np.float32))
    big = torch.nn.functional.interpolate(cells, size=((grid + 1) * ch, (grid + 1) * cw), mode="bilinear", align_corners=False)
    dy, dx = rng.integers(0, ch, n), rng.integers(0, cw, n)
    return np.stack([big[i, 0, y : y + h, x : x + w].numpy() for i, (y, x) in enumerate(zip(dy, dx))])


def rise(score, image, fill, n_masks=4000, grid=8, p=0.5, batch_size=32, seed=0):
    """返回显著图 [K, H, W]。image、fill 是同尺寸的 PIL 图像；fill 是遮挡处显示的内容（blurred(image) 或全黑）。"""
    rng = np.random.default_rng(seed)
    x, f = np.asarray(image, dtype=np.float32), np.asarray(fill, dtype=np.float32)
    kept = masked = sum_kept = sum_masked = 0  # Σ s·m、Σ s·(1−m)、Σ m、Σ (1−m)
    for i in tqdm(range(0, n_masks, batch_size)):
        m = random_masks(min(batch_size, n_masks - i), grid, x.shape[:2], p, rng)
        mixed = x * m[..., None] + f * (1 - m[..., None])
        s = score([Image.fromarray(a) for a in mixed.round().astype(np.uint8)])
        kept = kept + np.einsum("bk,bhw->khw", s, m)
        masked = masked + np.einsum("bk,bhw->khw", s, 1 - m)
        sum_kept, sum_masked = sum_kept + m.sum(axis=0), sum_masked + (1 - m).sum(axis=0)
    return (kept / sum_kept - masked / sum_masked).astype(np.float32)
