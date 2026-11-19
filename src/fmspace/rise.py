"""RISE (Petsiuk et al., BMVC 2018), following the authors' implementation
(https://github.com/eclique/RISE, explanations.py);
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
from tqdm.auto import tqdm


def generate_masks(N, s, p1, input_size, rng):
    """Section 3.2 of the paper. Returns masks [N, 1, H, W] with values in [0, 1]; 1 = keep."""
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = (s + 1) * cell_size

    grid = rng.random((N, s, s)) < p1
    grid = grid.astype("float32")

    masks = np.empty((N, *input_size), dtype="float32")

    for i in range(N):
        # Random shifts
        x = rng.integers(0, cell_size[0])
        y = rng.integers(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode="reflect", anti_aliasing=False)[
            x : x + input_size[0], y : y + input_size[1]
        ]
    masks = masks.reshape(-1, 1, *input_size)
    return torch.from_numpy(masks)


@torch.inference_mode()
def rise(model, x, N=4000, s=7, p1=0.5, gpu_batch=32, seed=0):
    """Saliency maps [CL, H, W].

    model: tensor [n, C, H, W] -> scores [n, CL] (the black box f).
    x: preprocessed (normalized) image [1, C, H, W]. Masking the normalized image, as the authors do,
       replaces masked regions with 0, i.e. the mean color of the normalization.
    """
    _, _, H, W = x.size()
    masks = generate_masks(N, s, p1, (H, W), np.random.default_rng(seed))

    p = []
    for i in tqdm(range(0, N, gpu_batch)):
        # Apply array of filters to the image
        stack = torch.mul(masks[i : i + gpu_batch].to(x.device), x)
        p.append(model(stack).float().cpu())
    p = torch.cat(p)
    p = p - p.mean(dim=0)  # centering: the only difference from the paper

    # Number of scores
    CL = p.size(1)
    sal = torch.matmul(p.transpose(0, 1), masks.view(N, H * W))
    sal = sal.view((CL, H, W))
    sal = sal / N / p1
    return sal


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of [n, C, H, W] with reflect padding; sigma in pixels of x."""
    r = int(3 * sigma)
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) ** 2) / (2 * sigma**2))
    k = (k / k.sum()).repeat(x.size(1), 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k, groups=x.size(1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3), groups=x.size(1))


@torch.inference_mode()
def rise_shapley(model, x, baseline, N=16000, s=14, gpu_batch=32, seed=0):
    """Saliency maps [CL, H, W] that split model(x) - model(baseline) over regions (Shapley values).

    Differences from rise():
    - masked regions take the baseline image instead of 0;
    - each mask has its own keep probability q ~ U(0, 1) (Owen's formula: Shapley = Banzhaf averaged over q);
    - masks are generated per batch and summed on the fly, so memory does not grow with N.
    sal.sum() / cell_area ≈ model(x) - model(baseline).
    """
    _, _, H, W = x.size()
    rng = np.random.default_rng(seed)
    q = ((rng.permutation(N) + rng.random(N)) / N).astype("float32")  # stratified U(0, 1)
    s0, s_ref = model(x).float().cpu(), model(baseline).float().cpu()

    sal = 0
    for i in tqdm(range(0, N, gpu_batch)):
        qb = q[i : i + gpu_batch]
        masks = generate_masks(len(qb), s, qb[:, None, None], (H, W), rng)
        m = masks.to(x.device)
        p = model(m * x + (1 - m) * baseline).float().cpu()
        qb = torch.from_numpy(qb)[:, None]
        p = p - ((1 - qb) * s_ref + qb * s0)  # centering on the straight line from baseline (q=0) to image (q=1)
        sal = sal + p.T @ ((masks.view(len(qb), H * W) - qb) / (qb * (1 - qb)))

    CL = p.size(1)
    return (sal / N).view((CL, H, W))


def field_of_view(image, size, threshold=10):
    """Field-of-view mask [1, h, w] of a fundus PIL image (gray > threshold), resized to size = (h, w); 1 = inside."""
    gray = np.asarray(image.convert("L").resize(size[::-1]))
    return torch.from_numpy(gray > threshold).float()[None]


def fov_blur(x, fov, sigma):
    """Gaussian blur inside the field of view only (normalized convolution), so the black border does not
    leak in; outside the field of view the image is kept. x: [n, C, H, W]; fov: [n, 1, H, W]."""
    blurred = gaussian_blur(x * fov, sigma) / gaussian_blur(fov, sigma)
    return torch.where(fov.bool(), blurred, x)
