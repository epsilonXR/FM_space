"""RISE (Petsiuk et al., BMVC 2018), following the authors' implementation
(https://github.com/eclique/RISE, explanations.py);
"""

import numpy as np
import torch
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
