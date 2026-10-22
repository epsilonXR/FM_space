"""提取特征；特征文件为 npz，只含 image_ids 和 features。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


def extract(encode, paths, batch_size=32):
    """按 paths 的顺序提取，返回 [N, D] float32（模型原始输出）。"""
    paths = list(paths)
    features = []
    for start in tqdm(range(0, len(paths), batch_size)):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        with torch.inference_mode():
            features.append(encode(images).float().cpu().numpy())
    return np.concatenate(features)


def save_features(path, image_ids, features):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, image_ids=np.asarray(image_ids, dtype=str), features=features)


def load_features(path, image_ids):
    """按 image_ids 的顺序返回特征；文件里缺少任何一张就报错。"""
    data = np.load(path)
    position = pd.Index(data["image_ids"]).get_indexer(np.asarray(image_ids, dtype=str))
    if (position < 0).any():
        raise KeyError(f"{path} 缺少 {(position < 0).sum()} 张图的特征")
    return data["features"][position]
