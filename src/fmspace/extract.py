"""提取特征；npz 只含 image_ids 和 features（模型原始输出）。"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


def extract(encode, paths, batch_size=32):
    features = []
    for i in tqdm(range(0, len(paths), batch_size)):
        images = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        with torch.inference_mode():
            features.append(encode(images).cpu().numpy())
    return np.concatenate(features)


def save_features(path, image_ids, features):
    np.savez(path, image_ids=np.asarray(image_ids, dtype=str), features=features)


def load_features(path, image_ids):
    """按 image_ids 的顺序返回特征。"""
    data = np.load(path)
    return pd.DataFrame(data["features"], index=data["image_ids"]).loc[image_ids].to_numpy()
