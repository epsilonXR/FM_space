"""DeepDRiD regular-fundus-training：metadata 与患者划分。

data_dir 的结构：
    regular-fundus-training.csv
    regular-fundus-training/Images/<patient_id>/<image_id>.jpg
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 分析用的标签：ordinal 用 Spearman，binary 用 AUROC
LABELS = {
    "eye_dr_level": "ordinal",
    "artifact": "ordinal",
    "clarity": "ordinal",
    "field_definition": "ordinal",
    "overall_quality": "binary",
    "eye": "binary",
}


def load_metadata(data_dir):
    """每行一张图，path 为图像的完整路径。

    眼别以 image_id 末尾的 l/r 为准，取该眼的 DR 等级；
    该眼没有标签的图（image_id 与标签冲突，共 11 张）排除。
    """
    data_dir = Path(data_dir)
    raw = pd.read_csv(data_dir / "regular-fundus-training.csv")

    eye = raw["image_id"].str.extract(r"_([lr])\d+$", expand=False).str.upper()
    assert eye.notna().all(), "image_id 无法解析眼别"
    level = raw["left_eye_DR_Level"].where(eye == "L", raw["right_eye_DR_Level"])

    images = data_dir / "regular-fundus-training" / "Images"
    relative = raw["image_path"].str.replace("\\", "/").str.lstrip("/")

    metadata = pd.DataFrame(
        {
            "patient_id": raw["patient_id"],
            "image_id": raw["image_id"],
            "path": [images / p for p in relative],
            "eye": eye,
            "eye_dr_level": level,
            "overall_quality": raw["Overall quality"],
            "clarity": raw["Clarity"],
            "field_definition": raw["Field definition"],
            "artifact": raw["Artifact"],
        }
    )
    return metadata[level.notna()].astype({"eye_dr_level": int}).reset_index(drop=True)


def patient_split(metadata, seed=0, val_size=0.15, test_size=0.15):
    """按患者划分 train / val / test，同一患者的图只在一个集合里。"""
    groups = metadata["patient_id"].to_numpy()
    index = np.arange(len(groups))

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val, test = next(outer.split(index, groups=groups))

    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=seed
    )
    _, val = next(inner.split(train_val, groups=groups[train_val]))

    split = np.full(len(groups), "train", dtype=object)
    split[train_val[val]] = "val"
    split[test] = "test"
    return pd.Series(split, index=metadata.index, name="split")
