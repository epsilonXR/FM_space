"""特征空间分析：PCA、PC × 标签关联、rank axis。"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from .data import LABELS


def l2(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def rho(a, b):
    return spearmanr(a, b)[0]


def zscore(score, y=None):
    """标准化为均值 0、标准差 1；给出 y 时把方向调成与 y 正相关（PC 的正负号是任意的）。"""
    z = (score - score.mean()) / score.std()
    return -z if y is not None and rho(z, y) < 0 else z


def pca(X, n_components=50):
    return PCA(n_components=n_components, svd_solver="full").fit(X)


def mean_cosine(X):
    """L2 归一化后的特征里，任意两张不同图像之间余弦相似度的平均（越接近 1 越集中）。"""
    n = len(X)
    return (np.sum(X.sum(axis=0) ** 2) - n) / (n * (n - 1))


def association(scores, metadata, labels=LABELS):
    """每个 PC 与每个标签的关联：ordinal 用 Spearman，binary 用 AUROC（正类为排序后的第二个值）。"""
    rows = []
    for label, kind in labels.items():
        values = metadata[label].to_numpy()
        if kind == "binary":
            values = values == np.unique(values)[1]
        for k in range(scores.shape[1]):
            value = rho(scores[:, k], values) if kind == "ordinal" else roc_auc_score(values, scores[:, k])
            rows.append({"label": label, "component": k + 1, "metric": "spearman" if kind == "ordinal" else "auroc", "value": value})
    return pd.DataFrame(rows)


def _train_linear(X_train, y_train, X_val, y_val, lr, weight_decay, seed, epochs=100, batch_size=128):
    """MSE + Adam + 余弦退火；返回验证集 Spearman 最高的 epoch 的 (val_rho, weight, bias)。"""
    torch.manual_seed(seed)
    X_train, y_train, X_val = (torch.as_tensor(a, dtype=torch.float32) for a in (X_train, y_train, X_val))
    linear = torch.nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * int(np.ceil(len(X_train) / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0)

    best = (-np.inf, None, None)
    for _ in range(epochs):
        for batch in torch.randperm(len(X_train)).split(batch_size):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(linear(X_train[batch]).squeeze(1), y_train[batch]).backward()
            optimizer.step()
            scheduler.step()
        with torch.no_grad():
            val_rho = rho(linear(X_val).squeeze(1).numpy(), y_val)
        if val_rho > best[0]:
            best = (val_rho, linear.weight.detach().numpy().ravel().copy(), linear.bias.item())
    return best


def rank_axis(X, y, split, n_configs=30, seed=0):
    """Sonthalia et al. (NeurIPS 2025) 的流程：lr ~ logU[1e-6, 1e-1]、weight decay ~ logU[1e-7, 1e-4]
    随机搜索 n_configs 组，每组训练 100 个 epoch，按验证集 Spearman 选 epoch 和配置。不做翻转增强。"""
    rng = np.random.default_rng(seed)
    configs = [(10 ** rng.uniform(-6, -1), 10 ** rng.uniform(-7, -4)) for _ in range(n_configs)]
    train, val = (split == "train").to_numpy(), (split == "val").to_numpy()

    results = [
        (*_train_linear(X[train], y[train], X[val], y[val], lr, wd, seed + k), lr, wd)
        for k, (lr, wd) in enumerate(configs)
    ]
    val_rho, weight, bias, lr, wd = max(results, key=lambda r: r[0])
    return {"weight": weight, "bias": bias, "lr": lr, "weight_decay": wd, "val_rho": val_rho}


def patient_bootstrap(groups, statistic, n_boot=1000, seed=0):
    """按患者有放回抽样，返回 statistic(index) 的 95% 区间。"""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    patients = np.unique(groups)
    members = {p: np.flatnonzero(groups == p) for p in patients}
    values = [
        statistic(np.concatenate([members[p] for p in rng.choice(patients, len(patients))]))
        for _ in range(n_boot)
    ]
    return np.percentile(values, [2.5, 97.5])


def compare_with_pc1(rank_score, pc1, y, groups):
    """同一批图像上 rank axis 分数的 ρ、PC1 的 |ρ| 及两者之差（附患者 bootstrap 95% 区间）。"""
    diff = lambda i: rho(rank_score[i], y[i]) - abs(rho(pc1[i], y[i]))
    low, high = patient_bootstrap(groups, diff)
    return {
        "rank_rho": rho(rank_score, y),
        "pc1_rho": abs(rho(pc1, y)),
        "diff": diff(np.arange(len(y))),
        "diff_low": low,
        "diff_high": high,
    }
