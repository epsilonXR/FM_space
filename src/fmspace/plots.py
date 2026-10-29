"""画图。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from PIL import Image

MODEL_NAMES = {"siglip": "SigLIP", "medsiglip": "MedSigLIP", "conceptclip": "ConceptCLIP", "flair": "FLAIR"}
LABEL_NAMES = {
    "eye_dr_level": "DR severity",
    "artifact": "Artifact",
    "clarity": "Clarity",
    "field_definition": "Field definition",
    "overall_quality": "Overall quality",
    "eye": "Eye laterality",
}
METRIC_NAMES = {"spearman": r"Spearman $|\rho|$", "auroc": r"$2|\mathrm{AUC} - 0.5|$"}

INK, INK_SECONDARY, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#a3a29d", "#ffffff"
# 单色由浅到深：0 接近背景色，强关联最深；两种指标各用一个色相
CMAPS = {
    "spearman": LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "auroc": LinearSegmentedColormap.from_list("orange", [SURFACE, "#fbdccc", "#f4a27f", "#eb6834", "#b8461b", "#7a2c0e"]),
}


def _text_color(rgb):
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.3 else INK


def heatmap(assoc, model, labels, n_pcs=20):
    """PC × 标签的关联强度，有序标签（|ρ|）和二值标签（2|AUC − 0.5|）分成上下两块，各自一个色标。"""
    d = assoc[(assoc["model"] == model) & (assoc["component"] <= n_pcs)]
    d = d.assign(strength=d["value"].abs().where(d["metric"] == "spearman", (2 * (d["value"] - 0.5)).abs()))
    table = d.pivot(index="label", columns="component", values="strength")
    metric = d.drop_duplicates("label").set_index("label")["metric"]
    blocks = [(m, [l for l in labels if metric[l] == m]) for m in ("spearman", "auroc")]
    blocks = [(m, b) for m, b in blocks if b]

    fig, axes = plt.subplots(
        len(blocks), 2, squeeze=False, facecolor=SURFACE,
        figsize=(0.5 * n_pcs + 2.5, 0.42 * len(labels) + 1.2),
        gridspec_kw={"height_ratios": [len(b) for _, b in blocks], "width_ratios": [n_pcs, 0.3], "hspace": 0.18, "wspace": 0.03},
    )
    for (m, block), (ax, cax) in zip(blocks, axes):
        values = table.loc[block].to_numpy()
        mesh = ax.pcolormesh(values, cmap=CMAPS[m], vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
        for (i, j), v in np.ndenumerate(values):
            text = f"{v:.2f}"
            color = INK_MUTED if float(text) < 0.1 else _text_color(to_rgb(CMAPS[m](v)))
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=11, color=color)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(len(block)) + 0.5, [LABEL_NAMES[l] for l in block], color=INK)
        for tick, label in zip(ax.get_yticklabels(), block):
            tick.set_fontweight("bold" if label == "eye_dr_level" else "normal")
        ax.set_xticks(np.arange(n_pcs) + 0.5, range(1, n_pcs + 1), color=INK_SECONDARY)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)

        cbar = fig.colorbar(mesh, cax=cax, ticks=[0, 0.5, 1] if len(block) > 1 else [0, 1], format="%.1f")
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, colors=INK_SECONDARY)
        cbar.set_label(METRIC_NAMES[m], rotation=0, ha="left", va="center", color=INK)

    for ax in axes[:-1, 0]:
        ax.set_xticklabels([])
    axes[-1, 0].set_xlabel("Principal component", color=INK_SECONDARY)
    axes[0, 0].set_title(MODEL_NAMES[model], loc="left", color=INK, fontweight="bold")
    return fig


# 模型的颜色固定，所有图通用（分类色板前 4 个，按固定顺序）
MODEL_COLORS = {"siglip": "#2a78d6", "medsiglip": "#eb6834", "conceptclip": "#1baf7a", "flair": "#eda100"}
GRID, AXIS = "#e6e5e0", "#c9c8c3"


def _style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def variance_curve(variance, models, n_pcs=50, mark=20):
    """各模型前 n_pcs 个 PC 的累计解释方差（%）。在 PC mark 处画竖线，图例里给出各模型到这里的累计值。"""
    fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor=SURFACE)
    ax.axvline(mark, color=AXIS, linestyle="--", linewidth=1, zorder=1)
    for m in models:
        d = variance[(variance["model"] == m) & (variance["component"] <= n_pcs)]
        y = 100 * d["ratio"].cumsum().to_numpy()
        ax.plot(d["component"], y, color=MODEL_COLORS[m], linewidth=2, label=f"{MODEL_NAMES[m]}  {y[mark - 1]:.1f}%")
        ax.plot(mark, y[mark - 1], "o", color=MODEL_COLORS[m], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

    ax.set_xlim(1, n_pcs)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, *range(10, n_pcs + 1, 10)])
    ax.set_xlabel("Principal component", color=INK_SECONDARY)
    ax.set_ylabel("Cumulative explained variance (%)", color=INK_SECONDARY)
    _style(ax)
    legend = ax.legend(frameon=False, loc="lower right", labelcolor=INK, title=f"Cumulative at PC {mark}")
    legend.get_title().set_color(INK_SECONDARY)
    return fig


# DR 等级：OKLCH 色相每隔 72° 取一个（从 270° 起），亮度相同（L = 0.55），各色相取该亮度下的最大饱和度
GRADE_COLORS = ["#4963de", "#088285", "#5d7f00", "#ac5705", "#b63590"]


def grade_boxplot(scores, grades, models, ylabel="PC1 score (z-score)", width=0.15):
    """每个模型一组、组内各 DR 等级一个箱线图；4 个模型共用纵轴。
    箱为 25–75% 分位，中线为中位数，须延伸到 1.5 倍四分位距以内；0 处虚线为数据集均值。"""
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor=SURFACE)
    ax.axhline(0, color=AXIS, linewidth=0.8, linestyle="--", zorder=0)
    offsets = (np.arange(len(GRADE_COLORS)) - (len(GRADE_COLORS) - 1) / 2) * width * 1.25
    for k, m in enumerate(models):
        for g, color in enumerate(GRADE_COLORS):
            ax.boxplot(
                scores[m][grades == g], positions=[k + offsets[g]], widths=width, patch_artist=True,
                boxprops={"facecolor": to_rgba(color, 0.18), "edgecolor": color, "linewidth": 0.8},
                medianprops={"color": color, "linewidth": 1.6},
                whiskerprops={"color": color, "linewidth": 0.8},
                capprops={"color": color, "linewidth": 0.8},
                flierprops={"marker": "o", "markersize": 2, "markerfacecolor": to_rgba(color, 0.5), "markeredgecolor": "none"},
            )
    for g, color in enumerate(GRADE_COLORS):
        ax.plot([], [], color=to_rgba(color, 0.5), linewidth=6, label=f"DR {g}")

    ax.set_xticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    _style(ax)
    ax.tick_params(axis="x", colors=INK)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), labelcolor=INK)
    return fig



def pc_quantiles(scores, metadata, models, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95), window=0.05, mask=None, size=256):
    """每个模型一行，对应全部图像得分的各分位点。图下的数字是该分位点前后 window（百分位）内全部图像中占比最高的 DR 等级；
    图像从同一范围内、属于该等级、且 mask 选出的图像里，取得分最接近分位点的一张。"""
    grades = metadata["eye_dr_level"].to_numpy()
    mask = np.ones(len(metadata), bool) if mask is None else mask
    fig, axes = plt.subplots(len(models), len(quantiles), squeeze=False, facecolor=SURFACE,
                             figsize=(1.2 * len(quantiles) + 1.2, 1.7 * len(models)), gridspec_kw={"hspace": 0.22, "wspace": 0})
    for row, m in zip(axes, models):
        s = scores[m]
        rank = (np.argsort(np.argsort(s)) + 0.5) / len(s)  # 百分位
        for ax, q, v in zip(row, quantiles, np.quantile(s, quantiles)):
            near = np.abs(rank - q) <= window
            g = int(np.bincount(grades[near], minlength=len(GRADE_COLORS)).argmax())  # 占比最高的等级
            candidates = np.flatnonzero(near & mask & (grades == g))
            i = candidates[np.argmin(np.abs(s[candidates] - v))]
            image = Image.open(metadata["path"].iloc[i])
            w, h = image.size
            c = min(w, h)  # 裁成正方形，图像才能无缝拼在一起
            image = image.crop(((w - c) // 2, (h - c) // 2, (w + c) // 2, (h + c) // 2))
            image.thumbnail((size, size))
            ax.imshow(image)
            ax.axis("off")
            ax.text(0.5, -0.04, str(g), transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="bold", color=GRADE_COLORS[g])
        row[0].text(-0.12, 0.5, MODEL_NAMES[m], transform=row[0].transAxes, ha="right", va="center", color=INK, fontweight="bold")
    for ax, q in zip(axes[0], quantiles):
        ax.set_title(f"{q:.0%}", fontsize=10, color=INK_SECONDARY, pad=4)
    fig.text(0.5 * (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1), axes[0, 0].get_position().y1 + 0.05,
             "Percentile of PC1 score", ha="center", va="bottom", color=INK_SECONDARY)
    return fig
