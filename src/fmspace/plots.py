"""画图。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.stats import gaussian_kde

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


# DR 等级：由冷到暖的 5 种颜色（validate_palette --pairs all 通过；DR 1 与 DR 3 在红绿色弱下较接近，
# 所以这两条另用虚线 / 点划线区分）
GRADE_STYLES = [
    ("#2a78d6", "-"),
    ("#1baf7a", (0, (5, 2))),
    ("#eda100", "-"),
    ("#e87ba4", (0, (5, 2, 1, 2))),
    ("#a8322d", "-"),
]


def _tint(color, amount):
    """与背景色混合：amount = 0 为原色，1 为背景色。"""
    c, w = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return tuple(c + (w - c) * amount)


def grade_ridgeline(score, grades, model, xlim, xlabel="PC1 score (z-score)", bw=0.35, step=0.55):
    """各 DR 等级在一维分数上的 KDE，画成山脊图：每个等级一行、各自归一化，竖线为中位数。
    step 为行距（相对最高峰），小于 1 时相邻两行略有重叠。"""
    x = np.linspace(*xlim, 500)
    kdes = [gaussian_kde(score[grades == g], bw_method=bw) for g in range(len(GRADE_STYLES))]
    densities = [k(x) for k in kdes]
    h = step * max(d.max() for d in densities)

    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    for v in range(int(np.ceil(xlim[0])), int(np.floor(xlim[1])) + 1):
        ax.axvline(v, color=AXIS if v == 0 else GRID, linewidth=0.8, linestyle="--" if v == 0 else "-", zorder=0)

    for g in reversed(range(len(densities))):  # 下面的行画在前面，盖住上一行的底部
        color, _ = GRADE_STYLES[g]
        base, curve, z = g * h, g * h + densities[g], 10 * (len(densities) - g)
        ax.fill_between(x, base, curve, color=_tint(color, 0.55), linewidth=0, zorder=z)
        ax.plot(x, curve, color=SURFACE, linewidth=2.2, zorder=z + 1)
        ax.plot(x, curve, color=color, linewidth=1.3, zorder=z + 2)
        median = np.median(score[grades == g])
        ax.plot([median, median], [base, base + kdes[g](median)[0]], color=color, linewidth=1.3, zorder=z + 2)
        ax.text(-0.01, base + 0.1 * h, f"DR {g}", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                color=INK, fontweight="bold")
        ax.text(xlim[1], base + 0.1 * h, f"n = {(grades == g).sum()}", ha="right", va="bottom",
                color=INK_SECONDARY, fontsize=8, zorder=z + 3)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, (len(densities) - 1) * h + 1.05 * max(d.max() for d in densities))
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_title(MODEL_NAMES[model], loc="left", color=INK, fontweight="bold")
    return fig
