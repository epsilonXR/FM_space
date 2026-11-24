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


def _assoc_blocks(assoc, model, labels, n_pcs):
    """有序标签（|ρ|）和二值标签（2|AUC − 0.5|）各一块：[(指标, 标签, 数值 [标签, n_pcs])]。"""
    d = assoc[(assoc["model"] == model) & (assoc["component"] <= n_pcs)]
    d = d.assign(strength=d["value"].abs().where(d["metric"] == "spearman", (2 * (d["value"] - 0.5)).abs()))
    table = d.pivot(index="label", columns="component", values="strength")
    metric = d.drop_duplicates("label").set_index("label")["metric"]
    blocks = [(m, [l for l in labels if metric[l] == m]) for m in ("spearman", "auroc")]
    return [(m, b, table.loc[b].to_numpy()) for m, b in blocks if b]


def _draw_heatmap(ax, metric, block, values, fontsize=11):
    n_pcs = values.shape[1]
    mesh = ax.pcolormesh(values, cmap=CMAPS[metric], vmin=0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
    for (i, j), v in np.ndenumerate(values):
        text = f"{v:.2f}"
        color = INK_MUTED if float(text) < 0.1 else _text_color(to_rgb(CMAPS[metric](v)))
        ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=fontsize, color=color)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(block)) + 0.5, [LABEL_NAMES[l] for l in block], color=INK)
    for tick, label in zip(ax.get_yticklabels(), block):
        tick.set_fontweight("bold" if label == "eye_dr_level" else "normal")
    ax.set_xticks(np.arange(n_pcs) + 0.5, range(1, n_pcs + 1), color=INK_SECONDARY)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    return mesh


def _colorbar(fig, mesh, cax, metric, ticks, orientation="vertical"):
    cbar = fig.colorbar(mesh, cax=cax, ticks=ticks, format="%.1f", orientation=orientation)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, colors=INK_SECONDARY)
    if orientation == "vertical":
        cbar.set_label(METRIC_NAMES[metric], rotation=0, ha="left", va="center", color=INK)
    else:
        cbar.ax.set_title(METRIC_NAMES[metric], color=INK, fontsize="medium")


def heatmap(assoc, model, labels, n_pcs=20):
    """PC × 标签的关联强度，有序标签（|ρ|）和二值标签（2|AUC − 0.5|）分成上下两块，各自一个色标。"""
    blocks = _assoc_blocks(assoc, model, labels, n_pcs)
    fig, axes = plt.subplots(
        len(blocks), 2, squeeze=False, facecolor=SURFACE,
        figsize=(0.5 * n_pcs + 2.5, 0.42 * len(labels) + 1.2),
        gridspec_kw={"height_ratios": [len(b) for _, b, _ in blocks], "width_ratios": [n_pcs, 0.3], "hspace": 0.18, "wspace": 0.03},
    )
    for (m, block, values), (ax, cax) in zip(blocks, axes):
        mesh = _draw_heatmap(ax, m, block, values)
        _colorbar(fig, mesh, cax, m, [0, 0.5, 1] if len(block) > 1 else [0, 1])

    for ax in axes[:-1, 0]:
        ax.set_xticklabels([])
    axes[-1, 0].set_xlabel("Principal component", color=INK_SECONDARY)
    axes[0, 0].set_title(MODEL_NAMES[model], loc="left", color=INK, fontweight="bold")
    return fig


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


def _grade_handles():
    """DR 等级的图例（实心圆点），箱线图和散点图共用。"""
    return [plt.Line2D([], [], ls="", marker="o", ms=6, color=c, label=f"DR {g}") for g, c in enumerate(GRADE_COLORS)]


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
    ax.set_xticks(range(len(models)), [MODEL_NAMES[m] for m in models])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    _style(ax)
    ax.tick_params(axis="x", colors=INK)
    ax.legend(handles=_grade_handles(), frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), labelcolor=INK, handletextpad=0.3)
    return fig


def _scatter_handles():
    return _grade_handles() + [plt.Line2D([], [], ls="", marker="o", ms=6, color=INK_SECONDARY, label="Left eye"),
                               plt.Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec=INK_SECONDARY, label="Right eye")]


def _draw_scatter(ax, score, ratio, grades, right, size):
    """PC1 调成与 DR 正相关，PC2 调成右眼为正；颜色为 DR 等级，实心为左眼、空心为右眼；纵横同比例。"""
    z = score[:, :2] * [np.sign(np.corrcoef(score[:, 0], grades)[0, 1]), np.sign(np.corrcoef(score[:, 1], right)[0, 1])]
    order = np.random.default_rng(0).permutation(len(grades))  # 随机绘制顺序，避免某个等级总压在上面
    edge = np.array([to_rgba(GRADE_COLORS[g], 0.8) for g in grades])
    face = edge.copy()
    face[right, 3] = 0  # 右眼空心
    ax.scatter(*z[order].T, s=size, facecolors=face[order], edgecolors=edge[order], linewidths=0.6)
    ax.set_box_aspect(1)  # 图框为正方形
    ax.set_aspect("equal", adjustable="datalim")  # 纵横同比例：扩展坐标范围来适应正方形图框
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_color(AXIS)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})", color=INK_SECONDARY)
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})", color=INK_SECONDARY)


def pc_scatter(scores, ratios, grades, eyes, models, ncols=2, size=9):
    """各模型第 1、2 主成分得分的散点图（纵横同比例），每行 ncols 个。颜色为 DR 等级，实心为左眼、空心为右眼。
    主成分的正负号是任意的：PC1 调成与 DR 正相关，PC2 调成右眼为正。"""
    right = eyes == "R"
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(3.0 * ncols + 1.2, 3.3 * nrows), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.12, "hspace": 0.3})
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax, m in zip(axes.flat, models):
        _draw_scatter(ax, scores[m], ratios[m], grades, right, size)
        ax.set_title(MODEL_NAMES[m], loc="left", color=INK, fontweight="bold")
    axes[0, -1].legend(handles=_scatter_handles(), frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), labelcolor=INK, handletextpad=0.3)
    return fig


def pc_overview(assoc, scores, ratios, grades, eyes, models, labels, n_pcs=10, size=4):
    """每个模型一行：左为前 n_pcs 个主成分与标签的关联热图，右为 PC1–PC2 散点图。色标和图例放在最下面。"""
    right = eyes == "R"
    blocks = {m: _assoc_blocks(assoc, m, labels, n_pcs) for m in models}
    heights = [len(b) for _, b, _ in blocks[models[0]]]
    with plt.rc_context({"font.size": 8}):
        fig = plt.figure(figsize=(0.42 * n_pcs + 3.6, 1.8 * len(models) + 1.1), facecolor=SURFACE)
        outer = fig.add_gridspec(len(models) + 1, 2, width_ratios=[0.42 * n_pcs, 1.9], height_ratios=[1] * len(models) + [0.5],
                                 wspace=0.05, hspace=0.3)
        meshes = {}
        for r, m in enumerate(models):
            inner = outer[r, 0].subgridspec(len(heights), 1, height_ratios=heights, hspace=0.12)
            axes = [fig.add_subplot(inner[k]) for k in range(len(heights))]
            for ax, (metric, block, values) in zip(axes, blocks[m]):
                meshes[metric] = _draw_heatmap(ax, metric, block, values, fontsize=6.5)
            for ax in axes[:-1] if r == len(models) - 1 else axes:
                ax.set_xticklabels([])  # 各行的列对齐，只在最后一行标主成分编号
            axes[0].set_title(MODEL_NAMES[m], loc="left", color=INK, fontweight="bold", fontsize=9)
            _draw_scatter(fig.add_subplot(outer[r, 1]), scores[m], ratios[m], grades, right, size)
        axes[-1].set_xlabel("Principal component", color=INK_SECONDARY)

        bottom = outer[-1, 0].subgridspec(2, len(meshes) * 2 + 1, height_ratios=[1.6, 1],
                                          width_ratios=[1, 3] * len(meshes) + [1], wspace=0)
        for k, (metric, mesh) in enumerate(meshes.items()):
            _colorbar(fig, mesh, fig.add_subplot(bottom[1, 2 * k + 1]), metric, [0, 0.5, 1], orientation="horizontal")
        legend_ax = fig.add_subplot(outer[-1, 1])
        legend_ax.axis("off")
        legend_ax.legend(handles=_scatter_handles(), frameon=False, loc="center", ncol=2, labelcolor=INK,
                         handletextpad=0.2, columnspacing=0.8, fontsize=7, markerscale=0.8)
    return fig


def saliency_figure(images, maps, row_labels, col_labels, alpha=0.9, levels=(0.25, 0.5), panel=2.6):
    """每行一张图：原图 + 各主成分的显著图（叠在灰度原图上，视野圆外不显示）。
    原图保持原始分辨率，显著图 [K, h, w] 拉伸到原图大小叠加；红 = 推高，蓝 = 推低；
    不透明度随 |值| 增大（接近 0 处透明，露出原图，最大为 alpha）；
    等高线画在 ±levels × 上限处（红 = 正，蓝 = 负；越靠外越细）；
    每列共用以 0 为中心的色标（视野内 |值| 的 99% 分位）。"""
    n, k = len(images), len(col_labels)
    grays = [np.asarray(im.convert("L")) for im in images]
    fovs = [np.asarray(Image.fromarray(g).resize(m.shape[:0:-1])) > 10 for g, m in zip(grays, maps)]
    limits = [np.percentile(np.concatenate([np.abs(m[j][f]) for m, f in zip(maps, fovs)]), 99) for j in range(k)]

    fig, axes = plt.subplots(n + 1, k + 1, squeeze=False, figsize=(panel * (k + 1), panel * 0.92 * n + 0.6), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1] * n + [0.05], "wspace": 0.03, "hspace": 0.05})
    for i, (image, gray, m, fov) in enumerate(zip(images, grays, maps, fovs)):
        extent = (0, image.width, image.height, 0)
        axes[i, 0].imshow(image, interpolation="none")
        axes[i, 0].set_ylabel(row_labels[i], color=INK, fontsize=9)
        for j in range(k):
            mesh = plt.cm.ScalarMappable(plt.Normalize(-limits[j], limits[j]), "RdBu_r")
            rgba = mesh.to_rgba(m[j])
            rgba[..., 3] = alpha * np.clip(np.abs(m[j]) / limits[j], 0, 1) * fov
            axes[i, j + 1].imshow(gray, cmap="gray", interpolation="none")
            axes[i, j + 1].imshow(rgba, extent=extent, interpolation="bilinear")
            h, w = m[j].shape
            x, y = (np.arange(w) + 0.5) * image.width / w, (np.arange(h) + 0.5) * image.height / h
            widths = np.linspace(0.4, 0.8, len(levels))
            for sign, color in [(1, "#b2182b"), (-1, "#2166ac")]:
                axes[i, j + 1].contour(x, y, np.where(fov, sign * m[j], np.nan), levels=np.asarray(levels) * limits[j],
                                       colors=color, linewidths=widths)
            if i == n - 1:
                cbar = fig.colorbar(mesh, cax=axes[n, j + 1], orientation="horizontal", format="%.2g")
                cbar.outline.set_visible(False)
                cbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    axes[n, 0].axis("off")
    for ax, title in zip(axes[0], ["Image", *col_labels]):
        ax.set_title(title, color=INK, fontweight="bold", fontsize=10)
    for ax in axes[:n].flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    return fig
