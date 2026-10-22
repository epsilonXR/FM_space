"""画图。"""

import matplotlib.pyplot as plt
import numpy as np

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


def heatmap(assoc, model, labels, n_pcs=20):
    """PC × 标签的关联强度，有序标签（|ρ|）和二值标签（2|AUC − 0.5|）分成上下两块，各自一个色标。"""
    d = assoc[(assoc["model"] == model) & (assoc["component"] <= n_pcs)]
    d = d.assign(strength=d["value"].abs().where(d["metric"] == "spearman", (2 * (d["value"] - 0.5)).abs()))
    table = d.pivot(index="label", columns="component", values="strength")
    metric = d.drop_duplicates("label").set_index("label")["metric"]
    blocks = [(m, [l for l in labels if metric[l] == m]) for m in ("spearman", "auroc")]
    blocks = [(m, b) for m, b in blocks if b]

    fig, axes = plt.subplots(
        len(blocks), 2, squeeze=False,
        figsize=(0.45 * n_pcs + 2.5, 0.4 * len(labels) + 1.2),
        gridspec_kw={"height_ratios": [len(b) for _, b in blocks], "width_ratios": [n_pcs, 0.4], "hspace": 0.15, "wspace": 0.03},
    )
    for (m, block), (ax, cax) in zip(blocks, axes):
        values = table.loc[block].to_numpy()
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="black" if v > 0.55 else "white")
        ax.set_yticks(range(len(block)), [LABEL_NAMES[l] for l in block])
        ax.set_xticks(range(n_pcs), range(1, n_pcs + 1))
        ax.tick_params(length=0)
        cbar = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1] if len(block) > 1 else [0, 1], format="%.1f")
        cbar.set_label(METRIC_NAMES[m], rotation=0, ha="left", va="center")

    for ax in axes[:-1, 0]:
        ax.set_xticklabels([])
    axes[-1, 0].set_xlabel("Principal component")
    axes[0, 0].set_title(MODEL_NAMES[model])
    return fig
