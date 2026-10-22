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


def heatmap(assoc, model, labels, n_pcs=20):
    """PC × 标签的关联强度：Spearman 取 |ρ|，AUROC 取 2|AUC − 0.5|。
    有序标签与二值标签之间空一行。"""
    d = assoc[(assoc["model"] == model) & (assoc["component"] <= n_pcs)]
    d = d.assign(strength=d["value"].abs().where(d["metric"] == "spearman", (2 * (d["value"] - 0.5)).abs()))
    table = d.pivot(index="label", columns="component", values="strength")
    metric = d.drop_duplicates("label").set_index("label")["metric"]

    rows, names = [], []
    for i, label in enumerate(labels):
        if i and metric[label] != metric[labels[i - 1]]:
            rows.append(np.full(n_pcs, np.nan))
            names.append("")
        rows.append(table.loc[label].to_numpy())
        names.append(LABEL_NAMES[label])

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(rows) + 1))
    image = ax.imshow(rows, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(rows)), names)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks(range(0, n_pcs, 2), range(1, n_pcs + 1, 2))
    ax.set_xlabel("Principal component")
    ax.set_title(MODEL_NAMES[model])
    fig.colorbar(image, ax=ax, pad=0.02)
    return fig
