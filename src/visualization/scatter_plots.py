"""
2D PCA Patch Embedding Scatter Plot Visualization.
Produces publication-quality 1x3 multi-panel comparative figures matching Phung_Nhat_Minh_BT_GK.ipynb.
"""

from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np


def plot_patch_scatter(results: Dict[str, Dict[str, Any]], save_path: Path):
    """
    Renders 2D PCA patch projection side-by-side matching Cell 27 of Phung_Nhat_Minh_BT_GK.ipynb:
    Points represent individual spatial patches, colored by Cosine similarity to Query A (Cánh tay).
    """
    models = list(results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(18, 5.5), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]

    for i, name in enumerate(models):
        res = results[name]
        proj = res.get("projected", res.get("projected_pca"))
        cos_a = res.get("cos_a", res.get("cosine_maps", {}).get("A_subject"))
        sims_a = cos_a.reshape(-1)
        w = cos_a.shape[1]

        if "idx_a" in res:
            idx_a = res["idx_a"][0] * w + res["idx_a"][1]
            idx_b = res["idx_b"][0] * w + res["idx_b"][1]
        else:
            qa_row, qa_col = res["query_indices"]["A_subject"]
            qb_row, qb_col = res["query_indices"]["B_background"]
            idx_a = qa_row * w + qa_col
            idx_b = qb_row * w + qb_col

        rankme_val = res.get("rankme", res.get("effective_rank", 0.0))

        sc = axes[i].scatter(proj[:, 0], proj[:, 1], c=sims_a, cmap="coolwarm", alpha=0.8, s=40)
        axes[i].scatter(proj[idx_a, 0], proj[idx_a, 1], c="black", marker="X", s=150, label="Điểm A (Cánh tay)")
        axes[i].scatter(proj[idx_b, 0], proj[idx_b, 1], c="green", marker="X", s=150, label="Điểm B (Nền)")

        axes[i].set_title(f"{name} (RankMe = {rankme_val:.1f})", fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Principal Component 1")
        axes[i].set_ylabel("Principal Component 2")
        axes[i].grid(True, linestyle="--", alpha=0.4)
        axes[i].legend(loc="upper right", fontsize=9)
        fig.colorbar(sc, ax=axes[i], label="Cosine với Cánh tay A", fraction=0.046, pad=0.04)

    plt.suptitle("PHÂN BỐ CÁC PATCH TRÊN KHÔNG GIAN 2D PCA", fontsize=13, fontweight="bold")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
