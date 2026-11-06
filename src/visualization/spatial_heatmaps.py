"""
Spatial Representation & Similarity Heatmap Visualizations.
Produces publication-quality 3x3 multi-panel comparative figures matching Phung_Nhat_Minh_BT_GK.ipynb.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_spatial_comparison(
    display_image: np.ndarray,
    results: Dict[str, Dict[str, Any]],
    queries: Dict[str, Tuple[float, float]],
    save_path: Path
):
    """
    Renders a 3x3 comparison figure matching Cell 24 of Phung_Nhat_Minh_BT_GK.ipynb:
    Columns: [PCA RGB], [Cosine Sim vs A (Cánh tay)], [Cosine Sim vs B (Nền)]
    Rows: VAE, ViT, V-JEPA 2
    """
    models = list(results.keys())
    fig, axes = plt.subplots(len(models), 3, figsize=(14, 12), constrained_layout=True)
    if len(models) == 1:
        axes = np.expand_dims(axes, 0)

    for i, name in enumerate(models):
        res = results[name]
        var_pct = res.get("var_pct", res.get("variance_explained", 0.0))
        rgb_map = res.get("rgb_map", res.get("pca_rgb"))
        
        # Cosine maps & mean values
        if "cos_a" in res:
            cos_a = res["cos_a"]
            mean_a = res["mean_a"]
            r_a, c_a = res["idx_a"]
        else:
            cos_a = res["cosine_maps"]["A_subject"]
            mean_a = res["mean_cosine"]["A_subject"]
            r_a, c_a = res["query_indices"]["A_subject"]

        if "cos_b" in res:
            cos_b = res["cos_b"]
            mean_b = res["mean_b"]
            r_b, c_b = res["idx_b"]
        else:
            cos_b = res["cosine_maps"]["B_background"]
            mean_b = res["mean_cosine"]["B_background"]
            r_b, c_b = res["query_indices"]["B_background"]

        # Col 1: PCA RGB
        axes[i, 0].imshow(rgb_map)
        axes[i, 0].set_title(f"{name} - PCA RGB\n(Var: {var_pct:.1f}%)", fontsize=11, fontweight="bold")
        axes[i, 0].set_ylabel(name, fontsize=12, fontweight="bold")
        axes[i, 0].axis("off")

        # Col 2: Cosine vs Subject A
        im1 = axes[i, 1].imshow(cos_a, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 1].plot(c_a, r_a, 'kX', markersize=12, markeredgewidth=2)
        axes[i, 1].set_title(f"{name} - Cosine Sim vs A (Cánh tay)\nTB: {mean_a:.3f}", fontsize=10)
        axes[i, 1].axis("off")
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Col 3: Cosine vs Background B
        im2 = axes[i, 2].imshow(cos_b, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[i, 2].plot(c_b, r_b, 'gX', markersize=12, markeredgewidth=2)
        axes[i, 2].set_title(f"{name} - Cosine Sim vs B (Nền)\nTB: {mean_b:.3f}", fontsize=10)
        axes[i, 2].axis("off")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle("ĐỐI CHIẾU KHÔNG GIAN BIỂU DIỄN VAE vs ViT vs V-JEPA 2 TRÊN ROBOT BAXTER", fontsize=13, fontweight="bold")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
