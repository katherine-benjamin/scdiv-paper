"""Extract intermediate data for fig 2.

Sources (Mouse organogenesis atlas, not published here):
  data/fig2/mouse_plot.h5ad            11.6M cells x 2500 genes
  data/combined_cell_annotations.csv   per-cell trajectory/subtype labels

Outputs (stored in paper_figures/data/fig2/):
  mouse_plot_slim.parquet          per-cell UMAP + major_trajectory + leiden
  trajectory_subtype_counts.csv    cell counts per (trajectory, subtype)
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SRC_H5AD = Path("data/fig2/mouse_plot.h5ad")
SRC_ANNOT = Path("data/combined_cell_annotations.csv")
OUT = Path("paper_figures/data/fig2")

# Panel C: counts per (major_trajectory, celltype_update).
annot = pd.read_csv(SRC_ANNOT, usecols=["major_trajectory", "celltype_update"])
counts = (
    annot.groupby(["major_trajectory", "celltype_update"], observed=True)
    .size()
    .reset_index(name="count")
)
counts.to_csv(OUT / "trajectory_subtype_counts.csv", index=False)
print("wrote", OUT / "trajectory_subtype_counts.csv", tuple(counts.shape))

# Panels B and D: UMAP coords + trajectory + leiden per cell.
a = ad.read_h5ad(SRC_H5AD, backed="r")
umap = np.asarray(a.obsm["X_umap"], dtype=np.float32)
slim = pd.DataFrame(
    {
        "major_trajectory": a.obs["major_trajectory"].astype("category").values,
        "leiden": a.obs["leiden"].astype(str).astype("category").values,
        "umap_x": umap[:, 0].round(2),
        "umap_y": umap[:, 1].round(2),
    }
)
slim.to_parquet(OUT / "mouse_plot_slim.parquet", compression="zstd", index=False)
print("wrote", OUT / "mouse_plot_slim.parquet", tuple(slim.shape))
