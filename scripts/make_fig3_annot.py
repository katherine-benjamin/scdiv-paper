"""Extract per-slide tissue annotations from the raw MOSTA E1S1 slides for
fig3's annotation panels.

Needs the raw MOSTA slides under data/ (see README).
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

RAW_DIR = Path("data")
OUT = Path("paper_figures/data/fig3/annot")
OUT.mkdir(parents=True, exist_ok=True)

SECTION = "E1S1"
STAGES = ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]

for stage in STAGES:
    a = ad.read_h5ad(RAW_DIR / f"{stage}_{SECTION}.MOSTA.h5ad", backed="r")
    obs = pd.DataFrame({"annotation": a.obs["annotation"].values})
    obs.index = obs.index.astype(str)
    slim = ad.AnnData(
        obs=obs,
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=np.float32)},
    )
    slim.uns["annotation_colors"] = np.asarray(a.uns["annotation_colors"])
    slim.write_h5ad(OUT / f"{stage}_{SECTION}.h5ad")
    print("wrote", stage, slim.shape)
