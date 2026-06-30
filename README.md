This directory contains the notebooks and data used to generate the figures in
the paper "Diversity in transcriptomics without cell types".

LC diversity is computed with the `scdiv` package available at https://github.com/katherine-benjamin/scdiv. 

Each figure in the paper is associated to one [marimo](https://marimo.io) notebook under `notebooks/` that reads its
inputs from `data/` and writes to `figures/`.

| Notebook | Paper figure | Output | Inputs |
|----------|--------------|--------|--------|
| `notebooks/fig1.py` | Figure 1 | `fig1a`, `fig1b`, `fig1c`, `fig1d` (.svg/.png) | none (synthetic) |
| `notebooks/fig2.py` | Figure 2 | `fig2_mouse` (.pdf/.png) | `data/fig2/` |
| `notebooks/fig3.py` | Figure 3 | `fig3.pdf` | `data/fig3/` |
| `notebooks/fig4.py` | Figure 4 | `fig4.pdf` | `data/fig4/` |
| `notebooks/figS_rich.py` | SI Rich et al. figure | `figS_rich.pdf` | `data/figS_rich/` |

## Running

```bash
uv sync                                   # or: pip install -e .
uv run marimo edit notebooks/fig1.py      # open interactively
```

Running all cells writes the figure into `figures/`. Each notebook is also a
plain Python script, so you can also run it directly:

```bash
uv run python notebooks/fig1.py
```

Shared helpers  can be found in `notebooks/_figutils.py`.

## Data and provenance

To avoid redistributing heavy data sets, we include here in `data/` only the intermediate
files necessary to generate the figures in our paper. The scripts in
`scripts/` show how the fig2 and fig3 inputs were produced from those data
sets, but they are not needed to reproduce the figures.

- **fig2**: from the mouse development single-cell atlas (Qiu et al., 2024; GEO
  [GSE228590](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE228590)).
  `scripts/make_fig2_data.py` processes the atlas to `mouse_plot_slim.parquet`
  (per-cell UMAP + trajectory + leiden) and `trajectory_subtype_counts.csv`
  (cell counts per trajectory/subtype). `mouse_scdiv_results.csv` /
  `mouse_hill_results.csv` holds the diversity figures.
- **fig3**: from the MOSTA Stereo-seq atlas (Chen et al., 2022; CNGB
  [CNP0001543](https://db.cngb.org/search/project/CNP0001543),
  [portal](https://db.cngb.org/stomics/mosta/)), slice E1S1 at bin50, stages
  E9.5 to E16.5. `scripts/make_fig3_annot.py` extracts `annot/`
  (per-slide spatial coords + tissue annotation). `scdiv_cache/` holds the
  per-stage diversity results, which the fig3 notebook computes from the raw
  slides on first run.
- **fig4**: from upcoming mouse stem-cell-based
  embryo models by Smirnova et al. `scdiv_aggregoid.csv` is the per-sample summary (number of cell
  types, Hill number, cell-type and singleton LC diversity), computed by running
  scdiv on each (condition, timepoint) sample. The raw scRNA-seq is not yet
  public.
- **figS_rich**: obtained by following the
  [pachterlab/RMEJLBASBMP_2024](https://github.com/pachterlab/RMEJLBASBMP_2024)
  "modern" reproduction pipeline (Rich et al., 2026; Scanpy + Seurat +
  CellTypist) on the PBMC raw kb-python matrix from CaltechDATA (doi
  `10.22002/rwdfx-j4z76`).
  `counts_hvg.h5ad` is the QC'd integer counts restricted to the union of the
  runs' highly variable genes. The
  `*_clusters.csv` / `*_hvg.csv` / `celltype.csv` files are the per-run
  clusterings, HVG sets, and cell-type labels.
