import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import anndata as ad
    import scdiv
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    from pathlib import Path

    from _figutils import hill_number, set_paper_rc

    sns.set_theme(style="whitegrid")
    return (
        Path,
        ad,
        hill_number,
        mo,
        mpatches,
        np,
        pd,
        plt,
        scdiv,
        set_paper_rc,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure S1 (Rich et al. robustness)

    We reprocess one PBMC dataset through four pipelines (tool $\times$
    parameter-style) and check how much the resulting diversity moves.

    The cells below load the shared counts and the four clusterings, then
    compute the per-pipeline diversities, the HVG-set overlaps, and the shared
    embedding. The last cell assembles the figure.
    """)
    return


@app.cell
def _(Path):
    DATA_DIR = Path("./data/figS_rich")

    # (tool, mode) for each of the 6 cluster assignments.
    ASSIGNMENTS = [
        (t, m)
        for t in ("scanpy", "seurat")
        for m in ("default", "seurat_like", "scanpy_like")
    ]
    return ASSIGNMENTS, DATA_DIR


@app.cell
def _(DATA_DIR, ad):
    # QC'd integer counts (cells x genes), restricted to the union of the runs'
    # HVGs. The diversity step subsets these to each run's HVG columns.
    counts = ad.read_h5ad(DATA_DIR / "counts_hvg.h5ad")
    gene_pos = {g: i for i, g in enumerate(counts.var_names)}
    counts
    return counts, gene_pos


@app.cell
def _(ASSIGNMENTS, DATA_DIR, gene_pos, pd):
    # Each assignment: per-cell cluster labels + that pipeline's HVG gene set.
    def load_assignment(tool, mode):
        clusters = pd.read_csv(
            DATA_DIR / f"{tool}/{mode}_clusters.csv"
        ).set_index("barcode")["cluster"]
        hvg = (
            pd.read_csv(DATA_DIR / f"{tool}/{mode}_hvg.csv")["gene_id"]
            .astype(str)
            .tolist()
        )
        hvg_in = [g for g in hvg if g in gene_pos]
        return {
            "tool": tool,
            "mode": mode,
            "clusters": clusters,
            "hvg": hvg_in,
            "n_clusters": int(clusters.nunique()),
            "n_hvg": len(hvg_in),
        }

    assignments = [load_assignment(t, m) for t, m in ASSIGNMENTS]
    return (assignments,)


@app.cell
def _(assignments, pd):
    # Overview of the six loaded assignments
    pd.DataFrame(
        [{k: a[k] for k in ("tool", "mode", "n_clusters", "n_hvg")} for a in assignments]
    )
    return


@app.cell
def _(assignments):
    _bykey = {(a["tool"], a["mode"]): a for a in assignments}
    RUNS = {
        ("scanpy", "scanpy"): _bykey[("scanpy", "scanpy_like")],
        ("scanpy", "seurat"): _bykey[("scanpy", "seurat_like")],
        ("seurat", "scanpy"): _bykey[("seurat", "scanpy_like")],
        ("seurat", "seurat"): _bykey[("seurat", "seurat_like")],
    }
    BARKEYS = list(RUNS)
    BARSHORT = ["Sc/Sc", "Sc/Se", "Se/Sc", "Se/Se"]
    return BARKEYS, BARSHORT, RUNS


@app.cell
def _(BARKEYS, BARSHORT, RUNS, counts, gene_pos, hill_number, pd, scdiv):
    _rows = []
    for _k in BARKEYS:
        _a = RUNS[_k]
        _props = _a["clusters"].value_counts().to_numpy()
        _cols = [gene_pos[g] for g in _a["hvg"]]
        _sub = counts[:, _cols].copy()
        _sub.obs["cluster"] = (
            _a["clusters"].reindex(_sub.obs_names).astype("string").to_numpy()
        )
        scdiv.tl.diversity(
            _sub,
            order=2.0,
            cell_type_key="cluster",
            use_highly_variable=False,
            alpha=0.5,
        )
        _rows.append(
            {"Hill": hill_number(_props), "LC": float(_sub.uns["scdiv_diversity"])}
        )
    div_df = pd.DataFrame(_rows, index=BARSHORT)
    div_df
    return (div_df,)


@app.cell
def _(BARKEYS, BARSHORT, RUNS, np, pd):
    # HVG-set overlap between the four pipelines (intersection / union).
    _jsets = [frozenset(RUNS[_k]["hvg"]) for _k in BARKEYS]
    jaccard = np.array(
        [[len(_a & _b) / len(_a | _b) for _b in _jsets] for _a in _jsets]
    )
    pd.DataFrame(jaccard, index=BARSHORT, columns=BARSHORT).round(2)
    return (jaccard,)


@app.cell
def _(DATA_DIR, pd):
    _emb = pd.read_csv(DATA_DIR / "scanpy/default_clusters.csv").set_index("barcode")
    umap_xy = _emb[["UMAP1", "UMAP2"]].to_numpy()
    barcodes = _emb.index
    celltype = (
        pd.read_csv(DATA_DIR / "seurat/celltype.csv")
        .set_index("barcode")["celltype"]
        .reindex(barcodes)
    )
    ct_order = celltype.value_counts().index.tolist()
    celltype.value_counts()
    return barcodes, celltype, ct_order, umap_xy


@app.cell
def _(ct_order, plt):
    # Figure styling: per-cell-type colours (tab20, in frequency order) and the
    # panel row / column / bar labels. Pure presentation, used only by the
    # figure-assembly cell below.
    ct_color = {t: plt.get_cmap("tab20")(_i % 20) for _i, t in enumerate(ct_order)}
    TOOLS = ["scanpy", "seurat"]
    STYLES = ["scanpy", "seurat"]
    ROWLAB = {"scanpy": "Scanpy", "seurat": "Seurat"}
    COLLAB = {"scanpy": "Scanpy-style", "seurat": "Seurat-style"}
    BARLAB = [
        "Scanpy\nSc-style",
        "Scanpy\nSe-style",
        "Seurat\nSc-style",
        "Seurat\nSe-style",
    ]
    return BARLAB, COLLAB, ROWLAB, STYLES, TOOLS, ct_color


@app.cell
def _(
    BARLAB,
    BARSHORT,
    COLLAB,
    ROWLAB,
    RUNS,
    STYLES,
    TOOLS,
    barcodes,
    celltype,
    ct_color,
    ct_order,
    div_df,
    jaccard,
    mpatches,
    np,
    pd,
    plt,
    set_paper_rc,
    sns,
    umap_xy,
):
    # === Assembled SI figure: Rich et al. pipeline-robustness (2x2) ===
    # Four distinct pipelines = tool {Scanpy, Seurat} x param-style {Scanpy-, Seurat-}.
    # A: four clusterings on a shared UMAP, colored by majority CellTypist cell type.
    # B: HVG-set overlap (Jaccard) between the four pipelines.
    # C: Hill vs LC(alpha=0.5) diversity (q=2). D: same, as % deviation from the mean.
    sns.set_style("whitegrid")
    set_paper_rc(font=12, tick=10, legend=8)
    _MM = 1 / 25.4
    _W = 179 * _MM

    _HILL_C, _LC_C = "#c53030", "black"  # match fig1b: Hill red, LC black
    _y0, _y1 = float(umap_xy[:, 1].min()), float(umap_xy[:, 1].max())
    _yr = _y1 - _y0

    # --- layout: top = [2x2 UMAP | legend], bottom = [Jaccard | abs | %dev] -------
    _figS = plt.figure(figsize=(_W, _W * 1.0), dpi=300)
    _gs = _figS.add_gridspec(2, 1, height_ratios=[1.9, 1.0], hspace=0.18)
    _gtop = _gs[0].subgridspec(1, 2, width_ratios=[2.35, 0.9], wspace=0.0)
    _gumap = _gtop[0].subgridspec(2, 2, hspace=0.04, wspace=0.02)
    _gbot = _gs[1].subgridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.42)

    _ax_first = None
    for _ri, _tool in enumerate(TOOLS):
        for _ci, _style in enumerate(STYLES):
            _a = RUNS[(_tool, _style)]
            _ax = _figS.add_subplot(_gumap[_ri, _ci])
            if _ri == 0 and _ci == 0:
                _ax_first = _ax
            _lab = _a["clusters"].reindex(barcodes)
            _maj = (
                pd.DataFrame({"cl": _lab, "ct": celltype})
                .dropna()
                .groupby("cl")["ct"]
                .agg(lambda s: s.value_counts().idxmax())
            )
            _cc = [ct_color.get(_maj.get(v), (0.8, 0.8, 0.8, 1.0)) for v in _lab]
            _ax.scatter(
                umap_xy[:, 0], umap_xy[:, 1], s=0.8, c=_cc, linewidths=0, rasterized=True
            )
            _ax.set_aspect("equal", adjustable="box")
            _ax.set_ylim(_y0 - 0.03 * _yr, _y1 + 0.16 * _yr)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.grid(False)
            for _s in _ax.spines.values():
                _s.set_visible(False)
            _ax.text(
                0.5,
                0.98,
                f"{_a['n_clusters']} clusters",
                transform=_ax.transAxes,
                va="top",
                ha="center",
                fontsize=9,
                color="0.25",
            )
            if _ri == 0:
                _ax.set_title(
                    COLLAB[_style], fontsize=12, fontweight="bold", pad=10
                )
            if _ci == 0:
                _ax.text(
                    -0.05,
                    0.5,
                    ROWLAB[_tool],
                    transform=_ax.transAxes,
                    rotation=0,
                    va="center",
                    ha="right",
                    fontsize=12,
                    fontweight="bold",
                )

    # cell-type legend, vertical, to the right of the UMAP block
    _axleg = _figS.add_subplot(_gtop[1])
    _axleg.axis("off")
    _handles = [mpatches.Patch(color=ct_color[t], label=t) for t in ct_order]
    _axleg.legend(
        handles=_handles,
        loc="center left",
        ncol=1,
        fontsize=8,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.5,
        labelspacing=0.45,
        borderaxespad=0,
    )

    # Panel B: HVG-set Jaccard heatmap
    _axJ = _figS.add_subplot(_gbot[0])
    _axJ.imshow(jaccard, cmap="Blues", vmin=0, vmax=1)
    _axJ.set_xticks(range(4))
    _axJ.set_yticks(range(4))
    _axJ.set_xticklabels(BARLAB, fontsize=7)
    _axJ.set_yticklabels(BARLAB, fontsize=7)
    _axJ.grid(False)
    for _i in range(4):
        for _j in range(4):
            _axJ.text(
                _j,
                _i,
                f"{jaccard[_i, _j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if jaccard[_i, _j] > 0.6 else "0.15",
            )
    _axJ.set_title("HVG-set overlap (Jaccard)", fontsize=10, pad=6)

    # Panel C: absolute bars (horizontal)
    _axAbs = _figS.add_subplot(_gbot[1])
    _y = np.arange(len(div_df))
    _w = 0.4
    _axAbs.barh(_y - _w / 2, div_df["Hill"], _w, color=_HILL_C, label="Hill")
    _axAbs.barh(_y + _w / 2, div_df["LC"], _w, color=_LC_C, label="LC")
    _axAbs.set_yticks(_y)
    _axAbs.set_yticklabels(BARSHORT, fontsize=8)
    _axAbs.invert_yaxis()
    _axAbs.set_xlabel("Diversity", fontsize=10)
    _axAbs.tick_params(axis="x", labelsize=9)
    _axAbs.set_xlim(0, 15.0)
    _axAbs.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    for _yi, _h, _l in zip(_y, div_df["Hill"], div_df["LC"]):
        _axAbs.text(
            _h + 0.25,
            _yi - _w / 2,
            f"{_h:.1f}",
            ha="left",
            va="center",
            fontsize=7,
            color="0.2",
        )
        _axAbs.text(
            _l + 0.25,
            _yi + _w / 2,
            f"{_l:.2f}",
            ha="left",
            va="center",
            fontsize=7,
            color="0.2",
        )

    # Panel D: % deviation bars (horizontal)
    _hp = (div_df["Hill"] / div_df["Hill"].mean() - 1) * 100
    _lp = (div_df["LC"] / div_df["LC"].mean() - 1) * 100
    _axDev = _figS.add_subplot(_gbot[2])
    _axDev.axvline(0, color="0.5", lw=0.8, zorder=1)
    _axDev.barh(_y - _w / 2, _hp, _w, color=_HILL_C, label="Hill")
    _axDev.barh(_y + _w / 2, _lp, _w, color=_LC_C, label="LC")
    _axDev.set_yticks(_y)
    _axDev.set_yticklabels(BARSHORT, fontsize=8)
    _axDev.invert_yaxis()
    _axDev.set_xlabel("% deviation from mean", fontsize=10)
    _axDev.tick_params(axis="x", labelsize=9)
    _axDev.set_xlim(-20, 34)
    _axDev.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    for _yi, _h, _l in zip(_y, _hp, _lp):
        _axDev.text(
            _h + (1.0 if _h >= 0 else -1.0),
            _yi - _w / 2,
            f"{_h:+.1f}%",
            ha="left" if _h >= 0 else "right",
            va="center",
            fontsize=7,
            color="0.2",
        )
        _axDev.text(
            _l + (1.0 if _l >= 0 else -1.0),
            _yi + _w / 2,
            f"{_l:+.1f}%",
            ha="left" if _l >= 0 else "right",
            va="center",
            fontsize=7,
            color="0.2",
        )

    _figS.tight_layout(rect=[0, 0, 1, 0.99])
    _figS.canvas.draw()
    _r = _figS.canvas.get_renderer()
    _inv = _figS.transFigure.inverted()


    def _x0(ax):
        return _inv.transform((ax.get_tightbbox(_r).x0, 0))[0]


    def _y1(ax):
        return ax.get_position().y1


    def _put(x, y, s):
        _figS.text(x, y, s, ha="left", va="bottom", fontsize=14, fontweight="bold")


    _dy = 0.014
    _xleft = min(_x0(_ax_first), _x0(_axJ))  # A and B vertically aligned
    _yrow = (
        max(_y1(_axJ), _y1(_axAbs), _y1(_axDev)) + _dy
    )  # B, C, D on one baseline
    _put(_xleft, _y1(_ax_first) + _dy, "A")
    _put(_xleft, _yrow, "B")
    _put(_x0(_axAbs), _yrow, "C")
    _put(_x0(_axDev), _yrow, "D")

    _figS.savefig("./figures/figS_rich.pdf", bbox_inches="tight")
    _figS
    return


if __name__ == "__main__":
    app.run()
