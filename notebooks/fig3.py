import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import anndata as ad
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scanpy as sc
    from scipy import sparse

    import scdiv

    return Path, ad, mo, np, pd, plt, sc, scdiv, sparse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 3: spatial diversity over mouse organogenesis (MOSTA)

    For each of eight Stereo-seq slides (E9.5 to E16.5) we run the full scdiv
    spatial pipeline: HVG selection, kNN smoothing in PC space, a hex
    partition, then similarity-sensitive local (alpha) and global (gamma)
    diversity per region (q=2, similarity alpha=0.5). The heavy per-slide
    computation is cached on disk, so `run_stage` returns instantly here; the
    raw slides (~100 GB) are only needed to repopulate the cache.

    The cells below load the cached per-stage results and the tissue
    annotations, and summarise them into one table. The final cell assembles
    the figure (case-study maps, the per-stage grid, and the metacommunity
    trajectories).
    """)
    return


@app.cell
def _(np, sc, sparse):
    def knn_smooth(adata, *, n_neighbors=15, n_pcs=50, key_added="smoothed"):
        """Smooth each cell with its k nearest neighbours in PC space.

        Requires ``adata.var["highly_variable"]``. 
        Smoothed expression stored in ``adata.obsm[key_added]``.
        """
        from pynndescent import NNDescent

        sc.pp.pca(adata, n_comps=n_pcs, mask_var="highly_variable")
        nnd = NNDescent(adata.obsm["X_pca"], n_neighbors=n_neighbors, random_state=0)
        idx, _ = nnd.neighbor_graph
        n = adata.n_obs
        avg = sparse.csr_matrix(
            (
                np.full(n * n_neighbors, 1.0 / n_neighbors, dtype=np.float32),
                (np.repeat(np.arange(n), n_neighbors), idx.ravel()),
            ),
            shape=(n, n),
        )
        hvg = adata.var["highly_variable"].to_numpy()
        adata.obsm[key_added] = np.asarray(
            (avg @ adata.X[:, hvg]).toarray(), dtype=np.float32
        )

    return (knn_smooth,)


@app.cell
def _(Path):
    # Read per-stage diversity results

    DATA_DIR = Path("./data/fig3")
    CACHE_DIR = DATA_DIR / "scdiv_cache"
    ANNOT_DIR = DATA_DIR / "annot"
    RAW_DIR = DATA_DIR / "raw"

    SECTION = "E1S1"
    STAGES = ["E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]

    # MOSTA is binned at bin50, so one spatial-coord unit = 25 µm.
    SPATIAL_UNIT_UM = 25.0

    def path_for_sample(stage, section):
        return RAW_DIR / f"{stage}_{section}.MOSTA.h5ad"

    return (
        ANNOT_DIR,
        CACHE_DIR,
        SECTION,
        SPATIAL_UNIT_UM,
        STAGES,
        path_for_sample,
    )


@app.cell
def _():
    # Hyperparameters 

    N_NEIGHBORS = 15
    N_PCS = 50
    N_TOP_GENES = 2000
    REGION_SIZE = 15.0
    PARTITION_MIN_CELLS = 20
    MIN_DENSITY = 0.25
    ORDER = 2
    ALPHA = 0.5
    return (
        ALPHA,
        MIN_DENSITY,
        N_NEIGHBORS,
        N_PCS,
        N_TOP_GENES,
        ORDER,
        PARTITION_MIN_CELLS,
        REGION_SIZE,
    )


@app.cell
def _(CACHE_DIR, ad, knn_smooth, np, path_for_sample, sc, scdiv):
    import warnings as _warnings

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def run_stage(
        stage_name,
        section,
        *,
        n_neighbors=15,
        n_pcs=50,
        n_top_genes=2000,
        region_size=15.0,
        partition_min_cells=20,
        min_density=0.25,
        order=2,
        alpha=0.5,
        use_cache=True,
    ):
        """Run the full scdiv spatial pipeline on one slide and return per-region
        summaries: per-region alpha/gamma/zero_frac, the metacommunity scalars,
        the non-region singleton diversity, region centroids, and counts.

        Results are cached on disk keyed by the hyperparameter tuple.
        """
        cache_key = (
            f"{stage_name}_{section}_hvg{n_top_genes}_nn{n_neighbors}_npcs{n_pcs}"
            f"_rs{region_size}_rmc{partition_min_cells}"
            f"_md{min_density}_q{order}_a{alpha}.npz"
        )
        cache_path = CACHE_DIR / cache_key
        if use_cache and cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            return {
                k: z[k].item() if z[k].dtype == object else z[k].tolist()
                for k in z.files
            }

        slide = ad.read_h5ad(path_for_sample(stage_name, section))
        sc.pp.highly_variable_genes(slide, n_top_genes=n_top_genes, flavor="seurat")
        knn_smooth(slide, n_neighbors=n_neighbors, n_pcs=n_pcs)

        scdiv.tl.diversity(
            slide,
            order=order,
            groupby=None,
            cell_type_key=None,
            obsm="smoothed",
            alpha=alpha,
            key_added="scdiv_global_singleton",
        )

        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore", message="Total cell area exceeds region area"
            )
            scdiv.spatial.partition(
                slide,
                method="hex",
                region_size=region_size,
                min_cells=partition_min_cells,
                cell_radius=0.5,
                min_density=min_density,
            )

        for mode_name, key in [("alpha_norm", "scdiv_alpha"), ("gamma", "scdiv_gamma")]:
            scdiv.tl.diversity(
                slide,
                order=order,
                groupby="spatial_region",
                cell_type_key=None,
                mode=mode_name,
                obsm="smoothed",
                alpha=alpha,
                aggregate=True,
                key_added=key,
            )
        scdiv.tl.sparsity(slide, region_key="spatial_region", key_added="zero_frac")

        out = {
            "stage": stage_name,
            "section": section,
            "n_cells": int(slide.n_obs),
            "n_regions": int(slide.obs["spatial_region"].cat.categories.size),
            "alpha": dict(slide.uns["scdiv_alpha"]),
            "gamma": dict(slide.uns["scdiv_gamma"]),
            "alpha_meta": float(slide.uns["scdiv_alpha_metacommunity"]),
            "gamma_meta": float(slide.uns["scdiv_gamma_metacommunity"]),
            "global_singleton": float(slide.uns["scdiv_global_singleton"]),
            "zero_frac": dict(slide.uns["zero_frac"]),
            "region_centers": dict(
                slide.uns["spatial_region_params"]["region_centers"]
            ),
            "region_size": float(region_size),
        }
        np.savez(cache_path, **{k: np.array(v, dtype=object) for k, v in out.items()})
        return out

    return (run_stage,)


@app.cell
def _(
    ALPHA,
    MIN_DENSITY,
    N_NEIGHBORS,
    N_PCS,
    N_TOP_GENES,
    ORDER,
    PARTITION_MIN_CELLS,
    REGION_SIZE,
    SECTION,
    STAGES,
    run_stage,
):
    # Per-stage results (cache hit for each stage with the shipped cache).
    stage_summaries = {
        s: run_stage(
            s,
            SECTION,
            n_neighbors=N_NEIGHBORS,
            n_pcs=N_PCS,
            n_top_genes=N_TOP_GENES,
            region_size=REGION_SIZE,
            partition_min_cells=PARTITION_MIN_CELLS,
            min_density=MIN_DENSITY,
            order=ORDER,
            alpha=ALPHA,
        )
        for s in STAGES
    }
    return (stage_summaries,)


@app.cell
def _(ANNOT_DIR, SECTION, ad, stage_summaries):
    # Tissue annotations extracted from the raw slides (scripts/make_fig3_annot.py):

    pub_stage = "E16.5"
    stage_adatas = {
        s: ad.read_h5ad(ANNOT_DIR / f"{s}_{SECTION}.h5ad") for s in stage_summaries
    }
    pub_adata = stage_adatas[pub_stage]
    pub_adata
    return pub_adata, pub_stage, stage_adatas


@app.cell
def _(pd, stage_summaries):
    # Per-stage scalar summary

    summary = pd.DataFrame(
        [
            {
                "stage": s,
                "n_cells": d["n_cells"],
                "n_regions": d["n_regions"],
                "alpha_meta": d["alpha_meta"],
                "gamma_meta": d["gamma_meta"],
                "beta": d["gamma_meta"] / d["alpha_meta"],
                "global_singleton": d["global_singleton"],
            }
            for s, d in stage_summaries.items()
        ]
    ).set_index("stage")
    summary.round(3)
    return


@app.cell
def _(SPATIAL_UNIT_UM):
    # Figure styling: anchored µm scale bar helper. Pure presentation, used only
    # by the figure-assembly cell below.
    def add_scale_bar(
        ax, *, length_um=None, location="lower right", color="black", label=None
    ):
        """Anchored µm scale bar; auto-picks a round length from the x-range."""
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        from matplotlib.font_manager import FontProperties

        if length_um is None:
            xlim = ax.get_xlim()
            target_um = abs(xlim[1] - xlim[0]) * SPATIAL_UNIT_UM * 0.2
            candidates = [
                c for c in (5000, 2000, 1000, 500, 200, 100) if c <= target_um
            ]
            length_um = candidates[0] if candidates else 100

        bar_units = length_um / SPATIAL_UNIT_UM
        if label is None:
            label = f"{length_um:g} µm" if length_um < 1000 else f"{length_um / 1000:g} mm"
        ax.add_artist(
            AnchoredSizeBar(
                ax.transData,
                bar_units,
                label,
                loc=location,
                pad=0.4,
                color=color,
                frameon=False,
                size_vertical=bar_units * 0.04,
                fontproperties=FontProperties(size=9),
            )
        )

    return (add_scale_bar,)


@app.cell
def _(
    STAGES,
    add_scale_bar,
    np,
    plt,
    pub_adata,
    pub_stage,
    stage_adatas,
    stage_summaries,
):
    import seaborn as sns
    from matplotlib.patches import RegularPolygon as _RegPoly
    from matplotlib.collections import PatchCollection as _PatchColl

    from _figutils import set_paper_rc

    set_paper_rc(font=12, tick=10, legend=9)
    sns.set_style("whitegrid")


    def _draw_annotation(ax, adata, *, s):
        """Scatter cells coloured by their ``annotation`` category.

        Colours come from ``adata.uns['annotation_colors']`` (tab20 fallback).
        Returns ``(handles, labels)`` for building a legend. Axis limits/aspect
        are left to the caller, which differs between the case-study and grid
        panels.
        """
        xy = adata.obsm["spatial"]
        ann = adata.obs["annotation"]
        cats = ann.cat.categories
        palette = adata.uns.get("annotation_colors")
        ann_arr = ann.to_numpy()
        handles, labels = [], []
        for i, cat in enumerate(cats):
            mask = ann_arr == cat
            color = (
                palette[i]
                if palette is not None and i < len(palette)
                else plt.get_cmap("tab20")(i / max(len(cats) - 1, 1))
            )
            handles.append(
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    color=color,
                    s=s,
                    linewidths=0,
                    rasterized=True,
                )
            )
            labels.append(cat)
        return handles, labels


    def _draw_hex(ax, stage_data, key, *, cmap, vmin=None, vmax=None, linewidth=0.15):
        """Draw one slide's hex regions colour-mapped by ``stage_data[key]``.

        Returns the ``PatchCollection`` (for attaching a colorbar). Axis
        limits/aspect are left to the caller.
        """
        centers = stage_data["region_centers"]
        size = stage_data["region_size"]
        values = stage_data[key]
        patches = [
            _RegPoly((xy[0], xy[1]), 6, radius=size, orientation=0)
            for xy in centers.values()
        ]
        pc = _PatchColl(patches, cmap=cmap, edgecolor="white", linewidth=linewidth)
        pc.set_array(np.array([values[rid] for rid in centers]))
        if vmin is not None or vmax is not None:
            pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        return pc


    def _stage_label(ax, stage):
        ax.text(
            0.5, -0.02, stage, transform=ax.transAxes, ha="center", va="top",
            fontsize=10,
        )


    def _shared_spatial_lim(adata, stage_data, pad_frac=0.005):
        xy_cells = adata.obsm["spatial"]
        centers = np.array(list(stage_data["region_centers"].values()))
        size = stage_data["region_size"]
        lo = np.minimum(xy_cells.min(0), centers.min(0) - size)
        hi = np.maximum(xy_cells.max(0), centers.max(0) + size)
        span = hi - lo
        lo = lo - pad_frac * span
        hi = hi + pad_frac * span
        return (lo[0], hi[0]), (lo[1], hi[1])


    def _stage_day(stage_name):
        return float(stage_name[1:])


    def _day_axis(ax, days, ylabel):
        ax.set_xlabel("Embryonic day", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(days)
        ax.set_xticklabels([f"{d:.1f}" for d in days])
        ax.legend(loc="best", fontsize=9, frameon=False)


    def _panel_trajectory(ax, summaries):
        stages = list(summaries.keys())
        days = np.array([_stage_day(s) for s in stages])
        am = np.array([summaries[s]["alpha_meta"] for s in stages])
        gm = np.array([summaries[s]["gamma_meta"] for s in stages])

        c_alpha = "#b71657"  # ~rocket(0.45)
        c_gamma = "#3488a6"  # ~mako(0.55)

        ax.plot(
            days,
            am,
            "o-",
            color=c_alpha,
            lw=1.6,
            ms=5,
            label=r"Mean local diversity",
        )
        ax.plot(
            days,
            gm,
            "s-",
            color=c_gamma,
            lw=1.6,
            ms=5,
            label=r"Mean global diversity",
        )

        _day_axis(ax, days, r"Diversity")


    def _panel_beta(ax, summaries):
        stages = list(summaries.keys())
        days = np.array([_stage_day(s) for s in stages])
        alphas = np.array([summaries[s]["alpha_meta"] for s in stages])
        gammas = np.array([summaries[s]["gamma_meta"] for s in stages])
        betas = gammas / alphas

        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.plot(
            days,
            betas,
            "o-",
            color="black",
            linewidth=1.6,
            ms=6,
            label="Mean global/local ratio",
        )

        _day_axis(ax, days, r"Ratio")


    def _finish_slot(ax, xlim, ylim, stage, *, label_stage):
        """Shared per-slide axis setup for the panel-D grids: fix the limits,
        equal aspect, hide the axis, flip y to image orientation, then add the
        scale bar (labelled only on the first slide) and the stage caption."""
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.invert_yaxis()
        add_scale_bar(ax, length_um=1000, label=None if stage == label_stage else "")
        _stage_label(ax, stage)


    def _panel_D(
        fig,
        grid_slot,
        stage_adatas,
        summaries,
        *,
        n_top=4,
        pad=1.02,
        cmap="viridis",
    ):
        """2x4 annotation grid on the left, 2x4 gamma grid on the right."""
        grid_stages = STAGES

        x_centers, y_centers, x_spans, y_spans = {}, {}, {}, {}
        for s in summaries:
            xy = stage_adatas[s].obsm["spatial"][:]
            x_centers[s] = (float(xy[:, 0].max()) + float(xy[:, 0].min())) / 2
            y_centers[s] = (float(xy[:, 1].max()) + float(xy[:, 1].min())) / 2
            x_spans[s] = float(xy[:, 0].max() - xy[:, 0].min())
            y_spans[s] = float(xy[:, 1].max() - xy[:, 1].min())

        def _slot_width(c):
            return max(x_spans[grid_stages[c + r * n_top]] for r in range(2))

        def _slot_height(r):
            return max(y_spans[grid_stages[c + r * n_top]] for c in range(n_top))

        col_widths = [_slot_width(c) for c in range(n_top)]
        row_heights = [_slot_height(r) for r in range(2)]

        outer = grid_slot.subgridspec(
            1,
            3,
            width_ratios=[1, 1, 0.04],
            wspace=0.08,
        )
        ann_sub = outer[0, 0].subgridspec(
            2,
            n_top,
            wspace=0.04,
            hspace=0.1,
            width_ratios=col_widths,
            height_ratios=row_heights,
        )
        gamma_sub = outer[0, 1].subgridspec(
            2,
            n_top,
            wspace=0.04,
            hspace=0.1,
            width_ratios=col_widths,
            height_ratios=row_heights,
        )
        _cb_holder = outer[0, 2].subgridspec(3, 1, height_ratios=[1, 6, 1])
        cb_ax = fig.add_subplot(_cb_holder[1])

        all_g = np.concatenate(
            [list(summaries[s]["gamma"].values()) for s in summaries]
        )
        vmin = float(np.percentile(all_g, 5))
        vmax = float(np.percentile(all_g, 95))

        _label_stage = next(iter(summaries))

        ann_axes, gamma_axes = [], []
        pc = None
        for idx, s in enumerate(grid_stages):
            r, c = idx // n_top, idx % n_top
            ax_ann = fig.add_subplot(ann_sub[r, c])
            ax_g = fig.add_subplot(gamma_sub[r, c])
            ann_axes.append(ax_ann)
            gamma_axes.append(ax_g)

            xlim = (
                x_centers[s] - col_widths[c] / 2 * pad,
                x_centers[s] + col_widths[c] / 2 * pad,
            )
            ylim = (
                y_centers[s] - row_heights[r] / 2 * pad,
                y_centers[s] + row_heights[r] / 2 * pad,
            )

            # Annotation panel (left grid)
            _draw_annotation(ax_ann, stage_adatas[s], s=0.15)
            _finish_slot(ax_ann, xlim, ylim, s, label_stage=_label_stage)

            # Global-diversity (gamma) panel (right grid)
            pc = _draw_hex(
                ax_g,
                summaries[s],
                "gamma",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0.1,
            )
            _finish_slot(ax_g, xlim, ylim, s, label_stage=_label_stage)

        return ann_axes, gamma_axes, pc, cb_ax


    def make_publication_figure():
        _mm = 1 / 25.4
        fig = plt.figure(figsize=(179 * _mm, 193 * _mm))
        grid = fig.add_gridspec(
            nrows=4,
            ncols=1,
            height_ratios=[2.6, 0.34, 2.5, 1.05],
            hspace=0.2,
            left=0.07,
            right=0.97,
            bottom=0.05,
            top=0.97,
        )

        # Row 0: case study panels A, B, C
        row0 = grid[0].subgridspec(1, 3, wspace=0.05)
        ax_ann = fig.add_subplot(row0[0])
        ax_alpha = fig.add_subplot(row0[1])
        ax_gamma = fig.add_subplot(row0[2])

        # Heatmap colormaps matched to panel E (local=teal, global=gold).
        from _figutils import global_cmap, local_cmap

        _cmap_local = local_cmap()
        _cmap_global = global_cmap()

        handles, labels = _draw_annotation(ax_ann, pub_adata, s=0.4)
        pc_a = _draw_hex(
            ax_alpha, stage_summaries[pub_stage], "alpha", cmap=_cmap_local
        )
        pc_g = _draw_hex(
            ax_gamma, stage_summaries[pub_stage], "gamma", cmap=_cmap_global
        )

        xlim, ylim = _shared_spatial_lim(pub_adata, stage_summaries[pub_stage])
        for ax in (ax_ann, ax_alpha, ax_gamma):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.invert_yaxis()

        add_scale_bar(ax_ann)

        add_scale_bar(ax_alpha)
        cb_a = fig.colorbar(
            pc_a,
            ax=ax_alpha,
            location="right",
            fraction=0.05,
            pad=0.02,
            shrink=0.75,
            format="%.2f",
        )
        cb_a.ax.tick_params(labelsize=9)
        cb_a.ax.set_title("Local\ndiversity", fontsize=10)

        add_scale_bar(ax_gamma)
        cb_g = fig.colorbar(
            pc_g,
            ax=ax_gamma,
            location="right",
            fraction=0.05,
            pad=0.02,
            shrink=0.75,
        )
        cb_g.ax.tick_params(labelsize=9)
        cb_g.ax.set_title("Global\ndiversity", fontsize=10)

        # Lay row 0 out by hand: all three maps at identical scale (B's resolved
        # equal-aspect box), A left-aligned to the figure margin so it lines up
        # with D below. Gaps are equal, but B/C each reserve a colorbar slot
        # while A does not, so A sits closer to B than B does to C. Draw first
        # so the equal-aspect boxes and colorbar widths are resolved.
        fig.canvas.draw()
        _L, _R = 0.07, 0.97
        _pb = ax_alpha.get_position()
        _ew, _eh, _ey = _pb.width, _pb.height, _pb.y0
        _cba = cb_a.ax.get_position()
        _cbw = _cba.width
        _cb_pad = 0.006  # map -> bar gap
        _cb_lab = 0.045  # room for the bar's tick labels
        _slot = _cb_pad + _cbw + _cb_lab
        _gap = (_R - _L - (3 * _ew + 2 * _slot)) / 2

        ax_ann.set_position([_L, _ey, _ew, _eh])
        _bx = _L + _ew + _gap
        ax_alpha.set_position([_bx, _ey, _ew, _eh])
        cb_a.ax.set_position([_bx + _ew + _cb_pad, _cba.y0, _cbw, _cba.height])
        _cx = _bx + _ew + _slot + _gap
        ax_gamma.set_position([_cx, _ey, _ew, _eh])
        _cbg = cb_g.ax.get_position()
        cb_g.ax.set_position(
            [_cx + _ew + _cb_pad, _cbg.y0, _cbg.width, _cbg.height]
        )

        # Row 1: tissue legend
        ax_legend = fig.add_subplot(grid[1])
        ax_legend.set_axis_off()
        ax_legend.legend(
            handles,
            labels,
            ncol=5,
            loc="center",
            markerscale=12,
            frameon=False,
            handlelength=0.5,
            handletextpad=0.4,
            columnspacing=1.0,
            fontsize=8,
        )
        # Nudge the legend up a hair to leave a touch more breathing room
        # between it and panel D below.
        _lp = ax_legend.get_position()
        ax_legend.set_position([_lp.x0, _lp.y0 + 0.006, _lp.width, _lp.height])

        # Row 2: panel D (2x4 annotation on left, 2x4 gamma on right + cbar)
        ann_axes, gamma_axes, pc_grid, cb_ax = _panel_D(
            fig, grid[2], stage_adatas, stage_summaries, cmap=_cmap_global
        )
        # Pull panel D up to close the gap to A/B/C without touching E/F.
        # In figure-coord units; tune to taste.
        _d_shift_up = 0.02
        for _ax in ann_axes + gamma_axes + [cb_ax]:
            _pos = _ax.get_position()
            _ax.set_position(
                [_pos.x0, _pos.y0 + _d_shift_up, _pos.width, _pos.height]
            )
        cb_grid = fig.colorbar(pc_grid, cax=cb_ax, orientation="vertical")
        cb_grid.ax.tick_params(labelsize=9)
        cb_grid.ax.set_title("Global\ndiversity", fontsize=10)

        # Row 3: trajectory + beta. Inset panel E from the left (keep its right
        # edge in place) so y-axis tick labels and ylabel sit inside the figure
        # border instead of hanging off the left edge.
        row3 = grid[3].subgridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.3)
        ax_traj = fig.add_subplot(row3[0])
        ax_beta = fig.add_subplot(row3[1])

        _e_inset = 0.08
        _pos_e = ax_traj.get_position()
        ax_traj.set_position(
            [
                _pos_e.x0 + _e_inset,
                _pos_e.y0 + _d_shift_up,
                _pos_e.width - _e_inset,
                _pos_e.height,
            ]
        )
        _pos_f = ax_beta.get_position()
        ax_beta.set_position(
            [_pos_f.x0, _pos_f.y0 + _d_shift_up, _pos_f.width, _pos_f.height]
        )

        _panel_trajectory(ax_traj, stage_summaries)
        _panel_beta(ax_beta, stage_summaries)

        # Place panel letters in figure coordinates so A/D/E share an x
        # and ABC / EF each share a y, regardless of subpanel widths.
        _pad_x = 0.005

        def _letter_y(*axes):
            # Offset proportional to row height so rows with shorter
            # panels (like D) don't end up with the label hovering high.
            return max(ax.get_position().y1 for ax in axes) + 0.008 * max(
                ax.get_position().height for ax in axes
            )

        _col_x = (
            min(
                ax_ann.get_position().x0,
                ann_axes[0].get_position().x0,
                ax_traj.get_position().x0,
            )
            - _pad_x
        )
        _y_abc = _letter_y(ax_ann, ax_alpha, ax_gamma)
        _y_d = ann_axes[0].get_position().y1 - 0.026
        _y_ef = _letter_y(ax_traj, ax_beta)
        _letter_kw = dict(fontsize=14, fontweight="bold", va="bottom", ha="left")
        fig.text(_col_x, _y_abc, "A", **_letter_kw)
        fig.text(ax_alpha.get_position().x0 - _pad_x, _y_abc, "B", **_letter_kw)
        fig.text(ax_gamma.get_position().x0 - _pad_x, _y_abc, "C", **_letter_kw)
        fig.text(_col_x, _y_d, "D", **_letter_kw)
        fig.text(_col_x, _y_ef, "E", **_letter_kw)
        fig.text(ax_beta.get_position().x0 - 0.08, _y_ef, "F", **_letter_kw)

        return fig


    _pub_fig = make_publication_figure()
    plt.savefig("./figures/fig3.pdf", bbox_inches="tight")
    _pub_fig
    return


if __name__ == "__main__":
    app.run()
