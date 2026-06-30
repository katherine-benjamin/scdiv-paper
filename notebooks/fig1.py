import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import ConnectionPatch, FancyArrowPatch
    from scipy.spatial.distance import pdist, squareform

    from scdiv.diversity import diversity

    from _figutils import hill_number

    return (
        ConnectionPatch,
        FancyArrowPatch,
        diversity,
        hill_number,
        mo,
        np,
        pdist,
        plt,
        sns,
        squareform,
    )


@app.cell
def _(diversity, np, pdist, squareform):
    def make_blobs(centers, sizes, spreads, seed):
        rng = np.random.default_rng(seed)
        chunks = [
            rng.normal(loc=center, scale=spread, size=(n, 2))
            for center, n, spread in zip(centers, sizes, spreads)
        ]
        return np.vstack(chunks)

    def celltype_diversity_q2(coords, labels, sigma):
        # Cell-type mode in fake-UMAP space: aggregate to per-label centroids,
        # build similarity from Euclidean distances between centroids, weight
        # by label proportions. Analogous to scdiv.similarity.cell_type_similarity
        # but with Euclidean kernel instead of cosine.
        unique, counts = np.unique(labels, return_counts=True)
        n = len(coords)
        centroids = np.array([coords[labels == u].mean(axis=0) for u in unique])
        proportions = counts / n
        if len(centroids) == 1:
            return diversity(np.ones((1, 1)), order=2.0, distribution=proportions)
        d = squareform(pdist(centroids))
        s = np.exp(-d / sigma)
        return diversity(s, order=2.0, distribution=proportions)

    def umap_arrow_axes(ax, lw=0.8):
        # Schematic-style scatter axes: hide spines and draw arrows along
        # the bottom and left of the data limits.
        for spine in ax.spines.values():
            spine.set_visible(False)
        arrowprops = dict(
            arrowstyle="->", color="black", lw=lw,
            shrinkA=0, shrinkB=0,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.annotate(
            "", xy=(xlim[1], ylim[0]), xytext=(xlim[0], ylim[0]),
            arrowprops=arrowprops, zorder=5,
        )
        ax.annotate(
            "", xy=(xlim[0], ylim[1]), xytext=(xlim[0], ylim[0]),
            arrowprops=arrowprops, zorder=5,
        )

    return celltype_diversity_q2, make_blobs, umap_arrow_axes


@app.cell
def _():
    # length scale for the exponential kernel s = exp(-d / sigma).
    sigma = 2.0
    return (sigma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 1: method schematic
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panel A

    Here we build a 4-cluster toy data set.
    """)
    return


@app.cell
def _(diversity, hill_number, make_blobs, np, pdist, sigma, squareform):
    # Process: toy 4-cluster dataset, its composition, the centroid similarity
    # matrix, and the two diversities. The exponential kernel uses the
    # project-wide `sigma`, giving heterogeneous off-diagonal similarities
    # (~0.1 to 0.7) so the matrix in the panel is legible.
    _centers = [(1.5, 7.0), (7.5, 7.9), (8.4, 6.8), (3.6, 1.4)]
    _sizes = [6, 4, 3, 5]
    _spreads = [0.65] * 4
    c_coords = np.clip(make_blobs(_centers, _sizes, _spreads, seed=60), 0.0, 10.0)
    c_labels = np.concatenate([np.full(n, i) for i, n in enumerate(_sizes)])

    c_unique, _counts = np.unique(c_labels, return_counts=True)
    _centroids = np.array([c_coords[c_labels == u].mean(axis=0) for u in c_unique])
    c_proportions = _counts / len(c_coords)
    c_sim = np.exp(-squareform(pdist(_centroids)) / sigma)
    c_div = diversity(c_sim, order=2.0, distribution=c_proportions)
    c_hill = hill_number(_counts)

    {"LC diversity": round(float(c_div), 3), "Hill number": round(float(c_hill), 3)}
    return c_coords, c_div, c_hill, c_labels, c_proportions, c_sim, c_unique


@app.cell
def _(
    c_coords,
    c_div,
    c_hill,
    c_labels,
    c_proportions,
    c_sim,
    c_unique,
    plt,
    sns,
):
    def render_fig1a(fig):
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.transforms import blended_transform_factory

        coords, labels, unique = c_coords, c_labels, c_unique
        proportions, sim, div, hill = c_proportions, c_sim, c_div, c_hill

        palette = sns.color_palette("colorblind")
        bar_colors = [palette[u] for u in unique]

        gs = fig.add_gridspec(
            2,
            3,
            width_ratios=[1, 1, 1],
            height_ratios=[1, 1.4],
            wspace=0.7,
            hspace=0.55,
        )
        ax_umap = fig.add_subplot(gs[:, 0])
        ax_abund = fig.add_subplot(gs[0, 1])
        ax_heat = fig.add_subplot(gs[1, 1])
        ax_hill = fig.add_subplot(gs[0, 2])
        ax_lc = fig.add_subplot(gs[1, 2])

        # Single-cell embedding (toy UMAP). Circles only — colour distinguishes
        # cell types throughout the figure.
        for u in unique:
            m = labels == u
            ax_umap.scatter(
                coords[m, 0],
                coords[m, 1],
                s=25,
                color=palette[u],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        ax_umap.set_xlim(0, 10)
        ax_umap.set_ylim(0, 10)
        ax_umap.set_aspect("equal")
        ax_umap.set_xticks([])
        ax_umap.set_yticks([])
        for spine in ax_umap.spines.values():
            spine.set_visible(False)
        # Axes drawn as arrows from the origin; shrinkA/B=0 so tails meet
        # exactly at (0,0) for a clean L corner.
        axis_arrowprops = dict(
            arrowstyle="->",
            color="black",
            lw=1.0,
            shrinkA=0,
            shrinkB=0,
        )
        ax_umap.annotate(
            "",
            xy=(10, 0),
            xytext=(0, 0),
            arrowprops=axis_arrowprops,
            zorder=2,
        )
        ax_umap.annotate(
            "",
            xy=(0, 10),
            xytext=(0, 0),
            arrowprops=axis_arrowprops,
            zorder=2,
        )
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=palette[u],
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=5,
                linestyle="None",
                label=f"Cell type {u + 1}",
            )
            for u in unique
        ]
        ax_umap.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            frameon=False,
            fontsize=6,
            handletextpad=0.4,
            columnspacing=1.0,
        )

        # Cell-type composition bars (no x-tick labels — colour identifies type).
        ax_abund.bar(
            range(len(unique)),
            proportions,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        ax_abund.set_ylabel("Relative\nabundance", fontsize=7)
        ax_abund.set_ylim(0, 0.4)
        ax_abund.set_yticks([])
        ax_abund.set_xticks([])
        ax_abund.spines["top"].set_visible(False)
        ax_abund.spines["right"].set_visible(False)
        ax_abund.spines["left"].set_linewidth(0.8)
        ax_abund.spines["bottom"].set_linewidth(0.8)

        # Similarity heatmap: dark cells where types are similar (diagonal at
        # 1.0), white where dissimilar. Colored dots as tick labels link
        # rows/cols to cell-type colours from the scatter.
        blue_scale = LinearSegmentedColormap.from_list(
            "white_to_darkblue",
            ["#ffffff", "#08306b"],
        )
        vmin, vmax = float(sim.min()), 1.0
        text_threshold = 0.5 * (vmin + vmax)
        im = ax_heat.imshow(sim, vmin=vmin, vmax=vmax, cmap=blue_scale)
        ax_heat.set_xticks(range(len(unique)))
        ax_heat.set_yticks(range(len(unique)))
        ax_heat.set_xticklabels([])
        ax_heat.set_yticklabels([])
        ax_heat.xaxis.set_label_position("top")
        ax_heat.tick_params(axis="both", which="major", length=0)
        # Colored circle markers as row/col labels (matching the embedding
        # scatter style). One coordinate is in data space (so markers stay
        # aligned with cell centers) and the other in axes-fraction space
        # (so the offset outside the matrix is fixed and doesn't expand the
        # imshow's data limits the way pure-data scatter does).
        trans_top = blended_transform_factory(
            ax_heat.transData,
            ax_heat.transAxes,
        )
        trans_left = blended_transform_factory(
            ax_heat.transAxes,
            ax_heat.transData,
        )
        label_pad = 0.12  # axes-fraction units outside the matrix edge
        for i, c in enumerate(bar_colors):
            ax_heat.scatter(
                i,
                1 + label_pad,
                transform=trans_top,
                s=25,
                color=c,
                edgecolor="black",
                linewidth=0.5,
                clip_on=False,
                zorder=5,
            )
            ax_heat.scatter(
                -label_pad,
                i,
                transform=trans_left,
                s=25,
                color=c,
                edgecolor="black",
                linewidth=0.5,
                clip_on=False,
                zorder=5,
            )
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                v = sim[i, j]
                tc = "white" if v >= text_threshold else "black"
                ax_heat.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color=tc,
                    fontsize=4.5,
                )
        # x from ax_abund (gridspec column center, unshrunk by the colorbar)
        # so the label stays vertically aligned with "Cell-type composition";
        # y from ax_heat so it sits above the matrix.
        ax_heat.text(
            0.5,
            1.22,
            "Similarity matrix",
            transform=blended_transform_factory(
                ax_abund.transAxes,
                ax_heat.transAxes,
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.04)
        cbar.set_ticks([vmin + 0.03, vmax - 0.03])
        cbar.set_ticklabels(["Totally\ndissimilar", "Totally\nsimilar"])
        cbar.ax.tick_params(labelsize=5.5, length=0)

        # Right column: Hill (composition only) on top, LC (composition +
        # similarity) on bottom. The arrows below carry the visual cue —
        # nothing flows from the heatmap into the Hill panel.
        for ax in (ax_hill, ax_lc):
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        ax_hill.text(
            0.5,
            0.72,
            "Hill number",
            transform=ax_hill.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
        )
        ax_hill.text(
            0.5,
            0.28,
            f"$D_H = {hill:.3f}$",
            transform=ax_hill.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#444444",
        )
        ax_lc.text(
            0.5,
            0.72,
            "LC diversity",
            transform=ax_lc.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
        )
        ax_lc.text(
            0.5,
            0.28,
            f"$D_\\mathrm{{LC}} = {div:.3f}$",
            transform=ax_lc.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#08306b",
        )

        for ax, label in [
            (ax_umap, "Gene expression"),
            (ax_abund, "Cell-type composition"),
            (ax_hill, "Diversity"),
        ]:
            ax.set_title(label, fontsize=8, fontweight="bold", pad=6)


    _w_in = 119 / 25.4
    _h_in = 60 / 25.4
    _fig = plt.figure(figsize=(_w_in, _h_in), dpi=300)
    render_fig1a(_fig)
    _fig.savefig("./figures/fig1a.svg", bbox_inches="tight")
    _fig.savefig(
        "./figures/fig1a.png",
        dpi=300,
        bbox_inches="tight",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panel B

    Here we repeatedly subdivide a cell type and compare LC diversity to the Hill number
    """)
    return


@app.cell
def _(celltype_diversity_q2, hill_number, np, sigma):
    # Three cell types. Sweep through k = number of subclusters of C, splitting C
    # with k-means; for each k record LC diversity (which sees the splits are
    # similar) and the Hill number (which counts every split as a new type).
    from scipy.cluster.vq import kmeans2

    _rng = np.random.default_rng(7)
    _n_A, _n_B, _n_C = 30, 30, 70
    _A = _rng.normal(loc=(-3, 2), scale=0.5, size=(_n_A, 2))
    _B = _rng.normal(loc=(3, 2), scale=0.5, size=(_n_B, 2))
    _C_x = np.linspace(-2.0, 2.0, _n_C) + _rng.normal(0, 0.15, _n_C)
    _C_y = -2 + _rng.normal(0, 0.3, _n_C)
    _C = np.column_stack([_C_x, _C_y])

    b_coords = np.vstack([_A, _B, _C])
    _base_labels = np.concatenate(
        [
            np.zeros(_n_A, dtype=int),
            np.ones(_n_B, dtype=int),
            np.full(_n_C, 2, dtype=int),
        ]
    )

    b_K = 5
    b_k_values = np.arange(1, b_K + 1)
    _scdiv_vals = []
    _hill_vals = []
    b_labels_at_k = {}
    for _k in b_k_values:
        if _k == 1:
            _C_sub = np.zeros(_n_C, dtype=int)
        else:
            _, _C_sub = kmeans2(_C, _k, seed=42, minit="++")
        _new_labels = _base_labels.copy()
        _new_labels[_base_labels == 2] = 2 + _C_sub
        _scdiv_vals.append(celltype_diversity_q2(b_coords, _new_labels, sigma))
        _, _hc = np.unique(_new_labels, return_counts=True)
        _hill_vals.append(hill_number(_hc))
        b_labels_at_k[int(_k)] = _new_labels
    b_scdiv_vals = np.array(_scdiv_vals)
    b_hill_vals = np.array(_hill_vals)

    [
        {"k": int(_k), "LC": round(float(_s), 3), "Hill": round(float(_h), 3)}
        for _k, _s, _h in zip(b_k_values, b_scdiv_vals, b_hill_vals)
    ]
    return b_K, b_coords, b_hill_vals, b_k_values, b_labels_at_k, b_scdiv_vals


@app.cell
def _(
    FancyArrowPatch,
    b_K,
    b_coords,
    b_hill_vals,
    b_k_values,
    b_labels_at_k,
    b_scdiv_vals,
    np,
    plt,
    sns,
    umap_arrow_axes,
):
    def render_fig1b(fig1b):
        K = b_K
        k_values = b_k_values
        scdiv_vals = b_scdiv_vals
        hill_vals = b_hill_vals
        coords = b_coords
        labels_at_k = b_labels_at_k

        fig_w_in = fig1b.bbox.width / fig1b.dpi
        fig_h_in = fig1b.bbox.height / fig1b.dpi

        # Chart on top, panels below.
        margin_left = 0.16
        margin_right = 0.04
        chart_bottom = 0.52
        chart_height = 0.43
        ax_b = fig1b.add_axes(
            [
                margin_left,
                chart_bottom,
                1 - margin_left - margin_right,
                chart_height,
            ]
        )

        entropy_color = "#c53030"
        ours_color = "black"
        ax_b.plot(
            k_values,
            hill_vals,
            "s-",
            color=entropy_color,
            lw=1.0,
            ms=3.0,
        )
        ax_b.plot(
            k_values,
            scdiv_vals,
            "o-",
            color=ours_color,
            lw=1.6,
            ms=3.5,
        )

        ax_b.set_xlabel("Number of subclusters of C", fontsize=8)
        ax_b.set_ylabel("Diversity", fontsize=8)
        ax_b.tick_params(axis="both", labelsize=7, width=0.8, length=3)
        ax_b.set_xticks(k_values)
        ax_b.set_xlim(0.5, K + 0.5)
        y_max = float(hill_vals.max()) + 0.5
        ax_b.set_ylim(1, y_max)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.spines["left"].set_linewidth(0.8)
        ax_b.spines["bottom"].set_linewidth(0.8)
        ax_b.grid(
            True,
            which="major",
            axis="both",
            color="#bbb",
            linewidth=0.4,
            alpha=0.6,
            zorder=0,
        )
        ax_b.set_axisbelow(True)

        # End-of-curve labels overflow past the right spine via clip_on=False.
        ax_b.text(
            K + 0.18,
            hill_vals[-1],
            "Hill number",
            color=entropy_color,
            fontsize=8,
            ha="left",
            va="center",
            clip_on=False,
        )
        ax_b.text(
            K + 0.18,
            scdiv_vals[-1],
            "LC Diversity",
            color=ours_color,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
            clip_on=False,
        )

        # Inset panels at the bottom of the figure.
        palette_b = sns.color_palette("colorblind")
        inset_w = 0.3
        # Square in physical units: width-fraction * fig_w == height-fraction * fig_h
        inset_h = inset_w * fig_w_in / fig_h_in
        inset_y = 0.03
        inset_positions = {
            1: 0.10,
            K: 1 - inset_w + 0.05,
        }
        cluster_label_positions = [
            ("A", (-3.0, 4.0)),
            ("B", (3.0, 4.0)),
            ("C", (0.0, -4.0)),
        ]

        for k_show, ix in inset_positions.items():
            ax_in = fig1b.add_axes([ix, inset_y, inset_w, inset_h])
            labels_b = labels_at_k[k_show]
            unique = np.unique(labels_b)
            n_subs = int((unique >= 2).sum())
            # Vivid, evenly-spaced hues so each subdivision reads as a
            # distinct call. Avoid red (Hill curve), blue (A) and orange (B).
            c_palette = [
                "#33a02c",  # green
                "#6a3d9a",  # purple
                "#b15928",  # brown
                "#e7298a",  # magenta
                "#17becf",  # cyan
            ][: max(n_subs, 1)]
            for u in unique:
                m = labels_b == u
                color = palette_b[u] if u < 2 else c_palette[u - 2]
                ax_in.scatter(
                    coords[m, 0],
                    coords[m, 1],
                    s=5,
                    color=color,
                    edgecolors="black",
                    linewidths=0.3,
                )
            for label, (lx, ly) in cluster_label_positions:
                ax_in.text(
                    lx,
                    ly,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            ax_in.set_xticks([])
            ax_in.set_yticks([])
            ax_in.set_xlim(-5, 5)
            ax_in.set_ylim(-5, 5)
            ax_in.set_aspect("equal")
            umap_arrow_axes(ax_in, lw=0.6)
            ax_in.text(
                0.96,
                0.04,
                f"k={k_show}",
                transform=ax_in.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
            )

        # Arrow from left panel to right panel with caption above.
        arrow_y = inset_y + inset_h / 2
        arrow_x_start = inset_positions[1] + inset_w + 0.015
        arrow_x_end = inset_positions[K] - 0.015
        arrow = FancyArrowPatch(
            (arrow_x_start, arrow_y),
            (arrow_x_end, arrow_y),
            transform=fig1b.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            color="black",
            linewidth=0.8,
        )
        fig1b.add_artist(arrow)
        fig1b.text(
            (arrow_x_start + arrow_x_end) / 2 - 0.01,
            arrow_y + 0.02,
            "Increased\nsubclustering",
            ha="center",
            va="bottom",
            fontsize=7,
            linespacing=1.0,
        )


    _w_in = 60 / 25.4
    _h_in = 60 / 25.4
    _fig1b = plt.figure(figsize=(_w_in, _h_in), dpi=300)
    render_fig1b(_fig1b)
    _fig1b.savefig("./figures/fig1b.svg", bbox_inches="tight")
    _fig1b.savefig("./figures/fig1b.png", dpi=300, bbox_inches="tight")
    _fig1b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panel C

    Five toy tissues, from a single type to many distinct types. Each is
    scored with the cell-type-mode LC diversity, and the panel lays
    them out along that diversity axis.
    """)
    return


@app.cell
def _(make_blobs, np):
    def _():
        # Hand-picked "random-looking" centers, chosen to avoid obvious geometric
        # symmetry while keeping pairwise distances comparable. Index 0 and 1 are
        # the closest pair in random_far so the merge-mistake variant in the
        # cell-type comparison merges naturally-adjacent blobs.
        random_far = [
            (1.0, 6.5),
            (7.0, 4.0),
            (-7.0, 1.0),
            (-2.5, -7.0),
            (5.0, -5.0),
        ]
        random_close = [
            (0.0, 0.34),
            (3.06, 2.55),
            (-2.72, 2.04),
            (-2.38, -2.55),
            (2.89, -2.21),
        ]

        scenarios = {
            "one type": dict(
                centers=[(0, 0)],
                sizes=[30],
                spreads=[0.6],
                seed=0,
            ),
            "3 close, 1 far": dict(
                centers=[
                    (-3.0, 1.10),
                    (-3.96, -0.55),
                    (-2.04, -0.55),
                    (5.0, 0.5),
                ],
                sizes=[8, 8, 8, 8],
                spreads=[0.3, 0.3, 0.3, 0.3],
                seed=5,
            ),
            "many similar": dict(
                centers=random_close,
                sizes=[10] * 5,
                spreads=[1.0] * 5,
                seed=2,
            ),
            "one dominates": dict(
                centers=random_far,
                sizes=[18, 2, 2, 2, 2],
                spreads=[0.75, 0.2, 0.2, 0.2, 0.2],
                seed=3,
            ),
            "many distinct": dict(
                centers=random_far,
                sizes=[6] * 5,
                spreads=[0.3] * 5,
                seed=4,
            ),
        }

        coords_by_name = {}
        labels_by_name = {}
        for name, s in scenarios.items():
            coords_by_name[name] = make_blobs(
                s["centers"], s["sizes"], s["spreads"], s["seed"],
            )
            labels_by_name[name] = np.concatenate(
                [np.full(size, i) for i, size in enumerate(s["sizes"])]
            )
        return coords_by_name, labels_by_name


    coords_by_name, labels_by_name = _()
    return coords_by_name, labels_by_name


@app.cell
def _(celltype_diversity_q2, coords_by_name, labels_by_name, sigma):
    diversities = {
        name: celltype_diversity_q2(coords, labels_by_name[name], sigma)
        for name, coords in coords_by_name.items()
    }
    diversities
    return (diversities,)


@app.cell
def _(
    ConnectionPatch,
    coords_by_name,
    diversities,
    labels_by_name,
    np,
    plt,
    sns,
    umap_arrow_axes,
):
    def render_fig1c(fig1a):
        from matplotlib.lines import Line2D

        order = sorted(diversities, key=diversities.get)
        n_panels = len(order)
        palette_a = sns.color_palette("colorblind")

        div_vals = np.array([diversities[name] for name in order])
        d_min, d_max = div_vals.min(), div_vals.max()
        pad = 0.18 * (d_max - d_min)
        axis_lo, axis_hi = 1.0, 5 + pad
        left_pad = 0.5

        margin_left = 0.04
        margin_right = 0.16
        span = 1 - margin_left - margin_right
        ax_axis_a = fig1a.add_axes([margin_left, 0.22, span, 0.04])
        ax_axis_a.set_xlim(axis_lo - left_pad, axis_hi)
        ax_axis_a.set_ylim(-1, 1)
        ax_axis_a.spines[["top", "left", "right", "bottom"]].set_visible(False)
        ax_axis_a.set_yticks([])
        ax_axis_a.tick_params(
            axis="x", bottom=True, labelsize=7, width=0.8, length=3
        )
        ax_axis_a.set_xlabel("LC diversity", fontsize=8)

        ax_axis_a.annotate(
            "",
            xy=(axis_hi, 0),
            xytext=(axis_lo, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=8,
            ),
        )
        ax_axis_a.plot(
            [axis_lo - left_pad, axis_lo],
            [0, 0],
            color="black",
            lw=0.8,
            linestyle=(0, (3, 2)),
            zorder=1,
            clip_on=False,
        )
        ax_axis_a.plot(
            [axis_lo, axis_lo],
            [-0.4, 0.4],
            color="black",
            lw=0.8,
            zorder=2,
            clip_on=False,
        )

        for name in order:
            d = diversities[name]
            ax_axis_a.plot(d, 0, "o", color="black", markersize=2, zorder=3)

        panel_w = 0.13
        panel_h = 0.65
        panel_y = 0.20
        roman = ["(i)", "(ii)", "(iii)", "(iv)", "(v)"]
        for i, name in enumerate(order):
            x_center = margin_left + span * (i + 0.5) / n_panels
            ax_a = fig1a.add_axes(
                [x_center - panel_w / 2, panel_y, panel_w, panel_h]
            )
            coords_a = coords_by_name[name]
            labels_a = labels_by_name[name]
            for u_a in np.unique(labels_a):
                m_a = labels_a == u_a
                ax_a.scatter(
                    coords_a[m_a, 0],
                    coords_a[m_a, 1],
                    s=18,
                    color=palette_a[int(u_a)],
                    edgecolors="black",
                    linewidths=0.3,
                )
            ax_a.set_xticks([])
            ax_a.set_yticks([])
            ax_a.set_xlim(-10, 10)
            ax_a.set_ylim(-10, 10)
            ax_a.set_aspect("equal")
            umap_arrow_axes(ax_a, lw=0.8)
            ax_a.text(
                0.06,
                0.94,
                roman[i],
                transform=ax_a.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )

            d = diversities[name]
            con_a = ConnectionPatch(
                xyA=(0.5, 0.0),
                coordsA=ax_a.transAxes,
                xyB=(d, 0.0),
                coordsB=ax_axis_a.transData,
                color="#555",
                linewidth=0.6,
                linestyle=(0, (3, 3)),
            )
            # Attach to an axes rather than the figure: figure-level
            # ConnectionPatch artists render with stray strokes inside a
            # SubFigure parent.
            ax_a.add_artist(con_a)

        n_types = 5
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=palette_a[t],
                markeredgecolor="black",
                markeredgewidth=0.3,
                markersize=5,
                label=f"Cell type {t + 1}",
            )
            for t in range(n_types)
        ]
        fig1a.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(margin_left + span - 0.015, 0.525),
            frameon=False,
            fontsize=7,
            handletextpad=0.5,
            labelspacing=0.7,
        )


    _w_in = 178 / 25.4
    _h_in = _w_in * 5.2 / 15
    _fig1c = plt.figure(figsize=(_w_in, _h_in), dpi=300)
    render_fig1c(_fig1c)
    _fig1c.savefig("./figures/fig1c.svg")
    _fig1c.savefig("./figures/fig1c.png", dpi=300)
    _fig1c
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Panel D

    A synthetic circular tissue. The background is a patchwork of
    single-common-type domains; four planted regions of interest occupy the
    (low / high local) by (low / high global) quadrants. Hex-partitioning
    plus the partition diversity then recover, per region, local (alpha) and
    global (gamma) diversity.
    """)
    return


@app.cell
def _(np):
    # Process: a synthetic circular tissue for the local / global (Reeve) split.
    # The background is a patchwork of single-common-type domains (each hex
    # homogeneous, so low local alpha; every common type globally abundant, so
    # low global gamma). Four ROIs, one per (alpha, gamma) quadrant, are planted
    # in a 2x2 layout. We hex-partition, then compute per-region alpha and gamma.
    import anndata as ad
    import scdiv
    from scdiv.diversity import partition_diversity

    d_W, d_H = 150.0, 150.0  # square slide -> circular tissue holding a 2x2
    _CX, _CY = d_W / 2, d_H / 2
    _R_TISSUE = 0.97 * d_W / 2
    d_RR = 20.0
    # Cells on a jittered lattice (not uniform-random) so big markers tile
    # cleanly without piling up -- a schematic "cellular" look.
    _CELL_PITCH, _REGION_SIZE, _MIN_CELLS = 2.2, 8.0, 12
    d_NC, d_NR = 4, 4  # commons 0..3, rares 4..7
    d_K = d_NC + d_NR

    # ROI -> (centre, label). Spatial 2x2: x=local (low|high), y=global.
    d_ROI = {
        "lowA_lowG": ((45.0, 105.0), "Single\ncommon type"),
        "highA_lowG": ((105.0, 105.0), "Common\nmixture"),
        "lowA_highG": ((45.0, 45.0), "Single\nrare type"),
        "highA_highG": ((105.0, 45.0), "Rare\nmixture"),
    }

    _rng = np.random.default_rng(7)
    _gx, _gy = np.meshgrid(
        np.arange(_CELL_PITCH / 2, d_W, _CELL_PITCH),
        np.arange(_CELL_PITCH / 2, d_H, _CELL_PITCH),
    )
    d_xy = np.column_stack([_gx.ravel(), _gy.ravel()]).astype(float)
    d_xy += _rng.uniform(-0.32 * _CELL_PITCH, 0.32 * _CELL_PITCH, d_xy.shape)
    # Organic common-type domains: smooth Gaussian influence fields (one per
    # common type), argmax-assigned, with mild jitter so the borders feather
    # into curves rather than a hard Voronoi grid.
    _tseeds = np.array([(38, 112), (116, 118), (40, 36), (118, 44)], float)
    _sigma_t = 47.0
    _d2t = np.linalg.norm(d_xy[:, None, :] - _tseeds[None, :, :], axis=2)
    _infl = np.exp(-(_d2t**2) / (2 * _sigma_t**2))
    _infl *= _rng.uniform(0.85, 1.15, d_NC)
    _infl += _rng.normal(0, 0.05, _infl.shape)
    d_codes = np.argmax(_infl, axis=1)

    def _roi_mask(name):
        return np.linalg.norm(d_xy - np.array(d_ROI[name][0]), axis=1) < d_RR

    d_codes[_roi_mask("lowA_lowG")] = 0  # homog common
    _m = _roi_mask("highA_lowG")
    d_codes[_m] = _rng.integers(0, d_NC, _m.sum())  # mixed common
    d_codes[_roi_mask("lowA_highG")] = d_NC  # homog rare
    _m = _roi_mask("highA_highG")
    d_codes[_m] = _rng.integers(d_NC, d_K, _m.sum())  # mixed rare

    _keep = (d_xy[:, 0] - _CX) ** 2 + (d_xy[:, 1] - _CY) ** 2 <= _R_TISSUE**2
    d_xy, d_codes = d_xy[_keep], d_codes[_keep]

    d_adata = ad.AnnData(
        X=np.zeros((len(d_xy), 1), dtype=np.float32),
        obs={"cell_type": d_codes.astype(str)},
        obsm={"spatial": d_xy},
    )
    scdiv.spatial.partition(
        d_adata,
        method="hex",
        region_size=_REGION_SIZE,
        min_cells=_MIN_CELLS,
        min_density=0.0,
    )
    _regions = d_adata.obs["spatial_region"].to_numpy()
    _centers = d_adata.uns["spatial_region_params"]["region_centers"]
    _cats = list(_centers.keys())
    _dists, _weights = [], []
    for _c in _cats:
        _mm = _regions == _c
        _cnt = np.bincount(d_codes[_mm], minlength=d_K).astype(float)
        _dists.append(_cnt / _cnt.sum())
        _weights.append(_mm.sum())
    _dists = np.column_stack(_dists)
    _weights = np.array(_weights, float)
    _weights /= _weights.sum()
    _S = np.full((d_K, d_K), 0.05)
    np.fill_diagonal(_S, 1.0)
    _alpha, _ = partition_diversity(_S, _dists, _weights, 2.0, mode="alpha_norm")
    _gamma, _ = partition_diversity(_S, _dists, _weights, 2.0, mode="gamma")
    d_adata.uns["scdiv_alpha"] = dict(zip(_cats, _alpha))
    d_adata.uns["scdiv_gamma"] = dict(zip(_cats, _gamma))
    d_adata
    return d_H, d_K, d_NC, d_NR, d_ROI, d_RR, d_W, d_adata, d_codes, d_xy


@app.cell
def _(
    d_H,
    d_K,
    d_NC,
    d_NR,
    d_ROI,
    d_RR,
    d_W,
    d_adata,
    d_codes,
    d_xy,
    plt,
    sns,
):
    def render_fig1d(figd):
        import scdiv
        from matplotlib.patches import Circle
        from _figutils import global_cmap, local_cmap

        W, H, K, NC, NR, RR = d_W, d_H, d_K, d_NC, d_NR, d_RR
        ROI = d_ROI
        adata = d_adata
        xy = d_xy
        codes = d_codes

        # colours: muted, harmonious cool commons; vivid warm rares so the
        # common domains read as subtle tissue zones, not a checkerboard.
        common_pal = list(sns.color_palette("crest", NC))
        rare_pal = list(sns.color_palette("flare", NR))
        pal = common_pal + rare_pal

        cmap_local = local_cmap()
        cmap_global = global_cmap()

        gs = figd.add_gridspec(1, 4, width_ratios=[0.6, 1, 1, 1], wspace=0.08)
        ax_leg = figd.add_subplot(gs[0, 0])
        ax_cells = figd.add_subplot(gs[0, 1])
        ax_a = figd.add_subplot(gs[0, 2])
        ax_g = figd.add_subplot(gs[0, 3])

        # Cell-type key, styled like the panel a/c legends.
        from matplotlib.lines import Line2D

        ax_leg.set_axis_off()
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=c,
                markeredgecolor="none",
                label=f"Common {i + 1}",
            )
            for i, c in enumerate(common_pal)
        ] + [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=c,
                markeredgecolor="none",
                label=f"Rare {i + 1}",
            )
            for i, c in enumerate(rare_pal)
        ]
        ax_leg.legend(
            handles=handles,
            loc="center left",
            frameon=False,
            fontsize=7,
            handletextpad=0.5,
            labelspacing=0.6,
        )

        for t in range(K):
            mt = codes == t
            if mt.any():
                ax_cells.scatter(
                    xy[mt, 0],
                    xy[mt, 1],
                    s=5,
                    color=pal[t],
                    edgecolors="white",
                    linewidths=0.15,
                    rasterized=True,
                )
        for ax, key, cmap, title in (
            (ax_a, "scdiv_alpha", cmap_local, "Local diversity"),
            (ax_g, "scdiv_gamma", cmap_global, "Global diversity"),
        ):
            scdiv.pl.diversity_heatmap(
                adata,
                key=key,
                ax=ax,
                cmap=cmap,
                edgecolors="white",
                linewidths=0.15,
                colorbar=False,
            )
            cb = figd.colorbar(
                ax.collections[-1],
                ax=ax,
                shrink=0.58,
                aspect=13,
                pad=0.015,
            )
            cb.ax.tick_params(labelsize=6, length=2)
            cb.outline.set_linewidth(0.5)
            cb.ax.set_title(title, fontsize=8, pad=4)

        # ROI outlines + (i)-(iv) badges on the cell-type panel; the
        # descriptive labels go in the empty corners the circle leaves free.
        roman = {
            "lowA_lowG": "(i)",
            "highA_lowG": "(ii)",
            "lowA_highG": "(iii)",
            "highA_highG": "(iv)",
        }
        for name in ROI:
            (px, py), _lab = ROI[name]
            ax_cells.add_patch(
                Circle(
                    (px, py),
                    RR,
                    fill=False,
                    edgecolor="0.15",
                    linewidth=1.0,
                    linestyle=(0, (4, 2)),
                    zorder=6,
                )
            )
        corners = {
            "lowA_lowG": (0.0, 1.0, "left", "top"),
            "highA_lowG": (1.0, 1.0, "right", "top"),
            "lowA_highG": (0.0, 0.0, "left", "bottom"),
            "highA_highG": (1.0, 0.0, "right", "bottom"),
        }
        for name, (fx, fy, ha, va) in corners.items():
            ax_cells.text(
                fx,
                fy,
                f"{roman[name]} {ROI[name][1]}",
                transform=ax_cells.transAxes,
                ha=ha,
                va=va,
                fontsize=6.5,
                color="0.15",
                linespacing=1.0,
            )

        # Margin around the tissue: gives the hex polygons room at the sides
        # (so they aren't clipped) and shrinks the circle within each axes so
        # the corner labels clear the cells.
        margin = 20.0
        for ax in (ax_cells, ax_a, ax_g):
            ax.set_xlim(-margin, W + margin)
            ax.set_ylim(-margin, H + margin)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Reserve side margins so the centered colorbar titles (which are wider
        # than the thin colorbars) don't clip at the figure edge.
        figd.get_layout_engine().set(rect=(0.005, 0.0, 0.93, 1.0))


    _w_in = 179 / 25.4
    _h_in = 56 / 25.4
    # constrained layout + no tight bbox so the saved canvas stays exactly
    # 179 mm wide (a tight bbox would crop to content and change the width).
    _figd = plt.figure(figsize=(_w_in, _h_in), dpi=300, layout="constrained")
    render_fig1d(_figd)
    _figd.savefig("./figures/fig1d.svg")
    _figd.savefig("./figures/fig1d.png", dpi=300)
    _figd
    return


if __name__ == "__main__":
    app.run()
