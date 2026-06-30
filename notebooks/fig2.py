import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import re

    import marimo as mo
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from _figutils import set_paper_rc

    sns.set_style("whitegrid")
    FIG_WIDTH_MM = 179
    FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4

    set_paper_rc(font=8, tick=6.5, legend=6.5)
    return FIG_WIDTH_IN, mcolors, mo, mpatches, np, pd, plt, re, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 2: cell-type diversity over mouse organogenesis

    Here we load precomputed UMAP, cell-type assignment, and diversity data before assembling into the final figure.
    """)
    return


@app.cell
def _(pd, re):
    # Data prep: published mouse-atlas trajectory display names, plus a
    # subtype-label cleaner. These map the raw atlas labels onto the names used
    # across the figure's tables and panels.
    TRAJ_RENAME = {
        "Neuroectoderm_and_glia": "Neuroectoderm and glia",
        "Intermediate_neuronal_progenitors": "Intermediate neuronal prog.",
        "Eye_and_other": "Eye and other",
        "Ependymal_cells": "Ependymal cells",
        "CNS_neurons": "CNS neurons",
        "Mesoderm": "Mesoderm",
        "Definitive_erythroid": "Definitive erythroid",
        "Epithelial_cells": "Epithelial cells",
        "Endothelium": "Endothelium",
        "Muscle_cells": "Muscle cells",
        "Hepatocytes": "Hepatocytes",
        "White_blood_cells": "White blood cells",
        "Neural_crest_PNS_glia": "Neural crest (PNS glia)",
        "Adipocytes": "Adipocytes",
        "Primitive_erythroid": "Primitive erythroid",
        "Neural_crest_PNS_neurons": "Neural crest (PNS neurons)",
        "T_cells": "T cells",
        "Lung_and_airway": "Lung and airway",
        "Intestine": "Intestine",
        "B_cells": "B cells",
        "Olfactory_sensory_neurons": "Olfactory sensory neurons",
        "Cardiomyocytes": "Cardiomyocytes",
        "Oligodendrocytes": "Oligodendrocytes",
        "Mast_cells": "Mast cells",
        "Megakaryocytes": "Megakaryocytes",
        "Testis_and_adrenal": "Testis and adrenal",
    }

    def clean_celltype(label):
        if pd.isna(label):
            return label
        label = str(label).replace("_", " ")
        return re.split(r"[\(\/\[]", label)[0].strip()

    return TRAJ_RENAME, clean_celltype


@app.cell
def _(TRAJ_RENAME, np, pd):
    # Diversity-over-time tables

    def _to_numeric_day(day):
        return day.str.replace("[EP]", "", regex=True).replace("0", "19").astype(float)

    measures = ["coarse", "fine", "leiden", "singleton"]

    _scdiv = pd.read_csv("./data/fig2/mouse_scdiv_results.csv")
    _scdiv["t"] = _to_numeric_day(_scdiv["day"])
    scdiv = _scdiv.sort_values("t").melt(
        id_vars=["group", "t"],
        value_vars=[f"scdiv_{m}" for m in measures],
        var_name="measure",
        value_name="diversity",
    )
    scdiv["measure"] = scdiv["measure"].str.replace("scdiv_", "")

    _hill = pd.read_csv("./data/fig2/mouse_hill_results.csv")
    _hill["t"] = _to_numeric_day(_hill["day"])
    hill = _hill.sort_values("t").melt(
        id_vars=["group", "t"],
        value_vars=[f"hill_{m}" for m in measures[:3]],  # no Hill singleton
        var_name="measure",
        value_name="diversity",
    )
    hill["measure"] = hill["measure"].str.replace("hill_", "")

    # Per-cell UMAP coords + labels, slimmed from the atlas h5ad by
    # scripts/make_fig2_data.py.
    _plot = pd.read_parquet("./data/fig2/mouse_plot_slim.parquet")
    _traj = _plot["major_trajectory"].astype(str).map(TRAJ_RENAME)
    traj = _traj.fillna(_plot["major_trajectory"].astype(str)).to_numpy()
    umap_xy = _plot[["umap_x", "umap_y"]].to_numpy(dtype=np.float32)
    leiden = _plot["leiden"].astype(str).to_numpy()
    scdiv
    return hill, leiden, scdiv, traj, umap_xy


@app.cell
def _(TRAJ_RENAME, clean_celltype, pd):
    # Donut ring data (panel C)

    _df = pd.read_csv("./data/fig2/trajectory_subtype_counts.csv")
    _df["traj"] = _df["major_trajectory"].map(TRAJ_RENAME)
    _um = _df["traj"].isna()
    _df.loc[_um, "traj"] = _df.loc[_um, "major_trajectory"].apply(clean_celltype)
    _df["subtype"] = _df["celltype_update"].apply(clean_celltype)

    donut_traj_order = [t for t in TRAJ_RENAME.values() if t in _df["traj"].values]
    donut_traj_counts = (
        _df.groupby("traj")["count"].sum().reindex(donut_traj_order).fillna(0)
    )
    donut_sub_counts = (
        _df.groupby(["traj", "subtype"])["count"].sum().reset_index(name="count")
    )
    donut_traj_counts
    return donut_sub_counts, donut_traj_counts, donut_traj_order


@app.cell
def _(TRAJ_RENAME, leiden, pd, traj):
    # Composition data (panel D)
    leiden_traj_ct = pd.crosstab(
        pd.Series(leiden, name="leiden"),
        pd.Series(traj, name="traj"),
        normalize="index",
    )
    _ordered = [k for k in TRAJ_RENAME.values() if k in leiden_traj_ct.columns]
    leiden_traj_ct = leiden_traj_ct[
        _ordered + [c for c in leiden_traj_ct.columns if c not in _ordered]
    ]
    try:
        _order = sorted(leiden_traj_ct.index, key=lambda x: int(x))
    except ValueError:
        _order = sorted(leiden_traj_ct.index)
    leiden_traj_ct = leiden_traj_ct.loc[_order]
    leiden_traj_ct
    return (leiden_traj_ct,)


@app.cell
def _(mcolors):
    # Figure styling: the trajectory colour palette and the donut / label
    # plotting helpers. Pure presentation, used only by the figure-assembly cell
    # below. Palette keys are in the same canonical order as TRAJ_RENAME.
    PALETTE = {
        "Neuroectoderm and glia": "#f96100",
        "Intermediate neuronal prog.": "#2e0ab7",
        "Eye and other": "#00d450",
        "Ependymal cells": "#b75bff",
        "CNS neurons": "#e5c000",
        "Mesoderm": "#bb46c5",
        "Definitive erythroid": "#dc453e",
        "Epithelial cells": "#af9fb6",
        "Endothelium": "#00a34e",
        "Muscle cells": "#ffa1f5",
        "Hepatocytes": "#185700",
        "White blood cells": "#7ca0ff",
        "Neural crest (PNS glia)": "#fff167",
        "Adipocytes": "#7f3e39",
        "Primitive erythroid": "#ffa9a1",
        "Neural crest (PNS neurons)": "#b5ce92",
        "T cells": "#ff9d47",
        "Lung and airway": "#02b0d1",
        "Intestine": "#ff007a",
        "B cells": "#01b7a6",
        "Olfactory sensory neurons": "#e6230b",
        "Cardiomyocytes": "#643e8c",
        "Oligodendrocytes": "#916e00",
        "Mast cells": "#005361",
        "Megakaryocytes": "#3f283d",
        "Testis and adrenal": "#585d3b",
    }

    def text_color(hex_color, threshold=0.55):
        r, g, b = mcolors.to_rgb(hex_color)
        return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > threshold else "white"

    def radial_rot(angle_deg):
        a = angle_deg % 360
        return angle_deg + 180 if 90 < a < 270 else angle_deg

    def wrap(text, max_chars=13):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)
        return "\n".join(lines)

    def spread_labels(angles, min_gap=6.0, max_iter=400):
        # Push leader-line label angles (decreasing = clockwise) apart until
        # adjacent gaps reach min_gap degrees, so thin-wedge numbers don't collide.
        if len(angles) <= 1:
            return list(angles)
        s = list(angles)
        for _ in range(max_iter):
            changed = False
            for i in range(len(s) - 1):
                gap = s[i] - s[i + 1]
                if gap < min_gap:
                    mid = (s[i] + s[i + 1]) / 2.0
                    s[i] = mid + min_gap / 2.0
                    s[i + 1] = mid - min_gap / 2.0
                    changed = True
            if not changed:
                break
        return s

    return PALETTE, radial_rot, spread_labels, text_color, wrap


@app.cell
def _(
    FIG_WIDTH_IN,
    PALETTE,
    TRAJ_RENAME,
    donut_sub_counts,
    donut_traj_counts,
    donut_traj_order,
    hill,
    leiden_traj_ct,
    mpatches,
    np,
    plt,
    radial_rot,
    scdiv,
    sns,
    spread_labels,
    text_color,
    traj,
    umap_xy,
    wrap,
):
    from matplotlib.ticker import MaxNLocator

    MEASURE_ORDER = ["coarse", "fine", "leiden", "singleton"]
    MEASURE_TITLES = {
        "coarse": "Assignment 1\n(ref. coarse)",
        "fine": "Assignment 2\n(ref. fine)",
        "leiden": "Assignment 3\n($\\mathit{de\\ novo}$ coarse)",
        "singleton": "Singleton mode",
    }

    from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox

    # Tie every textual "Assignment N" reference to the Dark2 colour used for that
    # assignment's curves in panel A, drawn as a translucent highlight behind the
    # (black, legible) text.
    _D2 = sns.color_palette("Dark2")
    ASSIGN_COLOR = {
        "coarse": _D2[0],
        "fine": _D2[1],
        "leiden": _D2[2],
        "singleton": _D2[3],
    }


    def _hl(color):
        return dict(
            boxstyle="round,pad=0.25", facecolor=color, alpha=0.3, edgecolor="none"
        )


    def _segmented_text(
        target, xy, lines, *, xycoords, fontsize=8, box_alignment=(0.5, 0.0)
    ):
        # lines: list of lines; each line is a list of (text, color_or_None).
        # Coloured segments get a highlight box; None stays plain black. Assembled
        # with offsetbox so every glyph stays real editable text for Illustrator.
        rows = []
        for line in lines:
            cells = [
                TextArea(
                    t,
                    textprops=dict(
                        fontsize=fontsize,
                        color="black",
                        **({"bbox": _hl(c)} if c is not None else {}),
                    ),
                )
                for t, c in line
            ]
            rows.append(HPacker(children=cells, align="center", pad=0, sep=6))
        box = VPacker(children=rows, align="center", pad=0, sep=7)
        ab = AnnotationBbox(
            box,
            xy,
            xycoords=xycoords,
            frameon=False,
            box_alignment=box_alignment,
            pad=0,
            annotation_clip=False,
        )
        target.add_artist(ab)
        return ab


    def panel_A(fig, gs):
        # Diversity vs. embryonic day: Hill number (top) and LC diversity (bottom)
        # per assignment. Singleton has no Hill counterpart, so its top-right Hill
        # slot is left empty and it occupies only the bottom (LC diversity) row.
        sub = gs.subgridspec(2, 4, hspace=0.18, wspace=0.28)

        def draw(ax, data, m):
            d = data[data["measure"] == m]
            ax.scatter(
                d["t"],
                d["diversity"],
                s=8,
                alpha=0.55,
                color=ASSIGN_COLOR[m],
                linewidth=0,
                zorder=2,
            )
            means = d.groupby("t")["diversity"].mean()
            ax.plot(means.index, means.values, color=ASSIGN_COLOR[m], lw=1.2, zorder=3)
            ax.set_xlim(8.0, 19.6)
            ax.set_xticks([10, 12, 14, 16, 19])
            ax.set_xticklabels(["10", "12", "14", "16", "P0"])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(pad=1)

        tops, bots = {}, {}
        for i, m in enumerate(MEASURE_ORDER[:3]):
            ax_top = fig.add_subplot(sub[0, i])
            ax_bot = fig.add_subplot(sub[1, i])
            tops[m], bots[m] = ax_top, ax_bot
            draw(ax_top, hill, m)
            draw(ax_bot, scdiv, m)
            ax_top.set_title(
                MEASURE_TITLES[m], pad=10, fontsize=8, bbox=_hl(ASSIGN_COLOR[m])
            )
            ax_top.set_xlabel("")
            ax_top.tick_params(labelbottom=False)
            ax_bot.set_xlabel("Embryonic day")

        # Singleton: one panel spanning the column, then repositioned to a single-row
        # height centred on the column's vertical midpoint (between the two rows).
        # Singleton sits in the bottom (LC diversity) row, column 4; no y-label
        # since it shares the row with the labelled coarse panel.
        ax_s = fig.add_subplot(sub[1, 3])
        draw(ax_s, scdiv, "singleton")
        ax_s.set_title(
            MEASURE_TITLES["singleton"],
            pad=10,
            fontsize=8,
            bbox=_hl(ASSIGN_COLOR["singleton"]),
        )
        ax_s.set_xlabel("Embryonic day")

        tops["coarse"].set_ylabel("Hill number")
        bots["coarse"].set_ylabel("LC diversity")
        for ax in (tops["coarse"], bots["coarse"]):
            ax.yaxis.set_label_coords(-0.20, 0.5)
        return tops["coarse"]


    def panel_B(fig, gs):
        # UMAP coloured by major trajectory, numbered centroids + 2-col legend.
        sub = gs.subgridspec(1, 2, width_ratios=[1.2, 2.3], wspace=0.02)
        ax_u = fig.add_subplot(sub[0, 0])
        ax_l = fig.add_subplot(sub[0, 1])

        order = [t for t in TRAJ_RENAME.values() if t in set(traj)]
        num = {t: i for i, t in enumerate(order, 1)}

        for t in order:
            m = traj == t
            ax_u.scatter(
                umap_xy[m, 0],
                umap_xy[m, 1],
                s=0.2,
                c=PALETTE[t],
                rasterized=True,
                linewidths=0,
            )
        for t in order:
            m = traj == t
            if not m.any():
                continue
            cx, cy = (
                float(np.median(umap_xy[m, 0])),
                float(np.median(umap_xy[m, 1])),
            )
            ax_u.scatter(
                cx,
                cy,
                s=90,
                c=PALETTE[t],
                zorder=5,
                linewidths=0.6,
                edgecolors="white",
                clip_on=False,
            )
            ax_u.text(
                cx,
                cy,
                str(num[t]),
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                color=text_color(PALETTE[t]),
                zorder=6,
                clip_on=False,
            )
        ax_u.set_aspect("equal")
        ax_u.set_anchor(
            "C"
        )  # centre in cell: slides UMAP off the left edge into the dead gap
        ax_u.margins(0.01)  # trim default padding so the UMAP fills its whitespace
        ax_u.axis("off")

        # Corner UMAP-axis indicator.
        # Equal aspect keeps the L-arrows perpendicular and equal length even
        # though the UMAP panel itself isn't square.
        ax_arr = ax_u.inset_axes([0.0, 0.0, 0.16, 0.16])
        ax_arr.set_xlim(0, 1)
        ax_arr.set_ylim(0, 1)
        ax_arr.set_aspect("equal")
        ax_arr.axis("off")
        aw = dict(arrowstyle="->", color="black", lw=0.8, shrinkA=0, shrinkB=0)
        ax_arr.annotate("", xy=(0.95, 0.1), xytext=(0.1, 0.1), arrowprops=aw)
        ax_arr.annotate("", xy=(0.1, 0.95), xytext=(0.1, 0.1), arrowprops=aw)
        ax_arr.text(
            0.55, -0.06, "UMAP 1", ha="center", va="top", fontsize=6, clip_on=False
        )
        ax_arr.text(
            -0.06,
            0.55,
            "UMAP 2",
            ha="right",
            va="center",
            rotation=90,
            fontsize=6,
            clip_on=False,
        )

        ax_l.axis("off")
        ax_l.set_xlim(0, 1)
        ax_l.set_ylim(0, 1)
        n_col = 3
        n_row = (len(order) + n_col - 1) // n_col
        y_step = 0.09  # tighter than full-height spacing
        y_top = 0.5 + n_row * y_step / 2  # centre the block vertically
        x_cols = [0.0, 1 / 3, 2 / 3]
        for idx2, t in enumerate(order):
            col, row = idx2 // n_row, idx2 % n_row
            xc = x_cols[col] + 0.02
            yc = y_top - (row + 0.5) * y_step
            ax_l.scatter(xc, yc, s=70, c=PALETTE[t], zorder=3, linewidths=0)
            ax_l.text(
                xc,
                yc,
                str(idx2 + 1),
                ha="center",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color=text_color(PALETTE[t]),
                zorder=4,
            )
            ax_l.text(xc + 0.035, yc, t, ha="left", va="center", fontsize=6.5)
        # Title over the legend (the colour key for Assignment 1 = major
        # trajectory). Trim the legend axis top, which butts the figure
        # margin, so the title fits; matches panels B/C/D at 8pt.
        p_l = ax_l.get_position()
        ax_l.set_position([p_l.x0, p_l.y0, p_l.width, p_l.height - 0.035])
        ax_l.set_title(
            "Assignment 1 (ref. coarse)",
            fontsize=8,
            pad=9,
            bbox=_hl(ASSIGN_COLOR["coarse"]),
        )
        return ax_u


    def panel_C(fig, gs):
        # Nested donut: trajectory (inner) -> subtype (outer). Thin wedges get a
        # bracket + leader line to an outside number rather than an inside label.
        ax = fig.add_subplot(gs)
        XLIM = 1.10
        PTS_PER_UNIT = 4.0 * 72 / (2 * XLIM)
        R_IN, R_MID, R_OUT = 0.28, 0.52, 0.87
        R_BRKT, R_ELBOW, R_LEADER, R_TEXT = (
            R_OUT + 0.03,
            R_OUT + 0.08,
            R_OUT + 0.14,
            R_OUT + 0.18,
        )
        r_t, r_s = (R_IN + R_MID) / 2.0, (R_MID + R_OUT) / 2.0

        def inner_fits(sweep):
            return r_t * np.radians(sweep) >= (8 / PTS_PER_UNIT) * 1.6

        def best_outer_font(subtype, sweep):
            wrapped = wrap(subtype)
            nl = wrapped.count("\n") + 1
            for fs in (5, 4, 3):
                if r_s * np.radians(sweep) >= (nl * fs * 1.3 / PTS_PER_UNIT) * 1.6:
                    return fs, wrapped
            return None, None

        # Donut ring data aggregated upstream (donut_* analysis cell).
        traj_order = donut_traj_order
        traj_num = {t: i for i, t in enumerate(traj_order, 1)}
        traj_counts = donut_traj_counts
        sub_counts = donut_sub_counts
        total = float(traj_counts.sum())

        wedges = []
        cur = 90.0
        for t in traj_order:
            tc = traj_counts[t]
            if tc == 0:
                continue
            sweep = tc / total * 360
            wedges.append(
                dict(
                    traj=t,
                    num=traj_num[t],
                    color=PALETTE.get(t, "#BBB"),
                    sweep=sweep,
                    theta1=cur - sweep,
                    theta2=cur,
                    mid=cur - sweep / 2,
                )
            )
            cur -= sweep
        needs = [w for w in wedges if not inner_fits(w["sweep"])]
        if needs:
            for w, sa in zip(needs, spread_labels([w["mid"] for w in needs])):
                w["spread"] = sa
        for w in wedges:
            w.setdefault("spread", w["mid"])

        ax.set_xlim(-XLIM, XLIM)
        ax.set_ylim(
            -XLIM + 0.2, XLIM + 0.2
        )  # shift donut south, away from the title
        ax.set_aspect("equal")
        ax.axis("off")
        for w in wedges:
            t1, t2, mid, spr = w["theta1"], w["theta2"], w["mid"], w["spread"]
            mr, sr = np.radians(mid), np.radians(spr)
            color, num, t = w["color"], w["num"], w["traj"]
            ax.add_patch(
                mpatches.Wedge(
                    (0, 0),
                    R_MID,
                    t1,
                    t2,
                    width=R_MID - R_IN,
                    facecolor=color,
                    linewidth=0.5,
                    edgecolor="white",
                    zorder=2,
                )
            )
            if inner_fits(w["sweep"]):
                ax.text(
                    r_t * np.cos(mr),
                    r_t * np.sin(mr),
                    str(num),
                    ha="center",
                    va="center",
                    fontsize=6,
                    fontweight="bold",
                    rotation=radial_rot(mid),
                    rotation_mode="anchor",
                    color=text_color(color),
                    zorder=5,
                )
            else:
                n_arc = max(12, int(abs(t2 - t1) * 4))
                arc = np.linspace(t1, t2, n_arc)
                ax.plot(
                    R_BRKT * np.cos(np.radians(arc)),
                    R_BRKT * np.sin(np.radians(arc)),
                    color=color,
                    lw=0.8,
                    zorder=10,
                    clip_on=False,
                    solid_capstyle="round",
                )
                for td in (t1, t2):
                    tr = np.radians(td)
                    ax.plot(
                        [R_OUT * np.cos(tr), R_BRKT * np.cos(tr)],
                        [R_OUT * np.sin(tr), R_BRKT * np.sin(tr)],
                        color=color,
                        lw=0.8,
                        zorder=10,
                        clip_on=False,
                    )
                ax.plot(
                    [
                        R_BRKT * np.cos(mr),
                        R_ELBOW * np.cos(mr),
                        R_LEADER * np.cos(sr),
                    ],
                    [
                        R_BRKT * np.sin(mr),
                        R_ELBOW * np.sin(mr),
                        R_LEADER * np.sin(sr),
                    ],
                    color=color,
                    lw=0.6,
                    zorder=10,
                    clip_on=False,
                )
                cx, cy = np.cos(sr), np.sin(sr)
                ha = (
                    ("left" if cx > 0 else "right")
                    if abs(cx) >= abs(cy)
                    else "center"
                )
                va = (
                    "center"
                    if abs(cx) >= abs(cy)
                    else ("bottom" if cy > 0 else "top")
                )
                ax.text(
                    R_TEXT * cx,
                    R_TEXT * cy,
                    str(num),
                    ha=ha,
                    va=va,
                    fontsize=6,
                    fontweight="bold",
                    color=color,
                    zorder=10,
                    clip_on=False,
                )
            subs = sub_counts[sub_counts["traj"] == t].sort_values(
                "count", ascending=False
            )
            ss = t2
            for _, row in subs.iterrows():
                sw = row["count"] / total * 360
                st1, st2 = ss - sw, ss
                sm = (st1 + st2) / 2
                ax.add_patch(
                    mpatches.Wedge(
                        (0, 0),
                        R_OUT,
                        st1,
                        st2,
                        width=R_OUT - R_MID,
                        facecolor=color,
                        linewidth=0.3,
                        edgecolor="white",
                        alpha=0.65,
                        zorder=2,
                    )
                )
                fs, wrapped = best_outer_font(row["subtype"], sw)
                if fs is not None:
                    ax.text(
                        r_s * np.cos(np.radians(sm)),
                        r_s * np.sin(np.radians(sm)),
                        wrapped,
                        ha="center",
                        va="center",
                        fontsize=fs,
                        linespacing=1.1,
                        rotation=radial_rot(sm),
                        rotation_mode="anchor",
                        color=text_color(color),
                        zorder=5,
                    )
                ss -= sw
        ax._panel_title = _segmented_text(
            ax,
            (0.5, 1.04),
            [
                [
                    ("Assignment 2 (ref. fine)", ASSIGN_COLOR["fine"]),
                    ("as a subassignment of", None),
                ],
                [("Assignment 1 (ref. coarse)", ASSIGN_COLOR["coarse"])],
            ],
            xycoords=ax.transAxes,
        )
        # Nudge the donut + its title down within panel B; the title is
        # anchored to ax.transAxes so it follows. No other panel is touched.
        _pc = ax.get_position()
        ax.set_position([_pc.x0, _pc.y0 - 0.03, _pc.width, _pc.height])
        return ax


    def panel_D(fig, gs):
        # Trajectory composition of each de-novo (Leiden) cluster, split over two
        # columns of stacked horizontal bars.
        sub = gs.subgridspec(1, 2, wspace=0.16)
        ax1 = fig.add_subplot(sub[0, 0])
        ax2 = fig.add_subplot(sub[0, 1], sharex=ax1)

        # Row-normalised Leiden x trajectory crosstab from the analysis cell.
        ct = leiden_traj_ct
        colors = [PALETTE.get(c, "#bdbdbd") for c in ct.columns]
        half = (len(ct) + 1) // 2
        # pandas barh hides tick labels on the shared-x partner, so force both
        # columns to show the proportion ticks and the x-axis label.
        for ax, data in zip((ax1, ax2), (ct.iloc[:half], ct.iloc[half:])):
            data.plot(
                kind="barh",
                stacked=True,
                color=colors,
                ax=ax,
                edgecolor="none",
                legend=False,
                width=0.85,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", linestyle="--", alpha=0.4)
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.xaxis.set_visible(True)
            ax.tick_params(axis="x", labelbottom=True, labelsize=6.5)
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel("Proportion of cells")
            ax.xaxis.label.set_visible(True)
        ax1.set_ylabel("Assignment 3 cluster", bbox=_hl(ASSIGN_COLOR["leiden"]))
        ax2.set_ylabel("")
        # Title centred over both columns, not just the left axis.
        _p1 = ax1.get_position()
        _p2 = ax2.get_position()
        ax1._panel_title = _segmented_text(
            fig,
            ((_p1.x0 + _p2.x1) / 2, _p1.y1 + 0.022),
            [
                [
                    ("Assignment 1 (ref. coarse)", ASSIGN_COLOR["coarse"]),
                    ("composition with", None),
                ],
                [
                    ("respect to", None),
                    (
                        "Assignment 3 ($\\mathit{de\\ novo}$ coarse)",
                        ASSIGN_COLOR["leiden"],
                    ),
                ],
            ],
            xycoords="figure fraction",
        )
        return ax1


    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 1.08), dpi=300)
    outer = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.0, 0.005, 1.2, 0.18, 1.35],
        width_ratios=[2, 3],
        wspace=0.16,
        hspace=0.22,
        left=0.07,
        right=0.985,
        top=0.965,
        bottom=0.04,
    )
    _b0 = panel_B(fig, outer[0, :])
    _c0 = panel_C(fig, outer[2, 0])
    _d0 = panel_D(fig, outer[2, 1])
    _a0 = panel_A(fig, outer[4, :])  # rows 1 and 3 are spacers (A->B and above D)
    # Draw once so title extents are final, then place the panel letters.
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    _inv = fig.transFigure.inverted()


    def _artist_top(_artist):
        _bb = _artist.get_window_extent(renderer=_rend)
        return _inv.transform((0, _bb.y1))[1]


    _LETTER_X = 0.013
    # A (no title) and D keep their current anchoring on the left edge.
    for _ax, _lab, _dy in ((_b0, "A", 1.02), (_a0, "D", 1.32)):
        _p = _ax.get_position()
        fig.text(
            _LETTER_X,
            _p.y0 + _dy * _p.height,
            _lab,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # B and C share one top, level with their (same-row) titles, so they align.
    _bc_top = max(_artist_top(_c0._panel_title), _artist_top(_d0._panel_title))
    fig.text(
        _LETTER_X,
        _bc_top,
        "B",
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )
    _dp = _d0.get_position()
    fig.text(
        _dp.x0 - 0.26 * _dp.width,
        _bc_top,
        "C",
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )

    fig.savefig("./figures/fig2_mouse.pdf", dpi=600)
    fig.savefig("./figures/fig2_mouse.png", dpi=600)
    fig
    return


if __name__ == "__main__":
    app.run()
