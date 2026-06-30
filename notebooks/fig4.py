import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from _figutils import set_paper_rc

    sns.set_style("whitegrid")
    set_paper_rc(font=8, tick=7, legend=7)
    FIG_WIDTH_IN = 89 / 25.4
    return FIG_WIDTH_IN, mo, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figure 4: aggregoid time-course

    scdiv was run on each (condition, timepoint) sample of the aggregoid
    experiment, producing four per-sample summaries: number of cell types,
    Hill number, cell-type diversity, and singleton diversity.
    """)
    return


@app.cell
def _(pd):
    aggregoid = pd.read_csv("./data/fig4/scdiv_aggregoid.csv")
    aggregoid["time_numeric"] = aggregoid["time"].str.removesuffix("h").astype(int)
    aggregoid
    return (aggregoid,)


@app.cell
def _(FIG_WIDTH_IN, aggregoid, plt):
    panels = [
        ("A", "num_cell_types", "Number of cell types"),
        ("B", "hill_number", "Hill number"),
        ("C", "scdiv_celltype", "Cell-type diversity"),
        ("D", "scdiv_singleton", "Singleton diversity"),
    ]
    conditions = ["Ctrl", "NACL", "RACL", "XAL"]
    cond_colors = dict(zip(conditions, ["#333333", "#0072B2", "#D55E00", "#E69F00"]))
    cond_markers = dict(zip(conditions, ["o", "^", "D", "s"]))
    timepoints = sorted(aggregoid["time_numeric"].unique())

    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.92), sharex=True
    )

    for ax, (letter, col, title) in zip(axes.flat, panels):
        for cond in conditions:
            sub = aggregoid[aggregoid["condition"] == cond].sort_values("time_numeric")
            ax.plot(
                sub["time_numeric"],
                sub[col],
                color=cond_colors[cond],
                lw=0.9,
                marker=cond_markers[cond],
                ms=3,
                zorder=2,
                label=cond,
            )
        ax.set_title(title, pad=4, loc="center")
        ax.set_xticks(timepoints)
        ax.tick_params(axis="both", pad=2)

    for ax in axes[1, :]:
        ax.set_xlabel("Time (h)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        title="Condition",
        handletextpad=0.5,
        columnspacing=1.2,
        handlelength=1.8,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=1.0, h_pad=0.8)

    # Panel letters aligned to each panel's tight bounding box (column-left,
    # row-top), so A/C and B/D share an x and A/B, C/D share a y.
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    _inv = fig.transFigure.inverted()
    _col_left, _row_top = {}, {}
    for _i in range(axes.shape[0]):
        for _j in range(axes.shape[1]):
            _bb = axes[_i, _j].get_tightbbox(_rend)
            (_x0, _), (_, _y1) = _inv.transform([[_bb.x0, _bb.y0], [_bb.x1, _bb.y1]])
            _col_left[_j] = min(_col_left.get(_j, _x0), _x0)
            _row_top[_i] = max(_row_top.get(_i, _y1), _y1)
    for _i in range(axes.shape[0]):
        for _j in range(axes.shape[1]):
            _letter = panels[_i * axes.shape[1] + _j][0]
            fig.text(
                _col_left[_j],
                _row_top[_i],
                _letter,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
            )

    fig.savefig("./figures/fig4.pdf")
    fig
    return


if __name__ == "__main__":
    app.run()
