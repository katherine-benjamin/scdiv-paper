"""Shared helpers for the scdiv paper-figure notebooks.

Imported by the ``fig*.py`` notebooks as ``from _figutils import ...``. marimo
puts the notebook's own directory on ``sys.path``, so this resolves both when a
notebook is opened with ``marimo edit`` and when an exported script is run from
this ``notebooks/`` directory (see the README's headless-render instructions).
"""

from __future__ import annotations

import numpy as np


def truncate_cmap(name, lo=0.2, hi=1.0, n=256):
    """Return the ``[lo, hi]`` slice of the named matplotlib colormap.

    Dropping the near-black low end keeps the darkest hexes from reading as
    "missing" in the diversity heatmaps. See ``local_cmap``/``global_cmap`` for
    the project-wide local/global pairing.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_t", base(np.linspace(lo, hi, n))
    )


# Project convention: local diversity is a truncated rocket, global a truncated
# mako, both clipped to [0.2, 1.0]. Colour means the same thing across figures.
def local_cmap():
    return truncate_cmap("rocket", 0.2, 1.0)


def global_cmap():
    return truncate_cmap("mako", 0.2, 1.0)


def hill_number(counts, q=2.0):
    """Hill number of order ``q`` from counts or proportions.

    Input is normalised internally, so raw counts or a probability vector both
    work. ``q=2`` gives the inverse Simpson index ``1 / sum(p_i ** 2)``.
    """
    p = np.asarray(counts, dtype=float)
    p = p / p.sum()
    if q == 1.0:
        return float(np.exp(-np.sum(p * np.log(p))))
    return float((p**q).sum() ** (1.0 / (1.0 - q)))


def set_paper_rc(*, font=8, axes=None, title=None, tick=7, legend=7):
    """Apply the shared matplotlib rcParams: Type-42 (editable-in-Illustrator)
    fonts plus the given point sizes. ``axes``/``title`` default to ``font``.

    The seaborn style/theme is left to the caller, since the notebooks differ
    (``set_style`` vs ``set_theme``).
    """
    import matplotlib.pyplot as plt

    axes = font if axes is None else axes
    title = font if title is None else title
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": font,
            "axes.labelsize": axes,
            "axes.titlesize": title,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "legend.fontsize": legend,
        }
    )
