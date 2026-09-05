r"""Exhibit candidate: grid design versus beyond-grid extrapolation accuracy.

Single panel in the paper's house style, addressing each element of the
proposed grid redesign: the paper's linear five-point grid as its own marked
point, double versus triple exponential spacing as separate series, extent
on the horizontal axis in units of a/h (assets to human wealth), and the
a = 2h reference marked. Metric: the paper's standard accuracy measure
(max absolute consumption error against a dense truth) beyond the grid top.

Run from the repo root:

    PYTHONPATH=code uv run python code/explorations/grid-extent/make_exhibit.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import run_grid_extent
from style import (
    ALPHA_MEDIUM,
    ARK_BLUE,
    ARK_LIGHTBLUE,
    ARK_PINK,
    LINE_STYLE_DASHED,
    LINE_WIDTH_THICK,
    setup_figure,
)

LOG = logging.getLogger("grid-extent-exhibit")

HERE = Path(__file__).resolve().parent
# Single source of truth: the paper's figure directory is the ONLY output
# location; no copy is kept beside the exploration.
IMAGES_DIR = HERE.parents[2] / "content" / "images"


def _load_variants():
    """Read the experiment output, refusing a JSON written by other parameters."""
    res = json.loads((HERE / "grid_extent.json").read_text())
    expected = {k: repr(v) for k, v in sorted(run_grid_extent.PARAMS.items())}
    if res.get("params") != expected:
        msg = (
            "grid_extent.json was produced by different parameters than the "
            "current run_grid_extent.py; rerun the experiment before plotting. "
            f"JSON stamp: {res.get('params')}; expected: {expected}"
        )
        raise RuntimeError(msg)
    return res["variants"]


def _plot_extent_series(ax, var) -> None:
    """One line per nesting factor, ordered along the grid-extent axis."""
    series = {
        3: ("Triple exponential spacing", ARK_BLUE, "solid"),
        2: ("Double exponential spacing", ARK_LIGHTBLUE, LINE_STYLE_DASHED),
    }
    for nest, (label, color, ls) in series.items():
        # Filter by label prefix: the count-confound control shares nest=3
        # but belongs to no extent series.
        rows = sorted(
            (v for v in var if v["nest"] == nest and v["label"].startswith("nest")),
            key=lambda v: v["a_over_h"],
        )
        ax.plot(
            np.array([v["a_over_h"] for v in rows]),
            np.array([v["max_abs_err"] for v in rows]),
            marker="o",
            color=color,
            linewidth=LINE_WIDTH_THICK,
            linestyle=ls,
            label=label,
        )


def _configure_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.1, 0.25, 0.5, 1, 2, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r"Grid top in units of human wealth, $\overline{a}/h$")
    # Plain words, matching SolutionErrorPlot's "Absolute Consumption Error".
    # The tilde that used to mark the approximation here appears nowhere in the
    # manuscript (which uses grave for the approximation and hat for the truth),
    # so it read as an unexplained diacritic on an otherwise identical c(m).
    ax.set_ylabel("Max absolute consumption error beyond grid")
    ax.legend()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    var = _load_variants()

    fig, ax = setup_figure(title="")
    _plot_extent_series(ax, var)

    paper = next(v for v in var if v["nest"] == -1)
    ax.plot(
        paper["a_over_h"],
        paper["max_abs_err"],
        marker="*",
        color=ARK_PINK,
        markersize=14,
        linestyle="none",
        label="Linear five-point grid",
    )
    ax.axvline(
        x=2.0,
        color="gray",
        linestyle=LINE_STYLE_DASHED,
        alpha=ALPHA_MEDIUM,
        label=r"$\overline{a} = 2h$",
    )
    _configure_axes(ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(IMAGES_DIR / f"GridExtentPlot.{ext}", dpi=300, bbox_inches="tight")
    LOG.info("Wrote GridExtentPlot.pdf/.png to %s", IMAGES_DIR)


if __name__ == "__main__":
    main()
