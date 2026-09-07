r"""Regenerate the static paper figures under ``content/images/``.

Historically the manuscript figures were committed as hand-produced PDFs with
no generator, so they could drift from the solver and from the house style.
This script solves the same calibration the companion notebook uses and
writes every paper figure from the shared plotting layer, so the images are
reproducible and consistent with ``plotting.py``/``style.py``.

Figures are written *without* an in-image title: the LaTeX ``\caption`` is the
figure title (see ``setup_figure``). Each figure is saved as both ``.pdf`` (the
source the MyST build converts) and ``.png`` (for quick inspection). PDF
metadata strips ``CreationDate``/``ModDate`` so an unchanged figure
regenerates byte-identical; the ``Producer`` string still carries the
matplotlib version and so changes on a matplotlib upgrade, which is expected.

Each figure is a standalone function registered in ``FIGURES``, and the shared
model solves it depends on are cached (``functools.cache``) so that
regenerating one figure only solves what that figure actually needs.

``GridExtentPlot`` is not produced here: see
``code/explorations/grid-extent/make_exhibit.py``.

Run from anywhere with::

    uv run python code/make_paper_figures.py
        # write every figure under content/images/ (the default action)

    uv run python code/make_paper_figures.py --write --only LogitFunctionPlot
        # write only the named figure(s); --only takes a repeatable list

    uv run python code/make_paper_figures.py --check
        # regenerate every figure to a temp dir and diff byte-for-byte
        # against content/images/, without writing; exits 1 and names any
        # file that differs or is missing

    uv run python code/make_paper_figures.py --check --only LogitFunctionPlot
        # --check a subset

    uv run python code/make_paper_figures.py --list
        # print the registered figure names, in build order, and exit
"""

from __future__ import annotations

import argparse
import filecmp
import logging
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from moderation import (
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
)
from plotting import (
    plot_consumption_bounds,
    plot_logit_function,
    plot_moderation_ratio,
    plot_mom_mpc,
    plot_precautionary_gaps,
    plot_share_bounds,
    plot_share_extrapolation,
    plot_share_logit,
    plot_solution_gaps,
)
from portfolio import ModeratedShareFunc, PortfolioMoMConsumerType

IMAGES_DIR = Path(__file__).resolve().parents[1] / "content" / "images"

logger = logging.getLogger("make_paper_figures")

# Shared calibration: verify_table.py is the single source of truth for the
# Table 1 params and grids (duplicated dicts drifted once; see the 2026-08-14
# review pass).
from verify_table import DENSE_GRID, PARAMS, SPARSE_GRID  # noqa: E402

# Portfolio choice is a different model from the consumption calibration
# above, so it takes HARK's init_portfolio rather than PARAMS. verify_table
# stays the single source of truth for the consumption figures.
_PORT_GRID = {"cycles": 0, "aXtraMax": 100.0, "aXtraCount": 200, "tolerance": 1e-8}


class _OutputTarget:
    """Holds the directory ``_save`` writes to, mutable for ``--check``."""

    directory: Path = IMAGES_DIR


_output = _OutputTarget()


@contextmanager
def _redirect_output(directory: Path) -> Iterator[None]:
    """Temporarily point ``_save`` at ``directory`` instead of ``IMAGES_DIR``."""
    previous = _output.directory
    _output.directory = directory
    try:
        yield
    finally:
        _output.directory = previous


def _save(name: str) -> None:
    """Save the current figure as PDF and PNG under ``_output.directory``."""
    fig = plt.gcf()
    fig.savefig(
        _output.directory / f"{name}.pdf",
        bbox_inches="tight",
        dpi=300,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(_output.directory / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _solve(consumer_cls, grid, **overrides):
    consumer = consumer_cls(**(PARAMS | grid | overrides))
    consumer.solve()
    return consumer.solution[0]


# --- Shared solves. Cached (zero-argument functions, so functools.cache just
# --- memoizes the single call) so that regenerating one figure on demand only
# --- solves the models that figure actually depends on.


@cache
def _truth():
    """High-precision EGM solution on the dense grid ("truth")."""
    return _solve(IndShockEGMConsumerType, DENSE_GRID)


@cache
def _egm_sparse():
    """Plain EGM solution on the sparse (Table 1) grid."""
    return _solve(IndShockEGMConsumerType, SPARSE_GRID)


@cache
def _mom_sparse():
    """MoM solution on the sparse (Table 1) grid."""
    return _solve(IndShockMoMConsumerType, SPARSE_GRID)


@cache
def _port_near():
    """MoM portfolio solution on the base grid (source of ``ShareLimit``)."""
    port = PortfolioMoMConsumerType(**(init_portfolio | _PORT_GRID))
    port.solve()
    return port.solution[0]


@cache
def _port_far():
    """MoM portfolio solution on a grid reaching the myopic limit.

    At m=100 the share is still 27% above the myopic limit, at m=18000 only
    0.3%. Nothing is plotted beyond what this solve actually reaches.
    """
    far = PortfolioMoMConsumerType(
        **(init_portfolio | {**_PORT_GRID, "aXtraMax": 20000.0, "aXtraCount": 800}),
    )
    far.solve()
    return far.solution[0]


@cache
def _port_ref():
    """Converged (not merely dense) HARK portfolio reference solution.

    The reference must be CONVERGED, not merely dense; a coarse one
    understates MoM by measuring its own error instead (15x at 100/200, 108x
    at 2000/1000).
    """
    ref = PortfolioConsumerType(
        **(
            init_portfolio
            | {"cycles": 0, "aXtraMax": 2000.0, "aXtraCount": 1000, "tolerance": 1e-8}
        ),
    )
    ref.solve()
    return ref.solution[0]


# --- One function per figure, registered in FIGURES below in the original
# --- build order.


def fig_ExtrapProblemPlot() -> None:
    """EGM extrapolation problem (negative precautionary saving)."""
    plot_precautionary_gaps(_truth(), _egm_sparse(), title="", subtitle="")
    _save("ExtrapProblemPlot")


def fig_IntExpFOCInvPesReaOptNeedHiPlot() -> None:
    """Truth bounded by optimist/pessimist theory."""
    plot_consumption_bounds(
        _truth(),
        title="",
        subtitle="",
        show_grid_points=False,
        legend="Realist",
    )
    _save("IntExpFOCInvPesReaOptNeedHiPlot")


def fig_ExtrapProblemSolvedPlot() -> None:
    """MoM solves the extrapolation problem."""
    plot_precautionary_gaps(_truth(), _mom_sparse(), title="", subtitle="")
    _save("ExtrapProblemSolvedPlot")


def fig_LogitFunctionPlot() -> None:
    """The object the method actually interpolates.

    chi(mu) is smooth and close to linear at both ends, which is why linear
    interpolation and extrapolation of the logit outperform the same
    operations on consumption.
    """
    plot_logit_function(_mom_sparse(), title="", subtitle="")
    _save("LogitFunctionPlot")


def fig_SolutionErrorPlot() -> None:
    """Absolute consumption error across the domain (companion to Table 1).

    Both methods on one log axis, sharing the sparse grid.
    """
    plot_solution_gaps(
        _truth(),
        [_egm_sparse(), _mom_sparse()],
        title="",
        subtitle="",
    )
    _save("SolutionErrorPlot")


def fig_IntExpFOCInvPesReaOptNeed45Plot() -> None:
    """MoM consumption with the tighter upper bound / cusp."""
    plot_consumption_bounds(
        _mom_sparse(),
        title="",
        subtitle="",
        m_max=3.0,
        show_tight_bound=True,
    )
    _save("IntExpFOCInvPesReaOptNeed45Plot")


def fig_ModerationRatioPlot() -> None:
    """The moderation ratio itself.

    The consumption-levels figure cannot show the bracket, whose width is a
    constant 0.430 and so only 7.7% of the optimist's rule by m = 10; omega
    uses its whole range over the same span.
    """
    plot_moderation_ratio(_mom_sparse(), title="", subtitle="", m_max=20.0)
    _save("ModerationRatioPlot")


def fig_MoMMPCPlot() -> None:
    """The MPC decomposition behind the Hermite section.

    The moderated MPC is a weighted average of the two limiting MPCs, so it
    is bounded by them at every level of wealth.
    """
    plot_mom_mpc(_mom_sparse(), title="", subtitle="")
    _save("MoMMPCPlot")


def fig_PortfolioShareBoundsPlot() -> None:
    """The share inside its bracket, myopic limit below and cap above."""
    plot_share_bounds(_port_far(), title="", subtitle="", m_max=20000.0)
    _save("PortfolioShareBoundsPlot")


def fig_PortfolioShareLogitPlot() -> None:
    """Log of the moderation ratio is linear in log m.

    The slope is the tail's decay exponent, measured rather than imposed.
    """
    plot_share_logit(_port_far(), title="", subtitle="", m_min=10.0, m_max=1000.0)
    _save("PortfolioShareLogitPlot")


def fig_PortfolioShareExtrapolationPlot() -> None:
    """Beyond the solved grid.

    The portfolio analogue of the consumption extrapolation pair - linear
    overshoots, HARK decays onto ShareLimit early, moderation inherits the
    tail slope it measured.
    """
    share_limit = _port_near().ShareLimit

    def moderated(m_nodes, s_nodes):
        return ModeratedShareFunc(m_nodes, s_nodes, share_limit)

    plot_share_extrapolation(
        _port_ref(),
        share_limit,
        moderated,
        title="",
        subtitle="",
    )
    _save("PortfolioShareExtrapolationPlot")


FIGURES: dict[str, Callable[[], None]] = {
    "ExtrapProblemPlot": fig_ExtrapProblemPlot,
    "IntExpFOCInvPesReaOptNeedHiPlot": fig_IntExpFOCInvPesReaOptNeedHiPlot,
    "ExtrapProblemSolvedPlot": fig_ExtrapProblemSolvedPlot,
    "LogitFunctionPlot": fig_LogitFunctionPlot,
    "SolutionErrorPlot": fig_SolutionErrorPlot,
    "IntExpFOCInvPesReaOptNeed45Plot": fig_IntExpFOCInvPesReaOptNeed45Plot,
    "ModerationRatioPlot": fig_ModerationRatioPlot,
    "MoMMPCPlot": fig_MoMMPCPlot,
    "PortfolioShareBoundsPlot": fig_PortfolioShareBoundsPlot,
    "PortfolioShareLogitPlot": fig_PortfolioShareLogitPlot,
    "PortfolioShareExtrapolationPlot": fig_PortfolioShareExtrapolationPlot,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate paper figures under content/images/.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--write",
        action="store_true",
        help="Write the selected figures under content/images/ (the default action).",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="Regenerate the selected figures to a temporary directory and "
        "compare byte-for-byte against the committed files under "
        "content/images/, without writing. Exits 1 and names any file that "
        "differs or is missing.",
    )
    action.add_argument(
        "--list",
        action="store_true",
        help="Print the registered figure names, in build order, and exit.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="Limit to the named figure(s). See --list for valid names. "
        "Default: every figure.",
    )
    args = parser.parse_args(argv)

    if args.only:
        unknown = [name for name in args.only if name not in FIGURES]
        if unknown:
            valid = ", ".join(FIGURES)
            parser.error(
                f"unknown figure name(s): {', '.join(unknown)}. Valid names: {valid}",
            )
    return args


def _check(names: list[str]) -> bool:
    """Regenerate ``names`` to a temp dir and diff against ``IMAGES_DIR``.

    Returns ``True`` iff every regenerated file matches the committed one
    byte-for-byte. Logs each differing or missing file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with _redirect_output(tmp_dir):
            for name in names:
                FIGURES[name]()

        problems = []
        for name in names:
            for ext in ("pdf", "png"):
                committed = IMAGES_DIR / f"{name}.{ext}"
                regenerated = tmp_dir / f"{name}.{ext}"
                if not committed.exists():
                    problems.append(f"{name}.{ext} (missing from content/images/)")
                elif not filecmp.cmp(committed, regenerated, shallow=False):
                    problems.append(f"{name}.{ext}")

    if problems:
        logger.error("Figures differ from committed content/images/:")
        for entry in problems:
            logger.error("  %s", entry)
        return False

    logger.info("OK: %d figure(s) match content/images/.", len(names))
    return True


def main(argv: list[str] | None = None) -> None:
    # Without this, plotting.py's warnings about failed gridpoint extraction go
    # nowhere (lastResort only handles WARNING and the default handler is
    # unconfigured), and a figure would be written silently missing markers.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    args = _parse_args(argv)

    if args.list:
        for name in FIGURES:
            logger.info(name)
        return

    names = args.only if args.only else list(FIGURES)

    if args.check:
        if not _check(names):
            sys.exit(1)
        return

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
