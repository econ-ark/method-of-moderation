"""Verify Table 1 approximation errors (EGM vs MoM).

Reproduces `tbl:approx-errors` from `content/paper/moderation_letters.md`.
Computes the maximum absolute difference between a high-precision EGM
solution ("truth") and each of the sparse EGM and MoM solutions on each
sub-interval of the sparse grid, plus the extrapolation region. Exits
non-zero if any MoM interval error is larger than the corresponding EGM
error, since the paper's central claim is that MoM is uniformly more
accurate.

Run from the repository root:

    uv run python code/verify_table.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Make `code/` importable regardless of caller CWD, without leaking a relative
# entry into sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from moderation import IndShockEGMConsumerType, IndShockMoMConsumerType  # noqa: E402

logger = logging.getLogger("verify_table")


PARAMS = {
    "CRRA": 2.0,
    "DiscFac": 0.96,
    "Rfree": [1.02],
    "TranShkStd": [1.0],
    "cycles": 1,
    "LivPrb": [1.0],
    "vFuncBool": True,
    "CubicBool": True,
    "PermGroFac": [1.0],
    "PermShkStd": [0.0],
    "TranShkCount": 7,
    "UnempPrb": 0.0,  # The Table 1 calibration; MPCmax is still finite via the
    # nonzero worst transitory shock from the TranShk grid.
    "BoroCnstArt": None,
}
DENSE_GRID = {"aXtraMin": 0.001, "aXtraMax": 40, "aXtraCount": 500, "aXtraNestFac": 3}
SPARSE_GRID = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 5, "aXtraNestFac": -1}
M_BAR = 30.0
N_EVAL = 1000


def _solve(consumer_cls, params, grid):
    consumer = consumer_cls(**(params | grid))
    consumer.solve()
    return consumer.solution[0]


def _interval_max_errors(truth_cfunc, sol_cfunc, grid_pts, m_bar, n_eval):
    """Maximum |truth - approx| on each sub-interval plus the extrapolation region."""
    errors = []
    for i in range(len(grid_pts) - 1):
        left, right = grid_pts[i], grid_pts[i + 1]
        m_eval = np.linspace(left + 1e-8, right - 1e-8, n_eval)
        errors.append(float(np.max(np.abs(truth_cfunc(m_eval) - sol_cfunc(m_eval)))))
    m_eval = np.linspace(grid_pts[-1] + 1e-8, m_bar, n_eval)
    errors.append(float(np.max(np.abs(truth_cfunc(m_eval) - sol_cfunc(m_eval)))))
    return errors


def main() -> int:
    # `force=True` is needed because HARK's import path installs root logging
    # handlers that would otherwise suppress this script's INFO output.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    truth_sol = _solve(IndShockEGMConsumerType, PARAMS, DENSE_GRID)
    egm_sol = _solve(IndShockEGMConsumerType, PARAMS, SPARSE_GRID)
    mom_sol = _solve(IndShockMoMConsumerType, PARAMS, SPARSE_GRID)

    # The paper's table is computed on the EGM's sparse grid; for an honest
    # comparison the MoM solution must place its evaluation points at the same
    # locations. Assert this rather than silently assume it.
    # `egm_sol.cFunc` is a HARK LinearInterp/CubicInterp with .x_list directly;
    # `mom_sol.cFunc` is TransformedFunctionMoM, which exposes .x_list as a
    # property that back-maps the underlying mu-grid to m-space. MoM by
    # construction adds at least one knot beyond the EGM domain (the
    # extrapolation/cusp point); the natural borrowing constraint at index 0
    # may also differ at the 1e-3 level because the two solvers reach it
    # through slightly different code paths. We therefore require only that
    # EGM's interior gridpoints (indices 1: onwards) appear in MoM's grid.
    try:
        egm_x = np.array(egm_sol.cFunc.x_list)
        mom_x = np.array(mom_sol.cFunc.x_list)
    except AttributeError as exc:
        logger.error(
            "Cannot read x_list from solver output: %s. The MoM consumer must "
            "expose .cFunc.x_list (in m-space) for grid-consistency checking.",
            exc,
        )
        return 2

    egm_interior = egm_x[1:]
    n_interior = len(egm_interior)
    if len(mom_x) < 1 + n_interior or not np.allclose(
        mom_x[1 : 1 + n_interior], egm_interior, rtol=0, atol=1e-9
    ):
        logger.error(
            "EGM interior gridpoints do not match MoM's first %d post-borrowing "
            "knots; intervals are not directly comparable.\n  EGM x_list = %s\n"
            "  MoM x_list = %s",
            n_interior,
            egm_x,
            mom_x,
        )
        return 2

    grid_pts = egm_interior  # Skip the natural borrowing constraint (index 0).
    logger.info("Grid points: %s", grid_pts)

    egm_errors = _interval_max_errors(
        truth_sol.cFunc, egm_sol.cFunc, grid_pts, M_BAR, N_EVAL
    )
    mom_errors = _interval_max_errors(
        truth_sol.cFunc, mom_sol.cFunc, grid_pts, M_BAR, N_EVAL
    )

    def _format(label, errors):
        logger.info("%s max absolute errors:", label)
        for j, err in enumerate(errors):
            if j < len(grid_pts) - 1:
                logger.info("  [m%d, m%d]: %.2e", j, j + 1, err)
            else:
                logger.info("  [m%d, m_bar]: %.2e", j, err)

    _format("EGM", egm_errors)
    _format("MoM", mom_errors)

    regressions = [i for i, (e, m) in enumerate(zip(egm_errors, mom_errors)) if m > e]
    if regressions:
        logger.error(
            "MoM error exceeds EGM error in intervals %s; the central claim of the "
            "paper is violated for this run. Investigate before declaring the table "
            "reproduced.",
            regressions,
        )
        return 1

    logger.info("All intervals: MoM error <= EGM error (consistent with Table 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
