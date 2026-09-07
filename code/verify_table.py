"""Verify Table 1 approximation errors (EGM vs MoM).

Reproduces `tbl:approx-errors` from `content/paper/moderation_extended.md`.
Computes the maximum absolute difference between a high-precision EGM
solution ("truth") and each of the sparse EGM and MoM solutions on each
sub-interval of the sparse grid, plus the extrapolation region. Exits
non-zero if any MoM interval error is larger than the corresponding EGM
error, since the paper's central claim is that MoM is uniformly more
accurate.

It is also the generator of the two fragments the manuscript includes,
`content/tables/approx-errors.md` and `content/tables/approx-errors-cubic.md`,
so the printed table cannot drift from the solver.

Run from the repository root:

    uv run python code/verify_table.py            # linear, report only
    uv run python code/verify_table.py --cubic    # Hermite refinement, report
    uv run python code/verify_table.py --list     # the table names it owns
    uv run python code/verify_table.py --write    # regenerate both fragments
    uv run python code/verify_table.py --check    # CI gate on both fragments
    uv run python code/verify_table.py --write --only approx-errors-cubic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make `code/` importable regardless of caller CWD, without leaking a relative
# entry into sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from calibration import (  # noqa: E402
    DENSE_GRID,
    M_BAR,
    N_EVAL,
    PARAMS,
    SPARSE_GRID,
)
from moderation import IndShockEGMConsumerType, IndShockMoMConsumerType  # noqa: E402
from table_io import add_table_args, emit, resolve_names, sci  # noqa: E402
from verify_euler import sparse_residual_rows  # noqa: E402

logger = logging.getLogger("verify_table")

# The generated fragments this script owns, in `content/tables/`. A name
# ending in `-cubic` is the Hermite refinement of the one before it.
TABLES = ("approx-errors", "approx-errors-cubic")


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


def _interval_headers(grid_pts) -> list[str]:
    """Column headers: one per sub-interval, then the extrapolation region."""
    last = len(grid_pts) - 1
    return [f"$[m_{j},m_{j + 1}]$" for j in range(last)] + [
        rf"$[m_{last},\overline{{m}}]$",
    ]


def _error_row(label: str, errors) -> str:
    return f"{label} & " + " & ".join(sci(e) for e in errors) + r" \\"


def _panel_heading(n_cols: int, title: str) -> str:
    return rf"\multicolumn{{{n_cols}}}{{l}}{{\emph{{{title}}}}} \\"


def _render_table1(grid_pts, egm_errors, mom_errors, euler_rows) -> str:
    """The linear-interpolation table, `tbl:approx-errors`."""
    points = ", ".join(f"$m_{j}={m:.2f}$" for j, m in enumerate(grid_pts))
    caption = (
        "Maximum approximation errors by interval for endogenous gridpoints "
        "(EGM) and the method of moderation (MoM) on the same five-point "
        "grid with linear interpolation: absolute consumption errors against "
        "the reference solution (upper panel) and the largest Euler-equation "
        "residual in consumption-equivalent units, which needs no reference "
        f"solution (lower panel). Gridpoints {points}; evaluation horizon "
        rf"$\overline{{m}}={M_BAR:g}$. Orders of magnitude in parentheses."
    )
    n_cols = 1 + len(grid_pts)
    return "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tbl:approx-errors}",
            r"\setlength{\tabcolsep}{8pt}",
            rf"\begin{{tabular}}{{l{'c' * len(grid_pts)}}}",
            r"\toprule",
            "Method & " + " & ".join(_interval_headers(grid_pts)) + r" \\",
            r"\midrule",
            _panel_heading(n_cols, "Absolute error"),
            _error_row("EGM", egm_errors),
            _error_row("MoM", mom_errors),
            r"\midrule",
            _panel_heading(n_cols, "Euler residual"),
            _error_row("EGM", euler_rows["EGM-5"]),
            _error_row("MoM", euler_rows["MoM-5"]),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )


def _render_table1_cubic(grid_pts, egm_errors, mom_errors) -> str:
    """The Hermite-refinement table, `tbl:approx-errors-cubic`."""
    caption = (
        "Maximum absolute approximation errors by interval, as in "
        # This caption is emitted straight into the LaTeX table, so a MyST
        # role here would pass through verbatim; use a plain \ref.
        r"Table~\ref{tbl:approx-errors} but with cubic Hermite "
        "interpolation in place of linear for both methods. Orders of "
        "magnitude in parentheses."
    )
    return "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tbl:approx-errors-cubic}",
            r"\setlength{\tabcolsep}{8pt}",
            rf"\begin{{tabular}}{{l{'c' * len(grid_pts)}}}",
            r"\toprule",
            "Method & " + " & ".join(_interval_headers(grid_pts)) + r" \\",
            r"\midrule",
            _error_row("EGM", egm_errors),
            _error_row("MoM", mom_errors),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )


def _generate(name: str, *, check_only: bool) -> int:
    """Write or check one of the two fragments this script owns.

    The table's name fixes the interpolation order it reports, so the run and
    the artifact cannot disagree. The Euler panel of the linear table comes
    from `verify_euler.py`, so both panels of `tbl:approx-errors` and the
    Euler audit report one set of numbers computed in one place.
    """
    cubic = name.endswith("-cubic")
    status, grid_pts, egm_errors, mom_errors = _verify_pass(cubic=cubic)
    if status:
        return status
    if cubic:
        return emit(
            name,
            _render_table1_cubic(grid_pts, egm_errors, mom_errors),
            script="code/verify_table.py",
            command=f"uv run python code/verify_table.py --write --only {name}",
            check_only=check_only,
        )
    logger.info("Computing the Euler-residual panel (verify_euler.py)...")
    euler_rows = sparse_residual_rows(grid_pts=grid_pts)
    return emit(
        name,
        _render_table1(grid_pts, egm_errors, mom_errors, euler_rows),
        script="code/verify_table.py",
        command=f"uv run python code/verify_table.py --write --only {name}",
        check_only=check_only,
    )


def _verify_pass(cubic: bool = False, perm_shk: bool = False):
    """Solve, log the per-interval errors, and score moderation against EGM.

    Returns `(status, grid_pts, egm_errors, mom_errors)`, where status is 0
    when the paper's ordering holds, 1 when moderation loses an interval, and
    2 when the run is unusable; the other three are None in that last case.
    """
    # `cubic` reproduces the Hermite-refinement numbers quoted in the paper's
    # Hermite section (run as `python verify_table.py cubic`); the default
    # linear run reproduces tbl:approx-errors.
    params = PARAMS | {"CubicBool": True} if cubic else PARAMS
    # `perm_shk` substantiates the model section's claim that the results do not
    # depend on shutting permanent shocks off; PARAMS zeroes PermShkStd for
    # exposition, so nothing else exercises this calibration with them on.
    if perm_shk:
        params = params | {"PermShkStd": [0.1], "PermShkCount": 5}
    logger.info(
        "Interpolation: %s",
        "cubic Hermite (refinement)" if cubic else "linear",
    )
    logger.info("Permanent shocks: %s", "on (std 0.1)" if perm_shk else "off")
    truth_sol = _solve(IndShockEGMConsumerType, params, DENSE_GRID)
    egm_sol = _solve(IndShockEGMConsumerType, params, SPARSE_GRID)
    mom_sol = _solve(IndShockMoMConsumerType, params, SPARSE_GRID)

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
        logger.exception(
            "Cannot read x_list from solver output: %s. The MoM consumer must "
            "expose .cFunc.x_list (in m-space) for grid-consistency checking.",
            exc,
        )
        return 2, None, None, None

    egm_interior = egm_x[1:]
    n_interior = len(egm_interior)
    if len(mom_x) < 1 + n_interior or not np.allclose(
        mom_x[1 : 1 + n_interior],
        egm_interior,
        rtol=0,
        atol=1e-9,
    ):
        logger.error(
            "EGM interior gridpoints do not match MoM's first %d post-borrowing "
            "knots; intervals are not directly comparable.\n  EGM x_list = %s\n"
            "  MoM x_list = %s",
            n_interior,
            egm_x,
            mom_x,
        )
        return 2, None, None, None

    grid_pts = egm_interior  # Skip the natural borrowing constraint (index 0).
    logger.info("Grid points: %s", grid_pts)

    egm_errors = _interval_max_errors(
        truth_sol.cFunc,
        egm_sol.cFunc,
        grid_pts,
        M_BAR,
        N_EVAL,
    )
    mom_errors = _interval_max_errors(
        truth_sol.cFunc,
        mom_sol.cFunc,
        grid_pts,
        M_BAR,
        N_EVAL,
    )

    def _format(label, errors) -> None:
        logger.info("%s max absolute errors:", label)
        for j, err in enumerate(errors):
            if j < len(grid_pts) - 1:
                logger.info("  [m%d, m%d]: %.2e", j, j + 1, err)
            else:
                logger.info("  [m%d, m_bar]: %.2e", j, err)

    _format("EGM", egm_errors)
    _format("MoM", mom_errors)

    # Fail closed: NaN passes any `m > e` comparison, so gate finiteness
    # first and phrase the regression test as `not m <= e`.
    all_errors = np.array(egm_errors + mom_errors)
    if not np.all(np.isfinite(all_errors)):
        logger.error(
            "Non-finite interval error; cannot score. EGM=%s MoM=%s",
            egm_errors,
            mom_errors,
        )
        return 2, None, None, None

    regressions = [
        i
        for i, (e, m) in enumerate(zip(egm_errors, mom_errors, strict=True))
        if not m <= e
    ]
    if regressions:
        logger.error(
            "MoM error exceeds EGM error in intervals %s; the central claim of the "
            "paper is violated for this run. Investigate before declaring the table "
            "reproduced.",
            regressions,
        )
        return 1, grid_pts, egm_errors, mom_errors

    logger.info("All intervals: MoM error <= EGM error (consistent with Table 1).")
    return 0, grid_pts, egm_errors, mom_errors


def main(
    cubic: bool = False,
    perm_shk: bool = False,
    write: bool = False,
    check: bool = False,
    only=None,
    list_names: bool = False,
) -> int:
    # `force=True` is needed because HARK's import path installs root logging
    # handlers that would otherwise suppress this script's INFO output.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    if list_names:
        for name in TABLES:
            logger.info(name)
        return 0

    if not (write or check):
        return _verify_pass(cubic=cubic, perm_shk=perm_shk)[0]

    if perm_shk:
        logger.error(
            "The generated tables report the calibration with permanent shocks "
            "off; --write and --check are meaningless under --permshk.",
        )
        return 2
    try:
        names = resolve_names(only, TABLES)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    statuses = [_generate(name, check_only=check) for name in names]
    return max(statuses, default=0)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the approximation-error tables of the paper and generate "
            "the fragments the manuscript includes."
        ),
    )
    parser.add_argument(
        "--cubic",
        action="store_true",
        help=(
            "Report the cubic Hermite refinement instead of linear "
            "interpolation (ignored by --write and --check, where the table "
            "name fixes the interpolation order)."
        ),
    )
    parser.add_argument(
        "--permshk",
        dest="perm_shk",
        action="store_true",
        help="Report the calibration with permanent shocks on (std 0.1).",
    )
    add_table_args(parser, TABLES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(**vars(_parse_args())))
