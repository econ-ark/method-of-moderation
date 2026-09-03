"""Robustness experiments extending Table 1 (`tbl:approx-errors`).

The paper's headline accuracy ratios come from a single T-1 cell: five
`aXtra` gridpoints, linear spacing, naive EGM as the comparison. A referee
can reasonably ask two questions about that cell.

Experiment A (production grid): does MoM's advantage survive when the grid
is no longer deliberately sparse? Same T-1 problem and the same
interval-wise max-abs-error protocol, but with `aXtraCount=20` and
exponential nesting `aXtraNestFac=3`.

Experiment B (bound-aware EGM): how much of MoM's advantage is just the
information in the analytical bounds? We hand naive EGM those bounds for
free as a pointwise clamp, `c = clip(cEGM, cPes, cOpt)`, and re-run the
same error protocol. A second, strictly tighter clamp additionally uses
the MPCmax bound (`cTight`). The clamp is the weakest possible bound-aware
competitor by construction: any smoother use of the same two bounds is
already the Method of Moderation.

Candidate solvers run with `CubicBool=False`, matching the paper's move to
a linear-interpolation baseline. The reference solution ("truth") is
instead a cubic solve on 16000 gridpoints, which is converged to roughly
1e-12; at T-1 the terminal rule c(m) = m is exact and every candidate
shares the same 7-point quadrature, so interpolation is the only source of
error and the reference is exact for practical purposes.

Run from the repository root:

    uv run python code/explorations/horserace/run_horserace.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Make `code/` importable regardless of caller CWD, without leaking a relative
# entry into sys.path.
_THIS_DIR = Path(__file__).resolve().parent
_CODE_DIR = _THIS_DIR.parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from moderation import IndShockEGMConsumerType, IndShockMoMConsumerType  # noqa: E402

# verify_table.py is the single source of truth for the Table 1 calibration.
from verify_table import PARAMS  # noqa: E402

logger = logging.getLogger("run_horserace")


# The Table 1 calibration, with the paper's new linear-interpolation baseline.
# Must extend past max(HORIZONS), or the sweep measures the reference
# solution's own extrapolation rather than the candidate's (asserted in main).
TRUTH_GRID = {
    "aXtraMin": 0.001,
    "aXtraMax": 420,
    "aXtraCount": 16000,
    "aXtraNestFac": 3,
}
TRUTH_CHECK_GRID = {
    "aXtraMin": 0.001,
    "aXtraMax": 400,
    "aXtraCount": 4000,
    "aXtraNestFac": 3,
}
SPARSE_GRID = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 5, "aXtraNestFac": -1}
PROD_GRID = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 20, "aXtraNestFac": 3}
M_BAR = 30.0
N_EVAL = 1000
HORIZONS = [10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 400.0]

# A measured error resolves the candidate only this far above the reference
# grid's own convergence error. Cells within FLOOR_MARGIN are logged and stored;
# cells within FLOOR_HARD_MARGIN mean the number IS the reference, failing run.
FLOOR_MARGIN = 100.0
FLOOR_HARD_MARGIN = 10.0

_OUT_JSON = _THIS_DIR / "horserace_results.json"


class ClampedCFunc:
    """An EGM consumption function clipped pointwise to analytical bounds.

    This hands naive EGM the Method of Moderation's information for free:
    the realist's consumption is known to satisfy c_pes(m) <= c(m) <= c_opt(m),
    so any EGM value outside that band can be projected back onto it at zero
    modelling cost. It is the natural control for how much of MoM's advantage
    is the bounds rather than the moderation transformation itself.
    """

    def __init__(self, cfunc, lower_funcs, upper_funcs) -> None:
        self.cfunc = cfunc
        self.lower_funcs = list(lower_funcs)
        self.upper_funcs = list(upper_funcs)

    def _band(self, m):
        lower = np.max([f(m) for f in self.lower_funcs], axis=0)
        upper = np.min([f(m) for f in self.upper_funcs], axis=0)
        # np.clip with inverted or NaN limits silently returns garbage, so the
        # band must be validated before any clamping is trusted.
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            msg = "non-finite bound in clamp band"
            raise RuntimeError(msg)
        if not np.all(lower <= upper):
            msg = "empty clamp band: a lower bound exceeds an upper"
            raise RuntimeError(msg)
        return lower, upper

    def __call__(self, m):
        lower, upper = self._band(m)
        return np.clip(self.cfunc(m), lower, upper)

    def binding(self, m):
        """Boolean masks (lower_binds, upper_binds) at the evaluation points."""
        raw = self.cfunc(m)
        lower, upper = self._band(m)
        return raw < lower, raw > upper


def _solve(consumer_cls, params, grid):
    consumer = consumer_cls(**(params | grid))
    consumer.solve()
    return consumer.solution[0]


def _interval_edges(grid_pts, m_bar):
    """Evaluation windows: each grid interval, then the extrapolation region."""
    edges = [(grid_pts[i], grid_pts[i + 1]) for i in range(len(grid_pts) - 1)]
    edges.append((grid_pts[-1], m_bar))
    return edges


def _interval_label(j, grid_pts) -> str:
    return f"[m{j}, m{j + 1}]" if j < len(grid_pts) - 1 else f"[m{j}, m_bar]"


def _interval_max_errors(truth_cfunc, sol_cfunc, grid_pts, m_bar, n_eval):
    """Maximum |truth - approx| on each sub-interval plus the extrapolation region."""
    errors = []
    for left, right in _interval_edges(grid_pts, m_bar):
        m_eval = np.linspace(left + 1e-8, right - 1e-8, n_eval)
        errors.append(float(np.max(np.abs(truth_cfunc(m_eval) - sol_cfunc(m_eval)))))
    return errors


def _grid_points(egm_sol, mom_sol):
    """EGM's interior gridpoints, checked to coincide with MoM's knots.

    MoM adds knots beyond the EGM domain by construction, and index 0 (the
    natural borrowing constraint) can differ at the 1e-3 level because the
    two solvers reach it by different code paths; so only EGM's interior
    points are required to appear in MoM's grid.
    """
    egm_x = np.array(egm_sol.cFunc.x_list)
    mom_x = np.array(mom_sol.cFunc.x_list)
    egm_interior = egm_x[1:]
    n_interior = len(egm_interior)
    if len(mom_x) < 1 + n_interior or not np.allclose(
        mom_x[1 : 1 + n_interior],
        egm_interior,
        rtol=0,
        atol=1e-9,
    ):
        msg = (
            f"EGM interior gridpoints do not match MoM's first {n_interior} "
            f"post-borrowing knots; intervals are not comparable.\n"
            f"  EGM x_list = {egm_x}\n  MoM x_list = {mom_x}"
        )
        raise RuntimeError(msg)
    return egm_interior


def _log_table(label, grid_pts, columns) -> None:
    """Log an interval-wise error table. `columns` maps a name to an error list."""
    names = list(columns)
    logger.info("")
    logger.info("%s", label)
    logger.info("  %-16s%s", "interval", "".join(f"{n:>14s}" for n in names))
    n_rows = len(next(iter(columns.values())))
    for j in range(n_rows):
        cells = "".join(f"{columns[n][j]:14.3e}" for n in names)
        logger.info("  %-16s%s", _interval_label(j, grid_pts), cells)
    for tag, rows in (("tabulated max", slice(0, -1)), ("overall max", slice(None))):
        cells = "".join(f"{max(columns[n][rows]):14.3e}" for n in names)
        logger.info("  %-16s%s", tag, cells)


def _log_ratios(label, grid_pts, numerator, denominator, num_name, den_name, floor):
    """Log numerator/denominator per interval, flagging floor-contaminated cells."""
    logger.info("")
    logger.info("%s (%s / %s)", label, num_name, den_name)
    ratios = []
    for j, (num, den) in enumerate(zip(numerator, denominator, strict=True)):
        if not (np.isfinite(num) and np.isfinite(den) and den > 0):
            msg = f"cannot form ratio from num={num!r} den={den!r}"
            raise RuntimeError(msg)
        ratio = float(num / den)
        ratios.append(ratio)
        flag = f"  (denominator within {FLOOR_MARGIN:.0f}x of reference error)"
        note = flag if den < FLOOR_MARGIN * floor[j] else ""
        logger.info("  %-16s %10.2f%s", _interval_label(j, grid_pts), ratio, note)
    return ratios


def _truth_floor(truth_cfunc, params, grid_pts, m_bar, n_eval):
    """Interval-wise gap between the reference solution and a coarser control.

    Any reported error near this magnitude is measuring the reference grid,
    not the candidate solution.
    """
    control = _solve(
        IndShockEGMConsumerType,
        params | {"CubicBool": True},
        TRUTH_CHECK_GRID,
    )
    return _interval_max_errors(truth_cfunc, control.cFunc, grid_pts, m_bar, n_eval)


def _binding_report(clamped, grid_pts, m_bar, n_eval):
    """Fraction of evaluation points per interval where each clamp side binds."""
    rows = []
    for j, (left, right) in enumerate(_interval_edges(grid_pts, m_bar)):
        m_eval = np.linspace(left + 1e-8, right - 1e-8, n_eval)
        lower_binds, upper_binds = clamped.binding(m_eval)
        rows.append(
            {
                "interval": _interval_label(j, grid_pts),
                "lower_frac": float(np.mean(lower_binds)),
                "upper_frac": float(np.mean(upper_binds)),
            },
        )
    return rows


def _band_widths(truth_cfunc, mom_sol, grid_pts, m_bar, n_eval):
    """Per-interval distance from the truth to each analytical bound.

    Once the clamp binds, the clamped solution's error is the bound's own
    gap, so these numbers are the floor on any clamp-based method.
    """
    return {
        "pes": _interval_max_errors(
            truth_cfunc,
            mom_sol.Pessimist.cFunc,
            grid_pts,
            m_bar,
            n_eval,
        ),
        "opt": _interval_max_errors(
            truth_cfunc,
            mom_sol.Optimist.cFunc,
            grid_pts,
            m_bar,
            n_eval,
        ),
    }


def _log_binding(binding, binding_tight, bands) -> None:
    """Where each clamp binds, beside the truth-to-bound distance it implies."""
    logger.info("")
    logger.info("Where the clamp binds (fraction of evaluation points), and the")
    logger.info("distance from the truth to each bound (the clamp's own error floor)")
    logger.info(
        "  %-16s %10s %10s %10s %13s %13s",
        "interval",
        "pes (lo)",
        "opt (hi)",
        "tight (hi)",
        "|truth-pes|",
        "|truth-opt|",
    )
    for j, (row, row_t) in enumerate(zip(binding, binding_tight, strict=True)):
        logger.info(
            "  %-16s %10.3f %10.3f %10.3f %13.3e %13.3e",
            row["interval"],
            row["lower_frac"],
            row["upper_frac"],
            row_t["upper_frac"],
            bands["pes"][j],
            bands["opt"][j],
        )


def _floor_contamination(candidate_errors, floor, grid_pts):
    """Flag tabulated cells sitting too close to the reference floor.

    Runs over every non-extrapolation cell of every candidate. Soft flags are
    recorded for the log; hard flags fail main, since there the reported number
    is measuring the reference grid rather than the candidate.
    """
    soft_flags, hard_flags = [], []
    for cand_name, errors in candidate_errors.items():
        for j in range(len(errors) - 1):
            if not errors[j] >= FLOOR_HARD_MARGIN * floor[j]:
                hard_flags.append(f"{cand_name} {_interval_label(j, grid_pts)}")
            elif not errors[j] >= FLOOR_MARGIN * floor[j]:
                soft_flags.append(f"{cand_name} {_interval_label(j, grid_pts)}")
    return soft_flags, hard_flags


def _run_case(name, params, grid, truth_sol):
    """Solve EGM, clamped EGM and MoM on one grid; tabulate against the truth."""
    egm_sol = _solve(IndShockEGMConsumerType, params, grid)
    mom_sol = _solve(IndShockMoMConsumerType, params, grid)
    grid_pts = _grid_points(egm_sol, mom_sol)

    c_pes = mom_sol.Pessimist.cFunc
    c_opt = mom_sol.Optimist.cFunc
    c_tight = mom_sol.TighterUpperBound.cFunc
    clamped = ClampedCFunc(egm_sol.cFunc, [c_pes], [c_opt])
    clamped_tight = ClampedCFunc(egm_sol.cFunc, [c_pes], [c_opt, c_tight])

    def errs(func):
        return _interval_max_errors(truth_sol.cFunc, func, grid_pts, M_BAR, N_EVAL)

    egm_err = errs(egm_sol.cFunc)
    mom_err = errs(mom_sol.cFunc)
    clamp_err = errs(clamped)
    tight_err = errs(clamped_tight)
    floor = _truth_floor(truth_sol.cFunc, params, grid_pts, M_BAR, N_EVAL)

    logger.info("")
    logger.info("=" * 78)
    logger.info("%s", name)
    logger.info("grid: %s", grid)
    logger.info("=" * 78)
    logger.info("Grid points (m-space): %s", np.array2string(grid_pts, precision=4))
    _log_table(
        "Max absolute consumption error",
        grid_pts,
        {
            "EGM": egm_err,
            "EGM+clamp": clamp_err,
            "EGM+tight": tight_err,
            "MoM": mom_err,
            "ref. error": floor,
        },
    )
    ratios_egm = _log_ratios("Ratios", grid_pts, egm_err, mom_err, "EGM", "MoM", floor)
    ratios_clamp = _log_ratios(
        "Ratios",
        grid_pts,
        clamp_err,
        mom_err,
        "EGM+clamp",
        "MoM",
        floor,
    )

    binding = _binding_report(clamped, grid_pts, M_BAR, N_EVAL)
    binding_tight = _binding_report(clamped_tight, grid_pts, M_BAR, N_EVAL)
    bands = _band_widths(truth_sol.cFunc, mom_sol, grid_pts, M_BAR, N_EVAL)

    _log_binding(binding, binding_tight, bands)

    soft_flags, hard_flags = _floor_contamination(
        {
            "EGM": egm_err,
            "EGM+clamp": clamp_err,
            "EGM+tight": tight_err,
            "MoM": mom_err,
        },
        floor,
        grid_pts,
    )
    if soft_flags:
        logger.warning(
            "Cells within %.0fx of the reference floor (absolute magnitudes "
            "partially measure the reference): %s",
            FLOOR_MARGIN,
            soft_flags,
        )
    if hard_flags:
        logger.error(
            "Cells within %.0fx of the reference floor (the number IS the "
            "reference grid): %s",
            FLOOR_HARD_MARGIN,
            hard_flags,
        )

    not_worse = [
        _interval_label(j, grid_pts)
        for j in range(len(mom_err))
        if clamp_err[j] <= mom_err[j]
    ]
    logger.info("")
    if not_worse:
        logger.info("Intervals where clamped EGM ties or beats MoM: %s", not_worse)
    else:
        logger.info("MoM is strictly more accurate than clamped EGM in every interval.")

    return {
        "grid": grid,
        "grid_points": grid_pts.tolist(),
        "egm_errors": egm_err,
        "clamped_errors": clamp_err,
        "clamped_tight_errors": tight_err,
        "mom_errors": mom_err,
        "reference_error": floor,
        "ratios_egm_over_mom": ratios_egm,
        "ratios_clamped_over_mom": ratios_clamp,
        "interior_mean_ratio_egm_over_mom": float(np.mean(ratios_egm[:-1])),
        "interior_mean_ratio_clamped_over_mom": float(np.mean(ratios_clamp[:-1])),
        "floor_soft_flags": soft_flags,
        "floor_hard_flags": hard_flags,
        "egm_tabulated_max": max(egm_err[:-1]),
        "clamped_tabulated_max": max(clamp_err[:-1]),
        "mom_tabulated_max": max(mom_err[:-1]),
        "egm_extrapolation": egm_err[-1],
        "clamped_extrapolation": clamp_err[-1],
        "clamped_tight_extrapolation": tight_err[-1],
        "mom_extrapolation": mom_err[-1],
        "binding_pes_opt": binding,
        "binding_with_tight": binding_tight,
        "band_widths": bands,
        "intervals_clamp_not_worse_than_mom": not_worse,
    }


def _run_horizon_sweep(truth_sol, params, grid):
    """Experiment C: extrapolation error as the evaluation horizon extends.

    The single m_bar = 30 snapshot in Table 1 conflates two different
    behaviours. Naive EGM diverges without bound, so its ratio against MoM
    is whatever the horizon is chosen to be. Clamped EGM and MoM both
    saturate, so the ratio between those two converges to a number that
    actually characterises the methods.
    """
    egm_sol = _solve(IndShockEGMConsumerType, params, grid)
    mom_sol = _solve(IndShockMoMConsumerType, params, grid)
    m_top = float(np.array(egm_sol.cFunc.x_list)[-1])
    clamped = ClampedCFunc(
        egm_sol.cFunc,
        [mom_sol.Pessimist.cFunc],
        [mom_sol.Optimist.cFunc],
    )

    logger.info("")
    logger.info("=" * 78)
    logger.info("Experiment C: extrapolation horizon sweep on the Table 1 grid")
    logger.info("region [m_top = %.4f, m_bar]", m_top)
    logger.info("=" * 78)
    logger.info(
        "  %8s %13s %13s %13s %10s %10s",
        "m_bar",
        "EGM",
        "EGM+clamp",
        "MoM",
        "EGM/MoM",
        "clamp/MoM",
    )
    rows = []
    for m_bar in HORIZONS:
        m_eval = np.linspace(m_top + 1e-8, m_bar, 20 * N_EVAL)
        truth = truth_sol.cFunc(m_eval)
        egm_e = float(np.max(np.abs(truth - egm_sol.cFunc(m_eval))))
        clamp_e = float(np.max(np.abs(truth - clamped(m_eval))))
        mom_e = float(np.max(np.abs(truth - mom_sol.cFunc(m_eval))))
        logger.info(
            "  %8.0f %13.4e %13.4e %13.4e %10.1f %10.1f",
            m_bar,
            egm_e,
            clamp_e,
            mom_e,
            egm_e / mom_e,
            clamp_e / mom_e,
        )
        rows.append(
            {
                "m_bar": m_bar,
                "egm": egm_e,
                "clamped": clamp_e,
                "mom": mom_e,
                "ratio_egm_over_mom": egm_e / mom_e,
                "ratio_clamped_over_mom": clamp_e / mom_e,
            },
        )
    return {"m_top": m_top, "rows": rows}


def main() -> int:
    # `force=True` is needed because HARK's import path installs root logging
    # handlers that would otherwise suppress this script's INFO output.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    if not max(HORIZONS) < TRUTH_GRID["aXtraMax"]:
        msg = (
            "TRUTH_GRID must extend past max(HORIZONS) or the sweep measures "
            "the reference solution's own extrapolation"
        )
        raise RuntimeError(
            msg,
        )
    truth_sol = _solve(
        IndShockEGMConsumerType,
        PARAMS | {"CubicBool": True},
        TRUTH_GRID,
    )

    anchor = _run_case(
        "Sanity anchor + Experiment B: Table 1 grid (5 points, linear spacing)",
        PARAMS,
        SPARSE_GRID,
        truth_sol,
    )
    experiment_a = _run_case(
        "Experiment A: production grid (20 points, nest factor 3)",
        PARAMS,
        PROD_GRID,
        truth_sol,
    )

    horizon = _run_horizon_sweep(truth_sol, PARAMS, SPARSE_GRID)

    results = {
        "params": PARAMS,
        "m_bar": M_BAR,
        "n_eval": N_EVAL,
        "truth_grid": TRUTH_GRID,
        "truth_check_grid": TRUTH_CHECK_GRID,
        "sparse_grid_case": anchor,
        "production_grid_case": experiment_a,
        "horizon_sweep": horizon,
    }

    def _all_finite(node):
        if isinstance(node, dict):
            return all(_all_finite(v) for v in node.values())
        if isinstance(node, list):
            return all(_all_finite(v) for v in node)
        if isinstance(node, float):
            return np.isfinite(node)
        return True

    if not _all_finite(results):
        logger.error("Non-finite value in results; refusing to write JSON.")
        return 2
    _OUT_JSON.write_text(json.dumps(results, indent=2))
    logger.info("")
    logger.info("Wrote %s", _OUT_JSON)
    hard = anchor["floor_hard_flags"] + experiment_a["floor_hard_flags"]
    if hard:
        logger.error("Floor-contaminated cells present: %s", hard)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
