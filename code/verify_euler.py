"""Euler-residual accuracy audit for the Table 1 comparison (EGM vs MoM).

Companion to `verify_table.py`. Where that script measures accuracy directly
against a high-precision reference, this one scores the same candidates with
the discipline's solver-independent instrument, the c-equivalent Euler
equation residual (Judd 1992; Santos 2000 bounds the policy error above by
the residual, Judd-Maliar-Maliar 2017 below), and then validates the
instrument itself before trusting it:

1. FLOOR (resolution): the residual of the 500-point reference solution. Its
   gridpoint residuals are zero by construction, so what remains measures
   interpolation between knots, the smallest difference the instrument can
   resolve here.
2. REJECTION (power): the residual must FIRE on known-bad policies: the
   optimist's perfect-foresight rule (ignores risk entirely) and the sparse
   EGM solution in its extrapolation region (linear extrapolation with the
   wrong slope). An instrument that cannot fail cannot verify.
3. SCORE: sparse EGM vs sparse MoM per interval, mirroring `tbl:approx-errors`.

Because the calibration solves one period back from the analytic terminal
rule, the exact truth is computable at ANY point by a scalar root-find
(`c_true_exact`), with no reference solution and no interpolation. The script
therefore also reports FUNCTIONAL DISTANCE, sup |c - c_exact| per interval,
which is the direct metric free of the residual's curvature weighting: there
the MoM <= EGM ordering holds uniformly, and the 500-point gridded reference
is scored as just another candidate, validating it to ~1e-10. The dense
REFERENCE solves are cubic (see `solve_consumer`) while the sparse
candidates are linear, matching the paper's baseline; whether the residual
metric flips the interval-0 ordering relative to the direct metric depends
on the candidates' interpolation order (it does under cubic candidates, not
under linear), so the interval-0 note in the output computes its figures
from the current run.

The script also reports the far field in GAP units (relative error of the
precautionary gap c_opt - c against the exact truth, plus a sign check for
negative precautionary saving), because at high wealth consumption is mostly
optimist and a consumption-unit metric understates errors in the object the
extrapolation is actually about.

Quadrature note: residuals are evaluated under the solver's own discretized
shock distribution, matching the claim of `tbl:approx-errors`, whose "truth"
lives under the same discretization. All candidates share the exact terminal
rule c_T(m) = m on the right-hand side (the calibration solves one period
back from the terminal, `cycles=1`), so the residual is a pure measure of
each candidate's interpolation and extrapolation error.

Run from the repository root:

    uv run python code/verify_euler.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import brentq

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
from HARK.ConsumptionSaving.ConsIndShockModel import calc_vp_next  # noqa: E402
from HARK.distributions import expected  # noqa: E402
from moderation import (  # noqa: E402
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
)

logger = logging.getLogger("verify_euler")

# Evaluation points per interval for the exact-truth functional-distance
# section: each point is a scalar root-find, so this is deliberately smaller
# than N_EVAL (sup gaps between 200 and 1000 points are negligible here).
N_EXACT = 200


def solve_consumer(consumer_cls, grid=None, *, cubic=False):
    """Solve `consumer_cls` on the Table 1 calibration; return the consumer.

    `cubic=True` is for the dense REFERENCE solves only: the paper's
    candidates are linear (tbl:approx-errors), but nothing forces the
    reference to share the candidates' interpolation order, and cubic
    restores the ~1e-10 reference accuracy the floor argument needs.
    """
    overrides = PARAMS | (DENSE_GRID if grid is None else grid)
    if cubic:
        overrides = overrides | {"CubicBool": True}
    consumer = consumer_cls(**overrides)
    consumer.solve()
    return consumer


class Table1Run(NamedTuple):
    """The four solved policies the Table 1 comparison scores.

    `truth` and `mom_dense` are the dense cubic references; `egm` and `mom`
    are the sparse five-point candidates the paper's table reports.
    """

    truth: Any
    mom_dense: Any
    egm: Any
    mom: Any


def solve_table1_candidates() -> Table1Run:
    """Solve every policy the Table 1 comparison needs, once."""
    return Table1Run(
        truth=solve_consumer(IndShockEGMConsumerType, cubic=True),
        mom_dense=solve_consumer(IndShockMoMConsumerType, cubic=True),
        egm=solve_consumer(IndShockEGMConsumerType, grid=SPARSE_GRID),
        mom=solve_consumer(IndShockMoMConsumerType, grid=SPARSE_GRID),
    )


def sparse_residual_rows(
    run: Table1Run | None = None,
    grid_pts=None,
) -> dict[str, list[float]]:
    """Sup |Euler residual| per interval for the two candidates Table 1 scores.

    Nothing else computes the Euler-residual panel of `tbl:approx-errors`:
    `main` logs these rows and `verify_table.py` imports them for the
    generated fragment, so the panel cannot drift between the audit and the
    table.

    `grid_pts` lets a caller that has already solved the sparse EGM problem
    (`verify_table.py`) supply its own interval edges, so that the two panels
    of the table are demonstrably scored on the same intervals. This function
    checks those edges against its own sparse knots and raises on a mismatch.
    """
    run = solve_table1_candidates() if run is None else run
    kwargs = model_kwargs(run.truth)
    own_pts = np.asarray(run.egm.solution[0].cFunc.x_list)[1:]
    if grid_pts is None:
        grid_pts = own_pts
    elif not np.allclose(grid_pts, own_pts, rtol=0, atol=1e-12):
        msg = (
            f"Caller's interval edges {np.asarray(grid_pts)} differ from this "
            f"run's sparse EGM knots {own_pts}; the residual panel would be "
            "scored on different intervals than the absolute-error panel."
        )
        raise ValueError(msg)
    return {
        name: interval_sup_residuals(cf, grid_pts, M_BAR, N_EVAL, **kwargs)
        for name, cf in (
            ("EGM-5", run.egm.solution[0].cFunc),
            ("MoM-5", run.mom.solution[0].cFunc),
        )
    }


def model_kwargs(consumer):
    """Extract the Euler-operator ingredients from a solved consumer.

    The Table 1 calibration solves one period back from the terminal rule
    (`cycles=1`), so next period's consumption rule is the exact c_T(m) = m
    for every candidate policy; HARK appends it as the last solution element.
    """
    solution = consumer.solution
    cfunc_next = (
        solution[1].cFunc if len(solution) > 1 else consumer.solution_terminal.cFunc
    )
    return {
        "cfunc_next": cfunc_next,
        "IncShkDstn": consumer.IncShkDstn[0],
        "DiscFacEff": consumer.DiscFac * consumer.LivPrb[0],
        "Rfree": consumer.Rfree[0],
        "CRRA": consumer.CRRA,
        "PermGroFac": consumer.PermGroFac[0],
    }


def euler_residual(
    cfunc,
    m,
    *,
    cfunc_next,
    IncShkDstn,
    DiscFacEff,
    Rfree,
    CRRA,
    PermGroFac,
):
    """C-equivalent Euler residual of policy `cfunc` at points `m`.

    Computes r(m) = c*(m)/c(m) - 1, where c*(m) inverts the Euler equation at
    the assets a = m - c(m) implied by the candidate itself:

        u'(c*) = DiscFacEff * R * G^{-rho} * E[ psi^{-rho} u'(c_next(m')) ],
        m' = (R / (G psi)) a + theta.

    The expectation reuses HARK's `calc_vp_next` and the solver's own
    discretized shock distribution, so the operator matches `solve_egm_step`
    factor for factor and a zero residual at the solver's knots is exact.
    Units are relative consumption (Judd's E(m)); the exact policy has
    r identically zero.
    """
    m = np.asarray(m, dtype=float)
    c = np.asarray(cfunc(m))
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        msg = (
            "Candidate policy returned nonpositive or non-finite consumption "
            "on the evaluation grid; the c-equivalent residual is undefined "
            "there."
        )
        raise ValueError(
            msg,
        )
    a = m - c

    def vp_next(m_next):
        # An infeasible plan (next-period consumption <= 0 in some state) has
        # infinite marginal utility there, driving c* to 0 and the residual to
        # its maximal value -1; without the guard numpy returns NaN instead.
        c_next = np.asarray(cfunc_next(m_next))
        c_safe = np.where(c_next > 0.0, c_next, 1.0)
        return np.where(c_next > 0.0, c_safe ** (-CRRA), np.inf)

    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(
        calc_vp_next,
        IncShkDstn,
        args=(a, Rfree, CRRA, PermGroFac, vp_next),
    )
    c_star = np.asarray(EndOfPrdvP) ** (-1.0 / CRRA)
    return c_star / c - 1.0


def interval_sup_residuals(cfunc, grid_pts, m_bar, n_eval, **kwargs):
    """Sup |residual| on each sub-interval plus the extrapolation region."""
    sups = []
    for i in range(len(grid_pts) - 1):
        m_eval = np.linspace(grid_pts[i] + 1e-8, grid_pts[i + 1] - 1e-8, n_eval)
        sups.append(float(np.max(np.abs(euler_residual(cfunc, m_eval, **kwargs)))))
    m_eval = np.linspace(grid_pts[-1] + 1e-8, m_bar, n_eval)
    sups.append(float(np.max(np.abs(euler_residual(cfunc, m_eval, **kwargs)))))
    return sups


def c_true_exact(
    m,
    *,
    mNrmMin,
    cfunc_next,
    IncShkDstn,
    DiscFacEff,
    Rfree,
    CRRA,
    PermGroFac,
):
    """Exact truth at arbitrary points by inverting the one-period operator.

    In the Table 1 calibration next period's rule is the analytic terminal
    c_T(m) = m, so the exact optimal policy at ANY m solves the scalar fixed
    point c = C*(m - c), where C*(a) = (u')^{-1}(EndOfPrdvP(a)) uses only the
    terminal rule and the shock discretization: no reference solution, no
    interpolation. C*(m - c) is strictly decreasing in c (spending more
    leaves fewer assets, raising end-of-period marginal value), so the root
    is unique and bracketed by c -> 0+ (where u' dominates) and the natural
    limit a -> mNrmMin+ (where the worst-shock continuation drives
    EndOfPrdvP to infinity). Accuracy is root-finder tolerance, which is why
    this evaluator collapses the accuracy floor of the gridded reference.
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)

    def c_star_of_a(a):
        def vp_next(m_next):
            c_next = np.asarray(cfunc_next(m_next))
            c_safe = np.where(c_next > 0.0, c_next, 1.0)
            return np.where(c_next > 0.0, c_safe ** (-CRRA), np.inf)

        EndOfPrdvP = vPfacEff * expected(
            calc_vp_next,
            IncShkDstn,
            args=(np.array([a]), Rfree, CRRA, PermGroFac, vp_next),
        )
        return float(np.asarray(EndOfPrdvP).item() ** (-1.0 / CRRA))

    out = np.empty_like(m)
    for i, mi in enumerate(m):

        def fixed_point_gap(c, mi=mi):
            return c_star_of_a(mi - c) - c

        c_hi = mi - mNrmMin  # assets at their natural lower limit
        out[i] = brentq(
            fixed_point_gap,
            1e-12,
            c_hi * (1.0 - 1e-12),
            xtol=1e-13,
        )
    return out


# Anchors for protocol_failures. Relative margins alone are scale-invariant, so
# a uniform units bug would leave them intact; rejections must therefore also
# clear absolute levels, calibrated 10x inside the measured separations.
FLOOR_CEILING = 1e-5
REJECT_ABS_MIN = 0.02
EGM_FAR_ABS_MIN = 1e-3
INTERVAL0_RATIO_CAP = 3.0
INTERVAL0_POWER_MARGIN = 3.0
MOM_GAP_REL_MAX = 0.10
EGM_GAP_REL_MIN = 1.0
GAP_SIGN_MARGIN = 1e-3
REF_DIST_CEILING = 1e-6
EGM_FAR_DIST_MIN = 1e-2
MOM_FAR_DIST_MAX = 1e-2


def protocol_failures(stats):
    """Pure protocol validation: return a list of failure strings (empty = pass).

    `stats` carries the measured summary statistics:

    - "sups": dict of per-interval sup |residual| lists for the candidates
      "floor (truth-500)", "EGM-5", "MoM-5", "optimist", "pessimist";
    - "gap": {"EGM-5"/"MoM-5": {"rel_err", "min"}} for the extrapolation
      region in gap units;
    - "dist": per-interval sup |candidate - exact truth| (FUNCTIONAL
      DISTANCE against the root-find truth) for "truth-500", "MoM-500",
      "EGM-5", "MoM-5";
    - "ratio0": MoM/EGM residual ratio in interval 0;
    - "pess_min_signed": minimum SIGNED pessimist residual over the domain
      (underconsumption must give uniformly positive residuals).

    Every check is phrased fail-closed (`if not good: fail`), so a NaN in any
    statistic is rejected rather than silently passing a comparison.
    """
    sups = stats["sups"]
    n = len(sups["floor (truth-500)"])
    if _non_finite(stats, sups):
        return ["non-finite value in protocol statistics"]
    return (
        _check_floor(sups, n)
        + _check_power(stats, sups, n)
        + _check_score(stats, sups, n)
        + _check_distance(stats["dist"], n)
        + _check_gap(stats["gap"])
    )


def _non_finite(stats, sups) -> bool:
    """True if any protocol statistic is non-finite, in which case nothing below
    it is trustworthy and every later comparison must be skipped.
    """
    all_vals = (
        [v for vals in sups.values() for v in vals]
        + [v for vals in stats["dist"].values() for v in vals]
        + [stats["gap"][k][f] for k in ("EGM-5", "MoM-5") for f in ("rel_err", "min")]
        + [stats["ratio0"], stats["pess_min_signed"]]
    )
    return not np.all(np.isfinite(all_vals))


def _check_floor(sups, n):
    """The instrument's resolution must sit far below what it certifies."""
    floor = sups["floor (truth-500)"]
    return [
        f"floor above ceiling in interval {j}"
        for j in range(n)
        if not floor[j] < FLOOR_CEILING
    ]


def _check_power(stats, sups, n):
    """Rejections must clear relative AND absolute anchors.

    Interval 0 carries a documented smaller relative margin: at the constraint
    edge the residual's curvature amplification inflates the linear candidate's
    own residual (measured separation from the optimist 5.9x), so demanding the
    10x margin there would reject a healthy instrument.
    """
    floor, egm, mom = sups["floor (truth-500)"], sups["EGM-5"], sups["MoM-5"]
    opt, pes = sups["optimist"], sups["pessimist"]
    failures = []
    for j in range(n):
        margin = INTERVAL0_POWER_MARGIN if j == 0 else 10.0
        if not opt[j] > max(REJECT_ABS_MIN, 10 * floor[j], margin * mom[j]):
            failures.append(f"optimist rejection lacks power in interval {j}")
        if not pes[j] > max(REJECT_ABS_MIN, 10 * floor[j]):
            failures.append(f"pessimist rejection lacks power in interval {j}")
    if not stats["pess_min_signed"] > 0:
        failures.append("pessimist residual not uniformly positive (sign power lost)")
    if not egm[-1] > max(EGM_FAR_ABS_MIN, 10 * mom[-1]):
        failures.append("EGM extrapolation not separated from MoM (power failure)")
    return failures


def _check_score(stats, sups, n):
    """MoM <= EGM from interval 1 on; interval 0 within its documented
    curvature-weighting cap (see the note logged by main()).
    """
    egm, mom = sups["EGM-5"], sups["MoM-5"]
    failures = []
    if not stats["ratio0"] <= INTERVAL0_RATIO_CAP:
        failures.append("interval-0 MoM/EGM ratio beyond documented margin")
    failures += [
        f"MoM Euler residual exceeds EGM in interval {j}"
        for j in range(1, n)
        if not mom[j] <= egm[j]
    ]
    return failures


def _check_distance(dist, n):
    """Functional distance against the exact root-find truth.

    The direct metric, free of the residual's curvature weighting. The uniform
    ordering holds in EVERY interval here, including interval 0, and the gridded
    reference must itself sit at its interpolation floor.
    """
    failures = []
    for j in range(n):
        # Both dense candidates must sit at their interpolation floors, far
        # below every table entry; no ordering is asserted between them (their
        # differences live at the exact evaluator's own tolerance).
        for dense in ("truth-500", "MoM-500"):
            if not dist[dense][j] < REF_DIST_CEILING:
                failures.append(f"{dense} drifts from exact truth ({j})")
        if not dist["MoM-5"][j] <= dist["EGM-5"][j]:
            failures.append(f"MoM functional distance exceeds EGM in interval {j}")
    if not dist["EGM-5"][-1] > EGM_FAR_DIST_MIN:
        failures.append("EGM extrapolation distance collapsed; known-bad lost")
    if not dist["MoM-5"][-1] < MOM_FAR_DIST_MAX:
        failures.append("MoM extrapolation distance above its documented level")
    return failures


def _check_gap(gap):
    """EGM must exhibit its known failure IN SIGN, not just size; MoM must stay
    small with positive precautionary saving throughout.
    """
    failures = []
    if not gap["EGM-5"]["rel_err"] > EGM_GAP_REL_MIN:
        failures.append("EGM gap error unexpectedly small; rejection lost power")
    if not gap["EGM-5"]["min"] < -GAP_SIGN_MARGIN:
        failures.append("EGM gap never materially negative; sign rejection lost")
    if not gap["MoM-5"]["rel_err"] <= MOM_GAP_REL_MAX:
        failures.append("MoM relative gap error exceeds 10% in the far field")
    if not gap["MoM-5"]["min"] > 0:
        failures.append("MoM gap goes nonpositive (negative precautionary saving)")
    return failures


def _interval_label(j, n_interior) -> str:
    return f"[m{j}, m{j + 1}]" if j < n_interior - 1 else f"[m{n_interior - 1}, m_bar]"


def _log_panel(title, header, rows) -> None:
    """Log one per-interval table. Both panels share this layout exactly."""
    logger.info("\n%s", title)
    logger.info(header)
    for name, vals in rows.items():
        logger.info(f"  {name:<18}" + "".join(f"{v:14.2e}" for v in vals))


def _exact_truth_grids(grid_pts, n_interior, truth_sol, kwargs):
    """Per-interval evaluation grids and the exact root-find truth on them."""
    exact_m, exact_c = [], []
    for i in range(n_interior):
        left = grid_pts[i]
        right = grid_pts[i + 1] if i < n_interior - 1 else M_BAR
        exact_m.append(np.linspace(left + 1e-6, right - 1e-6, N_EXACT))
        exact_c.append(c_true_exact(exact_m[-1], mNrmMin=truth_sol.mNrmMin, **kwargs))
    return exact_m, exact_c


def _functional_distance(dist_candidates, exact_m, exact_c, n_interior):
    """Sup |candidate - exact truth| per interval.

    The direct metric, free of the residual's curvature weighting and with no
    reference interpolation in the truth itself. The gridded 500-point
    reference is scored as just another candidate; its row IS the old floor.
    """
    return {
        name: [
            float(np.max(np.abs(cf(exact_m[i]) - exact_c[i])))
            for i in range(n_interior)
        ]
        for name, cf in dist_candidates.items()
    }


def _gap_panel(grid_pts, n_interior, mom_sol, candidates, exact_m, exact_c):
    """Score the extrapolation region in gap units, or None if unscoreable.

    At high wealth consumption is mostly optimist, so consumption-unit metrics
    understate errors in the precautionary gap, which is the object the
    extrapolation is about. This also checks the SIGN of implied precautionary
    saving, since EGM's classic failure is a negative gap.
    """
    if not grid_pts[-1] < M_BAR:
        logger.error("Top sparse gridpoint %.3f >= M_BAR %.1f", grid_pts[-1], M_BAR)
        return None
    m_far = exact_m[-1]  # reuse the extrapolation-interval exact-truth grid
    c_opt = mom_sol.Optimist.cFunc(m_far)
    gap_ref = c_opt - exact_c[-1]
    if not (np.all(np.isfinite(gap_ref)) and np.all(gap_ref > 0)):
        logger.error("Reference gap is not finite and positive; cannot score.")
        return None
    logger.info(
        "\nExtrapolation region [m%d, %.0f] in gap units:",
        n_interior - 1,
        M_BAR,
    )
    gap_stats = {}
    for name in ("EGM-5", "MoM-5"):
        gap = c_opt - candidates[name](m_far)
        gap_stats[name] = {
            "rel_err": float(np.max(np.abs(gap - gap_ref) / gap_ref)),
            "min": float(np.min(gap)),
        }
        logger.info(
            "  %s: max relative gap error %.2e; min gap %.3e%s",
            name,
            gap_stats[name]["rel_err"],
            gap_stats[name]["min"],
            " (NEGATIVE precautionary saving)" if gap_stats[name]["min"] < 0 else "",
        )
    return gap_stats


def _log_interval0_note(ratio0, dist_ratio0) -> None:
    """Explain interval 0, which is scored against a documented CAP not EGM.

    The residual multiplies the relative error by the curvature amplification
    1 + dC*/da, which peaks at the constraint edge. Whether that flips the
    interval-0 ordering depends on the candidates' interpolation order, so the
    note computes its figures from this run rather than asserting them.
    """
    if ratio0 > 1.0:
        logger.info(
            "\nInterval 0 note: MoM/EGM Euler-residual ratio = %.2f, yet MoM "
            "is %.1fx closer to the exact truth there by functional distance. "
            "The flip reflects the residual's curvature amplification "
            "1 + dC*/da, largest at the constraint edge where MoM's worst "
            "relative error sits; the Euler metric measures violation of the "
            "Euler equation, not distance from the true policy. See "
            "verify_table.py for the direct comparison.",
            ratio0,
            dist_ratio0,
        )
    else:
        logger.info(
            "\nInterval 0 note: MoM/EGM Euler-residual ratio = %.2f (no flip: "
            "MoM wins interval 0 on the residual metric and is %.1fx closer "
            "by functional distance). The documented cap "
            "(INTERVAL0_RATIO_CAP) remains as a guard against the "
            "curvature-amplified flip that occurs under cubic candidates.",
            ratio0,
            dist_ratio0,
        )


def main() -> int:
    """Run the floor / rejection / score protocol and the gap-unit panel."""
    # `force=True` is needed because HARK's import path installs root logging
    # handlers that would otherwise suppress this script's INFO output.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)

    run = solve_table1_candidates()
    truth_sol = run.truth.solution[0]
    egm_sol = run.egm.solution[0]
    mom_sol = run.mom.solution[0]
    kwargs = model_kwargs(run.truth)

    grid_pts = np.asarray(egm_sol.cFunc.x_list)[1:]  # interior sparse knots
    n_interior = len(grid_pts)
    logger.info("Sparse gridpoints: %s", grid_pts)

    candidates = {
        "floor (truth-500)": truth_sol.cFunc,
        "EGM-5": egm_sol.cFunc,
        "MoM-5": mom_sol.cFunc,
        "optimist": mom_sol.Optimist.cFunc,
        "pessimist": mom_sol.Pessimist.cFunc,
    }
    # The two candidate rows come from `sparse_residual_rows`, the same
    # function `verify_table.py` calls to fill the table's Euler panel.
    sups = {
        "floor (truth-500)": interval_sup_residuals(
            candidates["floor (truth-500)"],
            grid_pts,
            M_BAR,
            N_EVAL,
            **kwargs,
        ),
        **sparse_residual_rows(run),
        **{
            name: interval_sup_residuals(
                candidates[name],
                grid_pts,
                M_BAR,
                N_EVAL,
                **kwargs,
            )
            for name in ("optimist", "pessimist")
        },
    }

    header = "  {:<18}".format("candidate") + "".join(
        f"{_interval_label(j, n_interior):>14}" for j in range(n_interior)
    )
    _log_panel("Sup |Euler residual| by interval (consumption units):", header, sups)

    exact_m, exact_c = _exact_truth_grids(grid_pts, n_interior, truth_sol, kwargs)
    dist = _functional_distance(
        {
            "truth-500": truth_sol.cFunc,
            "MoM-500": run.mom_dense.solution[0].cFunc,
            "EGM-5": egm_sol.cFunc,
            "MoM-5": mom_sol.cFunc,
        },
        exact_m,
        exact_c,
        n_interior,
    )
    _log_panel("Sup |c - c_exact| by interval (functional distance):", header, dist)

    gap_stats = _gap_panel(grid_pts, n_interior, mom_sol, candidates, exact_m, exact_c)
    if gap_stats is None:
        return 2

    ratio0 = sups["MoM-5"][0] / sups["EGM-5"][0]
    _log_interval0_note(ratio0, dist["EGM-5"][0] / dist["MoM-5"][0])

    # Signed pessimist residuals over the whole domain: underconsumption must
    # produce uniformly positive residuals (opposite-sign power).
    m_all = np.linspace(grid_pts[0] + 1e-8, M_BAR, 4 * N_EVAL)
    pess_min_signed = float(
        np.min(euler_residual(candidates["pessimist"], m_all, **kwargs)),
    )

    stats = {
        "sups": sups,
        "gap": gap_stats,
        "dist": dist,
        "ratio0": ratio0,
        "pess_min_signed": pess_min_signed,
    }
    failures = protocol_failures(stats)
    if failures:
        for f in failures:
            logger.error("FAIL: %s", f)
        return 1

    logger.info(
        "\nAll checks passed: instrument has power (floor, rejections, and "
        "absolute anchors all separated), MoM Euler residuals <= EGM from "
        "interval 1 on with interval 0 inside its documented ratio cap, and "
        "the far-field gap error is small with strictly positive "
        "precautionary saving.",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
