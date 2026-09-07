r"""Grid design and beyond-grid extrapolation accuracy: Chris's proposal, tested arm by arm.

Chris's 2026-08-14 prescription, verbatim structure: the paper's linear
"0,1,2,3,4" grid "doesn't honestly work very well"; the fix is "double or
triple exponential gridpoints, up to a ratio of a to h ... at least a ratio
of 2 times human wealth". Each clause is an arm here:

- Baseline arm: the paper's 5-point linear sparse grid (a up to 4).
- Spacing arms: double exponential (aXtraNestFac 2) and triple exponential
  (aXtraNestFac 3), separately.
- Extent dial: grid top a in {h/2, h, 2h, 4h}, reported as a/h (Chris's
  ratio), bracketing his "at least 2".

Deterministic MoM, paper calibration, infinite horizon (h = 50; the extent
question only exists for converged solutions - at T-1, h is about 1 and the
paper's grid already tops out past 4h). All solves at tolerance 1e-10.
Truth: aXtraMax 2000 with 600 points. Metric: the paper's standard accuracy
measure, max |ctilde - c| against the dense truth, evaluated beyond the grid
top on m up to 1000. The bounds hold on every arm by construction (the
robustness Chris asks for is unconditional in MoM); extent buys accuracy.

Run from the repo root:

    PYTHONPATH=code uv run python code/explorations/grid-extent/run_grid_extent.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from moderation import IndShockMoMConsumerType

LOG = logging.getLogger("grid-extent")

HERE = Path(__file__).resolve().parent

PARAMS = {
    "CRRA": 2.0,
    "DiscFac": 0.96,
    "Rfree": [1.02],
    "TranShkStd": [1.0],
    "cycles": 0,
    "LivPrb": [1.0],
    "vFuncBool": False,
    # Linear interpolation matches the paper's baseline (tbl:approx-errors).
    "CubicBool": False,
    "PermGroFac": [1.0],
    "PermShkStd": [0.0],
    "TranShkCount": 7,
    "UnempPrb": 0.0,
    "BoroCnstArt": None,
    "aXtraMin": 0.001,
    "tolerance": 1e-10,
}
TRUTH_GRID = {"aXtraMax": 2000.0, "aXtraCount": 600, "aXtraNestFac": 3}
A_OVER_H = [0.5, 1.0, 2.0, 4.0]
NEST_FACS = [2, 3]
SPARSE_COUNT = 30
M_EVAL_TOP = 1000.0
N_EVAL = 60


def solve(grid_overrides):
    agent = IndShockMoMConsumerType(**(PARAMS | grid_overrides))
    agent.solve()
    # HARK stops silently at max_cycles; phrased fail-closed so NaN distances
    # cannot pass a `distance > tolerance` comparison.
    if agent.cycles == 0 and not agent.solution_distance <= agent.tolerance:
        msg = (
            f"solve did not converge: distance {agent.solution_distance:.3e} "
            f"> tolerance {agent.tolerance:.3e} "
            f"after {agent.completed_cycles} cycles"
        )
        raise RuntimeError(
            msg,
        )
    return agent.solution[0]


def score(sol, truth):
    # x_list carries one synthetic extrapolation knot at each end (see
    # _construct_mom_interpolants); the true grid top is the last REAL knot.
    mu_k = np.asarray(sol.cFunc.logitModRteFunc.x_list, dtype=float)[1:-1]
    m_min = float(sol.mNrmMin)
    m_ex_top = float(np.exp(mu_k[-1]))
    m_top = m_min + m_ex_top
    # Build the beyond-grid window in EXCESS-resources space: with the
    # infinite-horizon borrowing limit at -hNrmPes, m_top can be negative
    # (the five-point arm), where a geomspace in m-space breaks down.
    m_eval = m_min + np.geomspace(m_ex_top * 1.02, M_EVAL_TOP - m_min, N_EVAL)
    err = np.abs(sol.cFunc(m_eval) - truth.cFunc(m_eval))
    if not np.all(np.isfinite(err)):
        msg = f"non-finite errors in beyond-grid window: {err}"
        raise RuntimeError(msg)
    return m_top, {
        "max_abs_err": float(np.max(err)),
        "mean_abs_err": float(np.mean(err)),
        "max_rel_err": float(np.max(err / truth.cFunc(m_eval))),
    }


def main() -> None:
    # force=True: HARK's import installs a root handler at WARNING, which
    # makes a plain basicConfig a no-op and silently swallows INFO logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        force=True,
    )
    t0 = time.time()
    truth = solve(TRUTH_GRID)
    h_opt = float(truth.hNrm)
    LOG.info("truth solved in %.0f s (hNrm = %.3f)", time.time() - t0, h_opt)

    # Provenance stamp: make_exhibit.py refuses to plot a JSON whose params
    # do not match the current script, so a stale result cannot become a figure.
    results = {
        "hNrm": h_opt,
        "truth_grid": TRUTH_GRID,
        "params": {k: repr(v) for k, v in sorted(PARAMS.items())},
        "variants": [],
    }

    sol = solve({"aXtraMax": 4.0, "aXtraCount": 5, "aXtraNestFac": -1})
    m_top, metrics = score(sol, truth)
    results["variants"].append(
        {
            "label": "paper-linear-5pt",
            "nest": -1,
            "a_over_h": 4.0 / h_opt,
            "m_top": m_top,
            **metrics,
        },
    )
    LOG.info(
        "paper-linear-5pt: a/h %.2f  max_abs %.2e",
        4.0 / h_opt,
        metrics["max_abs_err"],
    )

    # Count-vs-extent confound control: same extent as the five-point arm,
    # production point count. Isolates what raising the COUNT alone buys.
    sol = solve({"aXtraMax": 4.0, "aXtraCount": SPARSE_COUNT, "aXtraNestFac": 3})
    m_top, metrics = score(sol, truth)
    results["variants"].append(
        {
            "label": "count30-same-extent",
            "nest": 3,
            "a_over_h": 4.0 / h_opt,
            "m_top": m_top,
            **metrics,
        },
    )
    LOG.info(
        "count30-same-extent: a/h %.2f  max_abs %.2e",
        4.0 / h_opt,
        metrics["max_abs_err"],
    )

    for nest in NEST_FACS:
        for ratio in A_OVER_H:
            amax = ratio * h_opt
            sol = solve(
                {"aXtraMax": amax, "aXtraCount": SPARSE_COUNT, "aXtraNestFac": nest},
            )
            m_top, metrics = score(sol, truth)
            results["variants"].append(
                {
                    "label": f"nest{nest}-a{ratio:g}h",
                    "nest": nest,
                    "a_over_h": ratio,
                    "m_top": m_top,
                    **metrics,
                },
            )
            LOG.info("nest%d a/h=%g: max_abs %.2e", nest, ratio, metrics["max_abs_err"])

    (HERE / "grid_extent.json").write_text(json.dumps(results, indent=2))
    LOG.info("Wrote grid_extent.json")


if __name__ == "__main__":
    main()
