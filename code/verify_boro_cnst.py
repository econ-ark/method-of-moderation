"""Verify Table 3, the artificial-borrowing-constraint sweep.

Reproduces `tbl:boro-cnst-art` from `content/paper/moderation_extended.md`:
for each imposed constraint in {natural, -0.10, -0.05, 0.00, 0.10} it reports
MPCmax, the maximum absolute consumption error at low wealth
(:math:`\\mNrmEx \\leq 1`) for sparse-grid EGM and sparse-grid MoM against a
dense EGM reference, and the interval over which the true rule coincides with
the tighter upper bound.

It also reproduces the two "unbounded" numbers quoted in the prose of the same
section: with the envelope switched off at the tightest reported constraint
(:math:`\\underline{\\aNrm} = 0`), the moderated rule's worst-case error and
the largest amount by which it exceeds the budget.

The envelope is switched off by setting the two attributes that gate it on the
solved MoM consumption function, which is the mechanism `moderation.py` itself
uses to express "no envelope":

* ``cFunc.tight_upper_slope = None`` disables `_apply_tight_upper_bound`, the
  clip to :math:`\\MPCmax\\,\\mNrmEx` (the 45-degree budget line when an
  artificial constraint binds).
* ``cFunc.mKink = None`` disables `_apply_constrained_branch`, the exact
  substitution :math:`\\cNrm = \\mNrmEx` below the kink.

Both guards are already written as ``if ... is None: return moderated``, so no
monkeypatching is needed and the disabled run exercises the same code path a
build without the envelope would.

This script is a reporter: it asserts nothing and always exits 0 unless a solve
raises. Compare its output against the paper by eye.

It is also the generator of `content/tables/boro-cnst-art.md`, the fragment the
manuscript includes in place of a hand-typed table.

Run from the repository root:

    uv run python code/verify_boro_cnst.py          # report only
    uv run python code/verify_boro_cnst.py --list   # the table name it owns
    uv run python code/verify_boro_cnst.py --write  # regenerate Table 3
    uv run python code/verify_boro_cnst.py --check  # CI gate
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np

# Make `code/` importable regardless of caller CWD, without leaking a relative
# entry into sys.path.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from calibration import DENSE_GRID, PARAMS, SPARSE_GRID  # noqa: E402
from moderation import (  # noqa: E402
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
)
from table_io import add_table_args, emit, resolve_names, sci  # noqa: E402

logger = logging.getLogger("verify_boro_cnst")

# The generated fragment this script owns, in `content/tables/`.
TABLES = ("boro-cnst-art",)

# The table scores errors at low wealth only; the header says mNrmEx <= 1.
M_EX_MAX = 1.0
N_EVAL = 200_001

# Rows of Table 3, in the paper's order. `None` is the natural constraint.
BORO_CNST_ART_VALUES = (None, -0.10, -0.05, 0.00, 0.10)

# The prose's unbounded numbers are quoted at the zero-borrowing row.
UNBOUNDED_ROW = 0.00

# Tolerance for declaring the true rule to coincide with the budget line. The
# dense reference's first segment IS the budget line, so agreement there is at
# machine precision, while past the kink the gap opens immediately.
COINCIDENCE_TOL = 1e-9


def _solve(consumer_cls, params, grid):
    consumer = consumer_cls(**(params | grid))
    consumer.solve()
    return consumer.solution[0]


def _eval_grid(m_nrm_min):
    """Evaluation points covering mNrmEx in (0, M_EX_MAX]."""
    return np.linspace(m_nrm_min, m_nrm_min + M_EX_MAX, N_EVAL)[1:]


def _max_abs_error(truth_cfunc, sol_cfunc, m_eval):
    return float(np.max(np.abs(truth_cfunc(m_eval) - sol_cfunc(m_eval))))


def _max_budget_breach(sol_cfunc, m_nrm_min, m_eval):
    """Largest amount by which the rule spends more than the agent has."""
    excess = np.asarray(sol_cfunc(m_eval)) - (m_eval - m_nrm_min)
    return float(np.max(np.maximum(excess, 0.0)))


def _coincidence_interval(truth_cfunc, m_nrm_min, m_eval):
    """Largest mNrmEx at which the truth still sits on the budget line.

    Returns 0.0 when the truth never coincides with the bound, which is the
    natural-constraint case: there the tighter bound has slope MPCmax < 1 and
    is touched only at the constraint point itself.
    """
    m_ex = m_eval - m_nrm_min
    on_line = np.abs(np.asarray(truth_cfunc(m_eval)) - m_ex) <= COINCIDENCE_TOL
    if not on_line.any():
        return 0.0
    # Take the last point of the leading run, so a chance re-crossing further
    # out cannot inflate the reported interval.
    first_off = np.argmin(on_line) if not on_line.all() else len(on_line)
    if first_off == 0:
        return 0.0
    return float(m_ex[first_off - 1])


def _disable_envelope(mom_sol):
    """Return a copy of the MoM solution with both envelope mechanisms off.

    See the module docstring: `tight_upper_slope` gates the clip to the tighter
    upper bound and `mKink` gates the exact constrained branch. Setting both to
    None is the same switch `moderation.py` flips when neither applies.
    """
    bare = copy.deepcopy(mom_sol)
    if bare.cFunc.tight_upper_slope is None and bare.cFunc.mKink is None:
        logger.warning(
            "MoM solution already had no envelope (tight_upper_slope and mKink "
            "both None); the 'unbounded' run is not measuring what it claims.",
        )
    bare.cFunc.tight_upper_slope = None
    bare.cFunc.mKink = None
    return bare


def _row(boro_cnst_art):
    """Solve one Table 3 row and return everything the table and prose need."""
    params = PARAMS | {"BoroCnstArt": boro_cnst_art}
    truth_sol = _solve(IndShockEGMConsumerType, params, DENSE_GRID)
    egm_sol = _solve(IndShockEGMConsumerType, params, SPARSE_GRID)
    mom_sol = _solve(IndShockMoMConsumerType, params, SPARSE_GRID)

    m_nrm_min = float(truth_sol.mNrmMin)
    if not np.isclose(egm_sol.mNrmMin, m_nrm_min) or not np.isclose(
        mom_sol.mNrmMin,
        m_nrm_min,
    ):
        logger.error(
            "mNrmMin differs across solves (dense %r, EGM %r, MoM %r); the three "
            "solutions are not on a common domain and the errors below are not "
            "comparable.",
            m_nrm_min,
            float(egm_sol.mNrmMin),
            float(mom_sol.mNrmMin),
        )
    m_eval = _eval_grid(m_nrm_min)
    bare_sol = _disable_envelope(mom_sol)

    return {
        "BoroCnstArt": boro_cnst_art,
        "mNrmMin": m_nrm_min,
        "MPCmax": float(mom_sol.MPCmax),
        "MPCmax_egm": float(egm_sol.MPCmax),
        "egm_err": _max_abs_error(truth_sol.cFunc, egm_sol.cFunc, m_eval),
        "mom_err": _max_abs_error(truth_sol.cFunc, mom_sol.cFunc, m_eval),
        "coincide_truth": _coincidence_interval(truth_sol.cFunc, m_nrm_min, m_eval),
        "mKink_mom_ex": (
            None if mom_sol.cFunc.mKink is None else mom_sol.cFunc.mKink - m_nrm_min
        ),
        "bare_err": _max_abs_error(truth_sol.cFunc, bare_sol.cFunc, m_eval),
        "bare_breach": _max_budget_breach(bare_sol.cFunc, m_nrm_min, m_eval),
        "mom_breach": _max_budget_breach(mom_sol.cFunc, m_nrm_min, m_eval),
    }


def _constraint_label(boro_cnst_art) -> str:
    return "natural" if boro_cnst_art is None else f"${boro_cnst_art:.2f}$"


def _coincidence_label(coincide_truth) -> str:
    return (
        "nowhere" if coincide_truth == 0.0 else rf"$\mNrmEx \leq {coincide_truth:.3f}$"
    )


def _render_table3(rows) -> str:
    """The artificial-constraint sweep, `tbl:boro-cnst-art`."""
    caption = (
        rf"Maximum absolute consumption errors for $\mNrmEx \leq {M_EX_MAX:g}$ "
        "as the artificial borrowing constraint tightens, EGM and MoM on the "
        "same five-point grid against a dense reference solution. $\\MPCmax$ "
        "is the period-$T-1$ value; the last column is the interval over "
        "which the realist's rule coincides with the tighter bound."
    )
    body = [
        f"{_constraint_label(row['BoroCnstArt']):<8} & {row['MPCmax']:.3f} & "
        f"{sci(row['egm_err'])} & {sci(row['mom_err'])} & "
        f"{_coincidence_label(row['coincide_truth'])} \\\\"
        for row in rows
    ]
    return "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            rf"\caption{{{caption}}}",
            r"\label{tbl:boro-cnst-art}",
            r"\setlength{\tabcolsep}{8pt}",
            r"\begin{tabular}{rcccl}",
            r"\toprule",
            r"$\underline{\aNrm}$ & $\MPCmax$ & EGM & MoM & "
            r"rule coincides with bound \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ],
    )


def main(
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

    if write or check:
        try:
            names = resolve_names(only, TABLES)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
    else:
        names = []

    logger.info(
        "Table 3 (tbl:boro-cnst-art): max |c_truth - c_approx| for mNrmEx <= %g,",
        M_EX_MAX,
    )
    logger.info(
        "scored on %d points against a dense EGM reference (aXtraCount=%d).",
        N_EVAL - 1,
        DENSE_GRID["aXtraCount"],
    )
    logger.info("")

    rows = [_row(value) for value in BORO_CNST_ART_VALUES]

    header = (
        f"{'aNrmMin':>8}  {'MPCmax':>6}  {'EGM':>9}  {'MoM':>9}  "
        f"{'EGM (raw)':>10}  {'MoM (raw)':>10}  coincides"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for row in rows:
        label = "natural" if row["BoroCnstArt"] is None else f"{row['BoroCnstArt']:.2f}"
        coincide = (
            "nowhere"
            if row["coincide_truth"] == 0.0
            else f"mNrmEx <= {row['coincide_truth']:.3f}"
        )
        logger.info(
            "%8s  %6.3f  %9s  %9s  %10.3e  %10.3e  %s",
            label,
            row["MPCmax"],
            sci(row["egm_err"]),
            sci(row["mom_err"]),
            row["egm_err"],
            row["mom_err"],
            coincide,
        )

    logger.info("")
    logger.info("Diagnostics per row (not printed in the paper):")
    for row in rows:
        label = "natural" if row["BoroCnstArt"] is None else f"{row['BoroCnstArt']:.2f}"
        kink = row["mKink_mom_ex"]
        logger.info(
            "  %8s  mNrmMin=%+.6f  MoM kink (mNrmEx)=%s  truth coincidence=%.6f  "
            "MoM breach=%.3e  envelope off: err=%.3e breach=%.3e",
            label,
            row["mNrmMin"],
            "none" if kink is None else f"{kink:.6f}",
            row["coincide_truth"],
            row["mom_breach"],
            row["bare_err"],
            row["bare_breach"],
        )

    logger.info("")
    unbounded = next(row for row in rows if row["BoroCnstArt"] == UNBOUNDED_ROW)
    logger.info(
        "Envelope disabled at aNrmMin = %.2f (tight_upper_slope=None, mKink=None):",
        UNBOUNDED_ROW,
    )
    logger.info(
        "  max |c_truth - c_MoM|  = %.4e  (%s)  [paper: 4.5(-2)]",
        unbounded["bare_err"],
        sci(unbounded["bare_err"]),
    )
    logger.info(
        "  max budget breach      = %.4e         [paper: 0.045]",
        unbounded["bare_breach"],
    )
    logger.info(
        "  with the envelope on, the same run gives %.4e and a breach of %.3e.",
        unbounded["mom_err"],
        unbounded["mom_breach"],
    )
    logger.info(
        "  EGM at the same row is %.4e, so without the envelope MoM is the %s "
        "of the two.",
        unbounded["egm_err"],
        "larger" if unbounded["bare_err"] > unbounded["egm_err"] else "smaller",
    )
    if np.isclose(unbounded["bare_err"], unbounded["bare_breach"], rtol=1e-12):
        logger.info(
            "  The two numbers agree because the worst overshoot lands inside the "
            "constrained region, where the truth IS the budget line, so the error "
            "there and the breach measure the same distance.",
        )

    statuses = [
        emit(
            name,
            _render_table3(rows),
            script="code/verify_boro_cnst.py",
            command="uv run python code/verify_boro_cnst.py --write",
            check_only=check,
        )
        for name in names
    ]
    return max(statuses, default=0)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the artificial-constraint sweep of the paper and generate "
            "the fragment the manuscript includes."
        ),
    )
    add_table_args(parser, TABLES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(**vars(_parse_args())))
