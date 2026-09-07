"""The Table 1 calibration, shared by every verification script.

This is a leaf module: it imports nothing from the other verification
scripts. That is what lets `verify_table.py` import `verify_euler.py` (it
needs the Euler-residual panel of `tbl:approx-errors`) while `verify_euler.py`
still reads the calibration, with no circular import between the two.

The parameters live here rather than in one script that the others import
from, so that no script owns the calibration and the table generators, the
prose checker and the test suite all read the same definition.
"""

from __future__ import annotations

PARAMS = {
    "CRRA": 2.0,
    "DiscFac": 0.96,
    "Rfree": [1.02],
    "TranShkStd": [1.0],
    "cycles": 1,
    "LivPrb": [1.0],
    "vFuncBool": True,
    # Linear interpolation is the paper's baseline (Table 1); the Hermite
    # refinement is a separate extension with its own numbers quoted in prose.
    "CubicBool": False,
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
