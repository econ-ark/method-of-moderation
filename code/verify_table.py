"""Verify Table 1 approximation errors (EGM vs MoM)."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "code")
from moderation import IndShockEGMConsumerType, IndShockMoMConsumerType

params = {
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
    "UnempPrb": 0.0,
    "BoroCnstArt": None,
}

dense_grid = {"aXtraMin": 0.001, "aXtraMax": 40, "aXtraCount": 500, "aXtraNestFac": 3}
sparse_grid = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 5, "aXtraNestFac": -1}

# Truth (dense EGM, 500 points with triple-exponential nesting)
truth = IndShockEGMConsumerType(**(params | dense_grid))
truth.solve()
truth_sol = truth.solution[0]

# Sparse EGM (5 points)
egm = IndShockEGMConsumerType(**(params | sparse_grid))
egm.solve()
egm_sol = egm.solution[0]

# Sparse MoM (5 points, same grid)
mom = IndShockMoMConsumerType(**(params | sparse_grid))
mom.solve()
mom_sol = mom.solution[0]

# Grid points (skip borrowing constraint)
grid_pts = np.array(egm_sol.cFunc.x_list)[1:]
m_bar = 30.0
n_eval = 1000

print(f"Grid points: {grid_pts}")
print()

for label, sol in [("EGM", egm_sol), ("MoM", mom_sol)]:
    errors = []
    for i in range(len(grid_pts) - 1):
        left, right = grid_pts[i], grid_pts[i + 1]
        m_eval = np.linspace(left + 1e-8, right - 1e-8, n_eval)
        gap = np.max(np.abs(truth_sol.cFunc(m_eval) - sol.cFunc(m_eval)))
        errors.append(gap)
    # Extrapolation region
    m_eval = np.linspace(grid_pts[-1] + 1e-8, m_bar, n_eval)
    gap = np.max(np.abs(truth_sol.cFunc(m_eval) - sol.cFunc(m_eval)))
    errors.append(gap)

    print(f"{label} max absolute errors:")
    for j, e in enumerate(errors):
        if j < len(grid_pts) - 1:
            print(f"  [m{j}, m{j+1}]: {e:.2e}")
        else:
            print(f"  [m{j}, m_bar]: {e:.2e}")
    print()
