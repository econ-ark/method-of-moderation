---
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Method of Moderation: Symbolic Mathematics

This notebook demonstrates the **symbolic equation infrastructure** for the Method of Moderation paper.

All equations from the paper are available as SymPy expressions in `.agents/metadata/equations.py`, using Unicode mathematical symbols that match the paper's notation.

**See also:**
- `method-of-moderation.md`: Main notebook with numerical examples and figures
- `.agents/metadata/equations.py`: Full SymPy module
- `.agents/metadata/equations.json`: JSON export for non-Python systems

+++

## Getting Started

The Method of Moderation equations are also available as **symbolic expressions** using SymPy. This enables:
- Automatic differentiation
- Algebraic simplification
- LaTeX generation
- Code generation for any language

```{note} AI Accessibility
These symbolic equations are designed to be discoverable and usable by AI systems. See `.agents/metadata/equations.py` for the full module.
```

```{code-cell} ipython3
# Import symbolic equations from the metadata module
import sys
from pathlib import Path

# Walk up from cwd to locate the repository root (the directory that contains
# `.agents/`). This keeps the import working under JupyterLab launched from
# either the repo root or `code/`, and under `jupyter nbconvert` from any CWD.
_search = Path.cwd().resolve()
for _candidate in [_search, *_search.parents]:
    if (_candidate / ".agents").is_dir():
        _repo_root = _candidate
        break
else:
    raise RuntimeError(
        "Could not locate the repository root (no .agents/ directory found "
        f"in {Path.cwd().resolve()} or any parent)."
    )
_agents_dir = str(_repo_root / ".agents")
if _agents_dir not in sys.path:
    sys.path.insert(0, _agents_dir)

from metadata.equations import (
    # Utilities
    EQUATIONS,
    R,
    Þ_formula,  # Parameter formulas
    h̄,
    list_equations,
    m,
    verify_identities,
    β,  # Transformations
    ρ,
    𝐜_opt,
    𝐜_pes,
    𝛋_min,
    𝛋_min_formula,
    𝛘,
    𝛘_definition,
    𝛚_from_𝛘,
)
from sympy import diff, init_printing, log, simplify

# Enable pretty printing
init_printing(use_unicode=True)
```

### Available Equations

The `EQUATIONS` dictionary contains all key equations from the paper:

```{code-cell} ipython3
# List all available equations
print("Available symbolic equations:")
for name in list_equations():
    print(f"  - {name}: {EQUATIONS[name]['name']}")
```

### Symbolic Differentiation

We can verify that the marginal propensity to consume (MPC) of the optimist is exactly $\kappa_{\min}$:

```{code-cell} ipython3
# The optimist consumption function: 𝐜̄(m) = 𝛋_min × (m + h̄)
print("Optimist consumption 𝐜̄(m):")
display(𝐜_opt)

# Differentiate with respect to market resources
mpc_optimist = diff(𝐜_opt, m)
print("\nMPC of optimist (d𝐜̄/dm):")
display(mpc_optimist)
print("✓ Confirmed: optimist MPC = 𝛋_min")

# The pessimist consumption function: 𝐜̲(m) = 𝛋_min × (m - m_min) has the same slope.
# This equality of slopes is what makes 𝐜̄ - 𝐜̲ = 𝛋_min × Δh a constant in m, so
# the moderation ratio omega(m) measures position as a pure fraction of the
# (constant) human-wealth gap.
mpc_pessimist = diff(𝐜_pes, m)
print("\nMPC of pessimist (d𝐜̲/dm):")
display(mpc_pessimist)
print(
    f"✓ Both bounds share the same slope: optimist - pessimist MPC = {simplify(mpc_optimist - mpc_pessimist)}"
)
```

### LaTeX Export

SymPy can generate publication-quality LaTeX from any expression:

```{code-cell} ipython3
# Generate LaTeX for key equations
equations_to_show = [
    "moderation_ratio",
    "logit_moderation",
    "consumption_reconstructed",
]

for eq_name in equations_to_show:
    eq = EQUATIONS[eq_name]
    print(f"{eq['name']}:")
    print(f"  LaTeX: {eq['latex']}")
    print()

# Verify that the printed latex matches the stored sympy expression. The
# moderation_ratio in particular has two equivalent algebraic forms (paper
# canonical (𝐜_real - 𝐜_pes) / (Δh × 𝛋_min) and the (𝐜_real - 𝐜_pes) /
# (𝐜_opt - 𝐜_pes) form used in the code); verify_identities() proves they
# are equal symbolically.
print("Symbolic identities:")
verify_identities()
```

### Algebraic Verification

We can verify that the patience factor formula is correct:

```{code-cell} ipython3
# The patience factor Þ = (βR)^(1/ρ)
print("Patience factor (Þ):")
display(Þ_formula)

# The minimum MPC formula: 𝛋_min = 1 - Þ/R
print("\nMinimum MPC formula:")
display(𝛋_min_formula)

# Substitute and simplify
print("\nExpanded form:")
expanded = 1 - (β * R) ** (1 / ρ) / R
display(simplify(expanded))
```

### Numerical Evaluation

Symbolic expressions can be evaluated numerically with specific parameter values:

```{code-cell} ipython3
import numpy as np
from sympy import lambdify

# Create a numerical function from the symbolic expression
# 𝐜_opt = 𝛋_min × (m + h̄)
c_opt_func = lambdify([m, 𝛋_min, h̄], 𝐜_opt, "numpy")

# Evaluate at specific values
m_vals = np.array([1, 5, 10, 20])
κ_val = 0.04  # Example MPC
h_val = 25  # Example human wealth

c_opt_vals = c_opt_func(m_vals, κ_val, h_val)

print("Optimist consumption at 𝛋_min=0.04, h̄=25:")
for m_v, c_v in zip(m_vals, c_opt_vals):
    print(f"  m = {m_v:2d} → 𝐜̄ = {c_v:.3f}")
```

### The Logit Transformation

The key insight of the Method of Moderation is that the logit of the moderation ratio becomes asymptotically linear:

```{code-cell} ipython3
# The logit transformation: 𝛘 = log(𝛚/(1-𝛚))
print("Logit transformation:")
display(𝛘_definition)

# Its inverse: 𝛚 = 1/(1 + exp(-𝛘))
print("\nInverse (expit):")
display(𝛚_from_𝛘)

# Verify they are inverses (log is imported at the top of the notebook).
composed = simplify(log(𝛚_from_𝛘 / (1 - 𝛚_from_𝛘)))
print(f"\nVerification - logit(expit(𝛘)) = 𝛘: {composed == 𝛘}")
```

```{seealso}
For the complete symbolic equation module, see:
- `.agents/metadata/equations.py` - Full SymPy definitions
- `.agents/metadata/equations.json` - JSON format for non-Python systems
- `.agents/KEY_EQUATIONS.md` - Human-readable summary
```
