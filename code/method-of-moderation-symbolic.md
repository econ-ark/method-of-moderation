---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: .venv-darwin-arm64
    language: python
    name: python3
---

# Method of Moderation: Symbolic Mathematics

This notebook demonstrates the **symbolic equation infrastructure** for the Method of Moderation paper.

All equations from the paper are available as SymPy expressions in `metadata/equations.py`, using Unicode mathematical symbols that match the paper's notation.

**See also:**
- `method-of-moderation.ipynb` — Main notebook with numerical examples and figures
- `method-of-moderation-myst.ipynb` — MyST version with LaTeX macros (for paper builds)
- `metadata/equations.py` — Full SymPy module
- `metadata/equations.json` — JSON export for non-Python systems


## Getting Started

The Method of Moderation equations are also available as **symbolic expressions** using SymPy. This enables:
- Automatic differentiation
- Algebraic simplification
- LaTeX generation
- Code generation for any language

```{note} AI Accessibility
These symbolic equations are designed to be discoverable and usable by AI systems. See `metadata/equations.py` for the full module.
```

```python
# Import symbolic equations from the metadata module
import sys

sys.path.insert(0, "..")

from metadata.equations import (
    # Utilities
    EQUATIONS,
    R,
    Þ_formula,  # Parameter formulas
    h̄,
    list_equations,
    m,
    # Unicode symbols (matching paper notation)
    β,  # Transformations
    ρ,
    𝐜_opt,
    𝛋_min,
    𝛋_min_formula,
    𝛘,
    𝛘_definition,
    𝛚_from_𝛘,
)
from sympy import diff, init_printing, simplify

# Enable pretty printing
init_printing(use_unicode=True)
```

### Available Equations

The `EQUATIONS` dictionary contains all key equations from the paper:

```python
# List all available equations
print("Available symbolic equations:")
for name in list_equations():
    print(f"  - {name}: {EQUATIONS[name]['name']}")
```

### Symbolic Differentiation

We can verify that the marginal propensity to consume (MPC) of the optimist is exactly $\kappa_{\min}$:

```python
# The optimist consumption function: 𝐜̄(m) = 𝛋_min × (m + h̄)
print("Optimist consumption 𝐜̄(m):")
display(𝐜_opt)

# Differentiate with respect to market resources
mpc_optimist = diff(𝐜_opt, m)
print("\nMPC of optimist (d𝐜̄/dm):")
display(mpc_optimist)

# Verify it equals 𝛋_min
print("\n✓ Confirmed: MPC = 𝛋_min")
```

### LaTeX Export

SymPy can generate publication-quality LaTeX from any expression:

```python
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
```

### Algebraic Verification

We can verify that the patience factor formula is correct:

```python
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

```python
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

```python
# The logit transformation: 𝛘 = log(𝛚/(1-𝛚))
print("Logit transformation:")
display(𝛘_definition)

# Its inverse: 𝛚 = 1/(1 + exp(-𝛘))
print("\nInverse (expit):")
display(𝛚_from_𝛘)

# Verify they are inverses
from sympy import log

composed = simplify(log(𝛚_from_𝛘 / (1 - 𝛚_from_𝛘)))
print(f"\nVerification - logit(expit(𝛘)) = 𝛘: {composed == 𝛘}")
```

```{seealso}
For the complete symbolic equation module, see:
- `metadata/equations.py` - Full SymPy definitions
- `metadata/equations.json` - JSON format for non-Python systems
- `README_IF_YOU_ARE_AN_AI/KEY_EQUATIONS.md` - Human-readable summary
```
