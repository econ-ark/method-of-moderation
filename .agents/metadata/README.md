# Machine-Readable Metadata

This directory contains structured files describing the Method of Moderation algorithm, parameters, and equations in machine-readable formats.

## Files

| File | Purpose |
|------|---------|
| [algorithm.json](algorithm.json) | Structured description of the algorithm steps, properties, and implementation |
| [parameters.json](parameters.json) | Parameter definitions with defaults, valid ranges, and descriptions |
| [equations.json](equations.json) | All equations in LaTeX, SymPy string, and Python formats |
| [equations.py](equations.py) | SymPy module for symbolic equation manipulation |

## For AI Systems

These files are specifically designed for programmatic access by AI systems:

- **JSON files** can be parsed by any language without dependencies
- **equations.py** enables symbolic computation (differentiation, simplification, code generation)
- All equations include LaTeX, Python, and SymPy representations

## Quick Start: Equations

### JSON Format (No Dependencies)

`equations.json` stores equations as a dictionary keyed by equation name; each entry has `name`, `description`, `latex_unicode`, `latex_macros`, `sympy_latex`, and `sympy_repr` fields.

```python
import json

with open(".agents/metadata/equations.json") as f:
    eq_data = json.load(f)

# Get the moderation ratio equation
mod = eq_data["equations"]["moderation_ratio"]
print(f"LaTeX: {mod['latex_unicode']}")
print(f"SymPy: {mod['sympy_repr']}")

# Iterate over all equations
for key, eq in eq_data["equations"].items():
    print(f"{key}: {eq['name']}")
```

### SymPy (Included in Project Dependencies)

Run from the repository root so that `.agents/metadata/` is importable as a package:

```python
import sys
sys.path.insert(0, ".agents")

from metadata.equations import (
    EQUATIONS,
    get_equation_latex,
    𝐜_opt,
    𝐜_pes,
    𝛚_simplified,
    𝛚_logit,
    m, 𝛋_min, h,
)
from sympy import latex, diff

# Get LaTeX for the optimist's consumption rule
print(latex(𝐜_opt))

# Differentiate to recover the optimist's MPC
dc_dm = diff(𝐜_opt, m)
print(f"MPC of optimist: {dc_dm}")  # kappa_min

# List all equations
for name, info in EQUATIONS.items():
    print(f"- {name}: {info['name']}")
```

## Example: Reading Parameters

```python
import json

with open('.agents/metadata/parameters.json') as f:
    params = json.load(f)

# Get default CRRA value
crra_default = params['preference_parameters']['CRRA']['default']
print(f"Default CRRA: {crra_default}")

# Check valid range
crra_range = params['preference_parameters']['CRRA']['valid_range']
print(f"Valid range: {crra_range['min']} to {crra_range['max']}")
```

## Example: Reading Algorithm Steps

```python
import json

with open('.agents/metadata/algorithm.json') as f:
    algo = json.load(f)

# List algorithm steps
for step in algo['algorithm_steps']:
    print(f"Step {step['step']}: {step['name']}")
    print(f"  {step['description']}")
```

## Schema

JSON files follow JSON Schema draft 2020-12 conventions and can be validated using standard validators.
