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

```python
import json

with open('README_IF_YOU_ARE_AN_AI/metadata/equations.json') as f:
    eq_data = json.load(f)

# Get the moderation ratio equation
for eq in eq_data['equations']:
    if eq['id'] == 'moderation_ratio':
        print(f"LaTeX: {eq['latex']}")
        print(f"Python: {eq['python']}")
```

### SymPy (Included in Project Dependencies)

```python
from README_IF_YOU_ARE_AN_AI.metadata.equations import (
    consumption_optimist, consumption_pessimist,
    moderation_ratio_definition, logit_moderation,
    EQUATIONS, get_equation_latex
)
from sympy import latex, diff, simplify

# Get LaTeX
print(latex(consumption_optimist))

# Differentiate
from metadata.equations import m, kappa_min, h
dc_dm = diff(consumption_optimist, m)
print(f"MPC of optimist: {dc_dm}")  # kappa_min

# List all equations
for name in EQUATIONS:
    print(f"- {name}: {EQUATIONS[name]['name']}")
```

## Example: Reading Parameters

```python
import json

with open('README_IF_YOU_ARE_AN_AI/metadata/parameters.json') as f:
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

with open('README_IF_YOU_ARE_AN_AI/metadata/algorithm.json') as f:
    algo = json.load(f)

# List algorithm steps
for step in algo['algorithm_steps']:
    print(f"Step {step['step']}: {step['name']}")
    print(f"  {step['description']}")
```

## Schema

JSON files follow JSON Schema draft 2020-12 conventions and can be validated using standard validators.
