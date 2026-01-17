# Machine-Readable Metadata

This directory contains structured JSON files describing the Method of Moderation algorithm and its parameters in a machine-readable format.

## Files

| File | Purpose |
|------|---------|
| [algorithm.json](algorithm.json) | Structured description of the algorithm steps, properties, and implementation |
| [parameters.json](parameters.json) | Parameter definitions with defaults, valid ranges, and descriptions |

## Usage

These files are designed for:

1. **AI systems** parsing repository content
2. **Automated tools** generating documentation
3. **Validation scripts** checking parameter ranges
4. **Integration** with other software systems

## Example: Reading Parameters in Python

```python
import json

with open('metadata/parameters.json') as f:
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

with open('metadata/algorithm.json') as f:
    algo = json.load(f)

# List algorithm steps
for step in algo['algorithm_steps']:
    print(f"Step {step['step']}: {step['name']}")
    print(f"  {step['description']}")
```

## Schema

Both files follow JSON Schema draft 2020-12 conventions and can be validated using standard JSON Schema validators.
