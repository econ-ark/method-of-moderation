# Overview: The Method of Moderation

## What This Repository Contains

This repository implements **"The Method of Moderation,"** a novel numerical technique for solving consumption-saving problems in economics. The method guarantees theoretically-valid extrapolation beyond computed gridpoints—solving a long-standing problem in computational economics.

## The Problem Being Solved

Standard numerical methods for consumption-saving problems (like the endogenous gridpoints method) produce consumption functions that behave poorly when evaluated outside the pre-computed grid. Specifically, linear extrapolation eventually predicts **negative precautionary saving**, which violates economic theory (consumers should always save more when facing risk).

## The Core Insight

Under standard assumptions, a "realist" consumer (who correctly perceives risks) will consume at a level strictly between:

- An **optimist** who ignores risk entirely (upper bound)
- A **pessimist** who assumes the worst possible outcome (lower bound)

Both the optimist and pessimist have **analytical perfect-foresight solutions**. The Method of Moderation leverages these known bounds to constrain the numerical solution, ensuring it always respects theoretical limits.

## Why It Matters

1. **Accuracy**: Prevents nonsensical predictions (negative precautionary saving)
2. **Robustness**: Works reliably even far outside the computed grid
3. **Efficiency**: Uses the same number of gridpoints as standard methods
4. **Theoretical grounding**: Based on established consumption theory, not ad-hoc fixes

## Quick Links

- **Paper**: [`content/paper/moderation_letters.md`](../content/paper/moderation_letters.md)
- **Core implementation**: [`code/moderation.py`](../code/moderation.py)
- **Illustrative notebook**: [`code/method-of-moderation.ipynb`](../code/method-of-moderation.ipynb)
- **Reproduction**: Run `./reproduce.sh` or `./reproduce_min.sh`

## Target Audience

- Computational economists solving dynamic optimization problems
- Researchers working with consumption-saving models
- Anyone interested in numerical methods for stochastic optimization
- Students learning about buffer-stock saving models
