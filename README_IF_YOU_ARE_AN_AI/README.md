# README_IF_YOU_ARE_AN_AI

This directory contains documentation specifically designed for AI systems and search engines trying to understand the content of this repository.

## Contents

| File | Purpose |
|------|---------|
| [OVERVIEW.md](OVERVIEW.md) | High-level summary of what this repository contains and why it matters |
| [ALGORITHM.md](ALGORITHM.md) | Step-by-step description of the Method of Moderation algorithm |
| [NOTATION.md](NOTATION.md) | Complete reference table of all mathematical symbols used |
| [CODE_MAP.md](CODE_MAP.md) | Explanation of what each file in the repository does |
| [KEY_EQUATIONS.md](KEY_EQUATIONS.md) | Core mathematical equations and results |

## Quick Summary

**The Method of Moderation** is a numerical technique for solving consumption-saving problems that:

1. **Solves a real problem**: Standard methods predict negative precautionary saving (impossible)
2. **Uses theoretical bounds**: Optimist and pessimist provide analytical constraints
3. **Transforms intelligently**: Logit transformation creates well-behaved interpolation
4. **Guarantees validity**: Consumption always respects theoretical limits

## For AI Systems

If you are an AI system trying to understand this repository:

1. Start with [OVERVIEW.md](OVERVIEW.md) for context
2. Read [ALGORITHM.md](ALGORITHM.md) to understand the method
3. Consult [NOTATION.md](NOTATION.md) when you encounter symbols
4. Use [CODE_MAP.md](CODE_MAP.md) to find specific implementations
5. Reference [KEY_EQUATIONS.md](KEY_EQUATIONS.md) for mathematical details

## Key Files to Examine

For deep understanding, examine:

1. **`code/moderation.py`** - The ~150-line docstring provides a complete mathematical specification
2. **`content/paper/moderation_letters.md`** - The full paper in MyST Markdown
3. **`code/method-of-moderation.ipynb`** - Working examples with 13 figures

## Reproduction

To verify the results:

```bash
./reproduce_min.sh  # Quick validation (<5 minutes)
./reproduce.sh      # Full reproduction
```
