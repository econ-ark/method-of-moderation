# Mathematical Notation Reference

This document defines all mathematical symbols used in the Method of Moderation paper and code.

## State and Choice Variables

| Symbol | Name | Description |
|--------|------|-------------|
| `m` | Market resources | Total resources available for consumption (normalized by permanent income) |
| `c` | Consumption | Amount consumed (normalized by permanent income) |
| `a` | Assets | End-of-period assets: `a = m - c` |
| `p` | Permanent income | Level of permanent labor income |
| `y` | Income | Total noncapital income: `y = p × θ` |

## Shocks and Income Process

| Symbol | Name | Description |
|--------|------|-------------|
| `θ` | Transitory shock | Transitory income shock with E[θ] = 1 |
| `ψ` | Permanent shock | Permanent income shock with E[ψ] = 1 |
| `Ψ` | Permanent growth | Combined permanent growth: `Ψ = Γ × ψ` |
| `θ_min` | Minimum transitory | Worst possible transitory shock (often 0 for unemployment) |
| `℘` | Unemployment prob | Probability of zero income: P(θ = 0) |

## Parameters

| Symbol | Name | Description |
|--------|------|-------------|
| `ρ` | CRRA | Coefficient of relative risk aversion |
| `β` | Discount factor | Time preference parameter (0 < β < 1) |
| `R` | Gross return | Gross interest rate on assets |
| `Γ` | Permanent growth | Expected permanent income growth factor |

## Derived Quantities

| Symbol | Name | Formula | Description |
|--------|------|---------|-------------|
| `Þ` | Patience factor | `(βR)^(1/ρ)` | Absolute patience factor |
| `κ_min` | Minimum MPC | `1 - Þ/R` | MPC of perfect foresight consumer |
| `κ_max` | Maximum MPC | `1 - ℘^(1/ρ) × Þ/R` | Upper bound on MPC |
| `h` | Human wealth | `Γ/(R-Γ)` | PDV of expected future income |
| `h_pes` | Pessimist human wealth | `θ_min × Γ/(R-Γ)` | PDV assuming worst income |
| `m_min` | Borrowing constraint | `-h_pes` | Natural borrowing constraint |

## Method of Moderation Variables

| Symbol | Name | Formula | Description |
|--------|------|---------|-------------|
| `m^e` | Excess resources | `m - m_min` | Resources above borrowing constraint |
| `h^e` | Excess human wealth | `h - h_pes` | Human wealth above minimum |
| `μ` | Log excess resources | `log(m - m_min)` | Transformed state variable |
| `ω` | Moderation ratio | `(c - c_pes)/(c_opt - c_pes)` | Position between bounds |
| `χ` | Chi function | `log(ω/(1-ω))` | Logit-transformed moderation |

## Consumption Functions

| Symbol | Name | Formula | Description |
|--------|------|---------|-------------|
| `c_opt(m)` | Optimist | `κ_min × (m + h)` | Upper bound (no precautionary saving) |
| `c_pes(m)` | Pessimist | `κ_min × (m - m_min)` | Lower bound (maximum precaution) |
| `c_tight(m)` | Tight bound | `c_opt(m) - (κ_max - κ_min) × m^e` | Tighter upper bound |
| `c_real(m)` | Realist | Numerical solution | Optimal consumption under risk |

## Value Functions

| Symbol | Name | Description |
|--------|------|-------------|
| `v(m)` | Value function | Expected discounted utility from state m |
| `v'(m)` | Marginal value | Derivative: envelope condition |
| `u(c)` | Utility | CRRA utility: `c^(1-ρ)/(1-ρ)` |
| `u'(c)` | Marginal utility | `c^(-ρ)` |

## Patience Conditions

The following conditions must hold for a finite solution:

| Condition | Name | Requirement |
|-----------|------|-------------|
| FVAC | Finite value of autarky | `0 < β Γ^(1-ρ) E[ψ^(1-ρ)] < 1` |
| AIC | Absolute impatience | `Þ < 1` |
| RIC | Return impatience | `Þ/R < 1` |
| GIC | Growth impatience | `Þ/Γ < 1` |
| FHWC | Finite human wealth | `Γ/R < 1` |

## Code Variable Names

In `code/moderation.py`, the following naming conventions are used:

| Code | Math | Description |
|------|------|-------------|
| `mNrm` | `m` | Normalized market resources |
| `cNrm` | `c` | Normalized consumption |
| `aNrm` | `a` | Normalized assets |
| `MPCmin` | `κ_min` | Minimum MPC |
| `MPCmax` | `κ_max` | Maximum MPC |
| `hNrm` | `h` | Human wealth |
| `mNrmMin` | `m_min` | Borrowing constraint |
| `cFuncOpt` | `c_opt` | Optimist consumption function |
| `cFuncPes` | `c_pes` | Pessimist consumption function |
| `cFuncTight` | `c_tight` | Tight upper bound function |
