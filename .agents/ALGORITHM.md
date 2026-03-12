# The Method of Moderation: Algorithm Description

## Prerequisites

The Method of Moderation applies to consumption-saving problems where:

1. The consumer has CRRA (constant relative risk aversion) utility
2. Income follows a Friedman-Muth process with transitory and permanent shocks
3. There is a positive probability of zero income (natural borrowing constraint)
4. Standard "patience conditions" hold (ensuring a finite solution exists)

## Step-by-Step Algorithm

### Step 1: Compute Analytical Bounds

First, compute the closed-form solutions for two extreme cases:

**Optimist's consumption function** (assumes no risk):
```
c_opt(m) = κ_min × (m + h_opt)
```
where:
- `κ_min` = minimum marginal propensity to consume (MPC)
- `h_opt` = human wealth assuming expected income forever

**Pessimist's consumption function** (assumes worst possible income):
```
c_pes(m) = κ_min × (m - m_min)
```
where:
- `m_min` = natural borrowing constraint (negative of minimum human wealth)

### Step 2: Solve Standard EGM at Gridpoints

Use the endogenous gridpoints method to compute `(m, c)` pairs:

1. Define end-of-period asset grid: `a_grid`
2. Use Euler equation to find consumption: `c = u'^{-1}(β R E[u'(c')])`
3. Compute market resources: `m = a + c`
4. Result: pairs `{(m_i, c_i)}` for i = 1, ..., N

### Step 3: Compute Moderation Ratio

For each gridpoint, calculate how close the realist is to the optimist:

```
ω_i = (c_i - c_pes(m_i)) / (c_opt(m_i) - c_pes(m_i))
```

This ratio satisfies `0 < ω < 1` by theory:
- `ω → 0`: Realist behaves like pessimist (maximum precautionary saving)
- `ω → 1`: Realist behaves like optimist (no precautionary saving)

### Step 4: Transform to Log-Excess-Resources Space

Define log excess market resources:
```
μ_i = log(m_i - m_min)
```

This maps the domain `(m_min, ∞)` to `(-∞, ∞)`.

### Step 5: Apply Logit Transformation

Transform the moderation ratio using the logit function:
```
χ_i = log(ω_i / (1 - ω_i))
```

**Key property**: The function `χ(μ)` is asymptotically linear as `μ → ∞`.
This means `χ'(μ) → 0`, which prevents extrapolation errors.

### Step 6: Interpolate χ Function

Create a smooth interpolant `χ̂(μ)` using the computed points `{(μ_i, χ_i)}`.

Options:
- Linear interpolation (simple)
- Cubic spline with derivatives (smoother)
- Hermite interpolation with computed slopes (best)

### Step 7: Reconstruct Consumption Function

For any market resources `m`:

1. Compute log excess resources: `μ = log(m - m_min)`
2. Evaluate interpolant: `χ̂ = χ̂(μ)`
3. Invert logit: `ω̂ = 1 / (1 + exp(-χ̂))`
4. Reconstruct consumption: `ĉ(m) = c_pes(m) + ω̂ × (c_opt(m) - c_pes(m))`

## Why This Works

1. **Bounded by construction**: `ĉ(m)` is always between `c_pes(m)` and `c_opt(m)`
2. **Asymptotic linearity**: `χ(μ)` flattens out for large `μ`, preventing divergence
3. **Smooth behavior**: The logit transformation creates a well-behaved function
4. **Theoretical guarantees**: Based on proven bounds from consumption theory

## Computational Complexity

- Same gridpoint count as standard EGM
- Additional overhead: O(N) for transformation computations
- Interpolation: Same as standard methods

## Extensions

The paper also describes:
- **Tighter upper bounds**: Using maximum MPC constraints
- **Stochastic returns**: Extension to risky rate of return
- **Value function approximation**: Similar moderation technique for value functions
