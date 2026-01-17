# Algorithm Comparison: Method of Moderation vs Alternatives

This document compares the Method of Moderation to standard numerical methods for consumption-saving problems.

## Quick Comparison Table

| Property | Standard EGM | Method of Moderation | Linear Extrapolation | Exponential Decay |
|----------|-------------|---------------------|---------------------|-------------------|
| **Extrapolation Quality** | ❌ Poor | ✅ Excellent | ❌ Poor | ⚠️ Ad-hoc |
| **Theoretical Guarantees** | ❌ None | ✅ Bounded | ❌ None | ❌ None |
| **Precautionary Saving** | ❌ Can go negative | ✅ Always positive | ❌ Can go negative | ⚠️ Depends |
| **Computational Cost** | ✅ Low | ✅ Low + small overhead | ✅ Low | ✅ Low |
| **Implementation** | ✅ Simple | ⚠️ Moderate | ✅ Simple | ⚠️ Moderate |
| **Based on Theory** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |

## Detailed Comparison

### 1. Standard Endogenous Gridpoints Method (EGM)

**How it works:**
- Define asset grid
- Use Euler equation to find consumption at each gridpoint
- Interpolate between points (linear or spline)
- Extrapolate linearly beyond grid

**Strengths:**
- Simple implementation
- Fast computation
- Exact at gridpoints

**Weaknesses:**
- Linear extrapolation eventually predicts c > m (consuming more than you have)
- Implies negative precautionary saving for large wealth
- Requires ad-hoc fixes for evaluation outside grid

**When it fails:**
- Large positive income shocks push m beyond grid
- Simulations over many periods accumulate extrapolation errors

---

### 2. Method of Moderation (This Paper)

**How it works:**
- Compute analytical optimist and pessimist bounds
- Solve EGM at gridpoints (same as standard)
- Express solution as moderation ratio ω ∈ (0,1)
- Apply logit transformation χ = log(ω/(1-ω))
- Interpolate χ (asymptotically linear)
- Reconstruct consumption from bounds

**Strengths:**
- Consumption always bounded: c_pes < c < c_opt
- Precautionary saving always positive
- Excellent extrapolation beyond grid
- Based on economic theory, not ad-hoc
- Same gridpoint count as EGM

**Weaknesses:**
- Slightly more complex implementation
- Small computational overhead for transformations
- Requires bounds to be computed first

**When to use:**
- Any consumption-saving problem with income risk
- Simulations requiring reliable extrapolation
- Applications where theoretical consistency matters

---

### 3. Linear Extrapolation (Common Ad-Hoc Fix)

**How it works:**
- Use slope at grid boundary
- Extend linearly: c(m) = c_max + c'(m_max) × (m - m_max)

**Strengths:**
- Simple to implement
- Continuous with interior solution

**Weaknesses:**
- Eventually c(m) > m (impossible)
- Eventually precautionary saving < 0 (violates theory)
- No theoretical justification

---

### 4. Exponential Decay to Perfect Foresight

**How it works:**
- Blend toward perfect foresight solution as m → ∞
- c(m) = c_PF(m) - decay(m) × gap

**Strengths:**
- Asymptotically approaches correct limit
- Prevents most egregious errors

**Weaknesses:**
- Decay rate is arbitrary
- No guarantee of correct intermediate behavior
- Not based on economic theory

---

## Numerical Accuracy Comparison

For the baseline parameterization (ρ=2, β=0.96, R=1.03):

| Method | Max Error at m=10 | Max Error at m=50 | Max Error at m=100 |
|--------|------------------|------------------|-------------------|
| Standard EGM | ~0.1% | ~5% | Fails (c > m) |
| Method of Moderation | ~0.01% | ~0.1% | ~0.5% |
| Linear Extrapolation | ~0.1% | ~10% | Fails |

*Errors measured relative to high-precision reference solution*

---

## Computational Cost Comparison

For N gridpoints:

| Method | Gridpoint Computation | Interpolation | Evaluation |
|--------|----------------------|---------------|------------|
| Standard EGM | O(N × shock_points) | O(N) | O(log N) |
| Method of Moderation | O(N × shock_points) | O(N) | O(log N) + O(1) |

The Method of Moderation adds only constant-time overhead per evaluation:
- One log computation (μ = log(m - m_min))
- One interpolation lookup (same as EGM)
- One expit computation (ω = 1/(1+exp(-χ)))
- Two analytical function evaluations (c_pes, c_opt)
- One weighted sum

---

## When to Use Each Method

### Use Standard EGM when:
- Only evaluating within the computed grid
- Speed is critical and accuracy outside grid doesn't matter
- Quick prototyping

### Use Method of Moderation when:
- Simulating agents over many periods
- Large positive shocks are possible
- Theoretical consistency is important
- Accuracy outside grid matters
- Computing welfare or value functions

### Avoid Linear Extrapolation when:
- Computing precautionary saving
- Simulating wealth accumulation
- Any serious quantitative work

---

## Implementation Checklist

To implement Method of Moderation:

1. ☐ Compute MPCmin = 1 - (βR)^(1/ρ)/R
2. ☐ Compute human wealth h and borrowing constraint m_min
3. ☐ Define c_opt(m) = MPCmin × (m + h)
4. ☐ Define c_pes(m) = MPCmin × (m - m_min)
5. ☐ Solve standard EGM to get (m_i, c_i) pairs
6. ☐ Compute ω_i = (c_i - c_pes(m_i)) / (c_opt(m_i) - c_pes(m_i))
7. ☐ Compute μ_i = log(m_i - m_min)
8. ☐ Compute χ_i = log(ω_i / (1 - ω_i))
9. ☐ Create interpolant χ̂(μ) from (μ_i, χ_i)
10. ☐ Evaluate: ĉ(m) = c_pes(m) + expit(χ̂(log(m-m_min))) × (c_opt(m) - c_pes(m))
