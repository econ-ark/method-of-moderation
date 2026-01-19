---
# MyST frontmatter (inherits authors, bibliography from myst.yml)
title: Illustrative Notebook
short_title: Notebook
description: A pedagogical introduction to the Method of Moderation with interactive code examples.
# Jupytext configuration
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

(notebook:illustrative)=

# The Method of Moderation: Illustrative Notebook

```python tags=["hide-input"]
# Import Econ-ARK styling and display header
from style import (
    HEADER_HTML_NOTEBOOK,
    apply_ark_style,
    apply_notebook_css,
)

# Apply Econ-ARK branding and styling
apply_ark_style()
apply_notebook_css()


# Display Econ-ARK header (for Jupyter notebooks)
from IPython.display import HTML, display

display(HTML(HEADER_HTML_NOTEBOOK))
```

**Author:** <span style="color: var(--ark-lightblue); font-weight: bold;">Alan Lujan</span>, <span style="color: var(--ark-blue); font-weight: bold;">Johns Hopkins University</span>

This notebook provides a pedagogical introduction to the Method of Moderation (MoM), a novel technique for solving consumption-saving models with superior accuracy and stability. We begin by motivating the problem that MoM solves: the "extrapolation problem" inherent in sparse-grid implementations of the Endogenous Grid Method (EGM). We then build the theoretical foundations for MoM, demonstrating how it leverages analytical bounds to ensure economically sensible behavior across the entire state space.

## Model Foundations: The Friedman-Muth Income Process

We adopt the canonical framework of {cite:t}`Friedman1957` and {cite:t}`Muth1960`: an agent receiving labor income subject to permanent and transitory shocks. The model implementation follows {cite:t}`SolvingMicroDSOPs`.

## The Extrapolation Problem in Consumption-Saving Models

The Method of Moderation (MoM) addresses the **extrapolation problem** in sparse-grid EGM implementations. EGM computes optimal consumption at finite grid points, but evaluating the policy function outside this grid via linear extrapolation can predict **negative precautionary saving**, violating established theory {cite:p}`Leland1968,Sandmo1970,Kimball1990`.

MoM operates in a transformed space defined by two analytical bounds from the buffer-stock saving literature {cite:p}`Carroll1997,StachurskiToda2019JET,MST2020JET`:

1. **The Optimist**: Ignores future income risk.
2. **The Pessimist**: Assumes worst-case income realizations.

The true consumption function lies between these extremes. MoM interpolates a **moderation ratio** guaranteed to remain within bounds, producing solutions that are economically coherent across the entire state space.

```python
from __future__ import annotations

from moderation import (
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
    IndShockMoMStochasticRConsumerType,
)
from plotting import (
    GridType,
    plot_consumption_bounds,
    plot_cusp_point,
    plot_moderation_ratio,
    plot_mom_mpc,
    plot_precautionary_gaps,
    plot_stochastic_bounds,
    plot_value_functions,
)

# Model setup: Consumer with income uncertainty
params = {
    "CRRA": 2.0,
    "DiscFac": 0.96,
    "Rfree": [1.02],
    "TranShkStd": [1.0],
    "cycles": 1,
    "LivPrb": [1.0],
    "vFuncBool": True,
    "CubicBool": True,
    "PermGroFac": [1.0],
    "PermShkStd": [0.0],
    "TranShkCount": 7,
    "UnempPrb": 0.0,
    "BoroCnstArt": None,
}

# Dense grid for "truth" solution (high precision)
dense_grid = {"aXtraMin": 0.001, "aXtraMax": 40, "aXtraCount": 500, "aXtraNestFac": 3}

# Sparse grid for practical comparison (5 points only)
sparse_grid = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 5, "aXtraNestFac": -1}

# Solve three versions: Truth (dense EGM), Sparse EGM, Sparse MoM
IndShockTruth = IndShockEGMConsumerType(**(params | dense_grid))
IndShockTruth.solve()
IndShockTruthSol = IndShockTruth.solution[0]

# Unpack theoretical bounds (same for all methods)
TruthOpt = IndShockTruthSol.Optimist
TruthPes = IndShockTruthSol.Pessimist
TruthTight = IndShockTruthSol.TighterUpperBound

# Sparse EGM solution (standard approach)
IndShockEGMApprox = IndShockEGMConsumerType(**(params | sparse_grid))
IndShockEGMApprox.solve()
IndShockEGMApproxSol = IndShockEGMApprox.solution[0]

# Sparse MoM solution (same grid, different method)
IndShockMoMApprox = IndShockMoMConsumerType(**(params | sparse_grid))
IndShockMoMApprox.solve()
IndShockMoMApproxSol = IndShockMoMApprox.solution[0]

# Grid parameters for plotting
mNrmMax = IndShockMoMApproxSol.mNrmMin + IndShockMoMApprox.aXtraGrid.max()
```

## Consumption Function Analysis

The first set of figures will focus on the core of the consumption-saving problem: the consumption function $\mathbf{c}(m)$, which maps market resources $m$ to consumption. We will demonstrate the extrapolation problem inherent in the standard EGM and show how the Method of Moderation resolves it by respecting theoretical bounds.

### Figure 1: The EGM Extrapolation Problem

The **precautionary saving gap** (optimist minus realist consumption) must be positive: income risk induces additional saving. As demonstrated in the paper and shown in [](#fig:egm-extrapolation-problem), standard EGM violates this constraint when extrapolating {cite:p}`carrollEGM`.

```python
# | label: fig:egm-extrapolation-problem

# Figure 1: EGM Extrapolation Failure
plot_precautionary_gaps(
    truth_solution=IndShockTruthSol,
    approx_solutions=IndShockEGMApproxSol,
    title="Figure 1: EGM Extrapolation Failure",
    subtitle="EGM Extrapolation Failure: Negative Precautionary Saving",
)
```

```{attention} EGM vs MoM Extrapolation
Figure 1 demonstrates the core difference: **EGM** can predict negative precautionary saving (violating economic theory), while **MoM** maintains positive precautionary saving through asymptotically linear extrapolation.
```

```{seealso} EGM in the Literature
:class: dropdown

The Endogenous Grid Method ({cite:t}`carrollEGM`) is a powerful and widely-used tool in computational economics. Its applications have been extended to solve multi-dimensional problems {cite:p}`BarillasFV2007`, models with occasionally binding constraints {cite:p}`HintermaierKoeniger2010`, non-smooth and non-concave problems {cite:p}`Fella2014`, and discrete-continuous choice models {cite:p}`IskhakovRustSchjerning2017`. For a comprehensive treatment of the theory and practice of EGM, see {cite:p}`White2015`.

A critical step in implementing any numerical solution is the discretization of continuous stochastic processes. Standard methods for discretizing income shocks include those proposed by {cite:p}`TauchenHussey1991` and {cite:p}`tauchen1986`.
```

### Figure 2: Truth Bounded by Theory

The optimal consumption function is bounded by analytical optimist and pessimist solutions. [](#fig:truth-bounded-by-theory) ({ref}`Figure 2 <fig:IntExpFOCInvPesReaOptNeedHi>` in the paper) confirms the high-precision "truth" lies between these bounds.

```python
# | label: fig:truth-bounded-by-theory

# Figure 2: Truth Bounded by Theory
plot_consumption_bounds(
    solution=IndShockTruthSol,
    title="Figure 2: Truth Bounded by Economic Theory",
    subtitle="True Consumption Always Lies Between Theoretical Bounds",
    show_grid_points=False,  # Truth solution has too many grid points to display clearly
)
```

```{important} Key Theoretical Insight
The constraint $\underline{\mathbf{c}}(m) \leq \hat{\mathbf{c}}(m) \leq \bar{\mathbf{c}}(m)$ is the foundation for MoM {cite:p}`CarrollShanker2024`. The lower bound arises from the natural borrowing constraint in incomplete markets models {cite:p}`Aiyagari1994,Huggett1993,Zeldes1989,Deaton1991`.
```

### Figure 3: Method of Moderation Solution

MoM interpolates a **moderation ratio** in a transformed space that guarantees bound compliance. [](#fig:mom-solution) ({ref}`Figure 3 <fig:ExtrapProblemSolved>` in the paper) shows MoM maintains positive precautionary saving even far beyond the computed grid.

```python
# | label: fig:mom-solution

# Figure 3: Method of Moderation Success
plot_precautionary_gaps(
    truth_solution=IndShockTruthSol,
    approx_solutions=IndShockMoMApproxSol,
    title="Figure 3: Method of Moderation Solves Extrapolation",
    subtitle="MoM Maintains Positive Precautionary Saving",
)
```

```{tip} Method of Moderation Success
MoM maintains positive precautionary saving when extrapolating far beyond its computed grid, matching high-precision truth.
```

MoM builds on EGM's computational efficiency while enforcing theoretical bounds. See [the paper's algorithm](../content/paper/moderation_letters.md#the-method-of-moderation).

(notebook:algorithm)=

```{hint} MoM Algorithm Details
:class: dropdown

MoM steps (notation matches the paper):
1. Solve standard EGM for realist consumption at gridpoints
2. Transform to $\boldsymbol{\mu} = \log(m - \underline{m})$
3. Compute $\boldsymbol{\omega}(\boldsymbol{\mu}) = (\hat{\mathbf{c}} - \underline{\mathbf{c}})/(\bar{\mathbf{c}} - \underline{\mathbf{c}}) \in [0,1]$ (Eq. {eq}`eq:modRte`)
4. Apply logit: $\boldsymbol{\chi} = \log(\boldsymbol{\omega}/(1-\boldsymbol{\omega}))$
5. Interpolate $\boldsymbol{\chi}(\boldsymbol{\mu})$ with derivatives
6. Reconstruct: $\hat{\mathbf{c}} = \underline{\mathbf{c}} + \boldsymbol{\omega} \cdot (\bar{\mathbf{c}} - \underline{\mathbf{c}})$

This ensures bound compliance via asymptotically linear extrapolation, as derived in the paper. HARK uses **cubic Hermite interpolation** {cite:p}`Fritsch1980,FritschButland1984` for accuracy; see {cite:p}`Santos2000,JuddMaliarMaliar2017` on function approximation and error bounding.
```

```{note} The Transformation
The logit maps $\boldsymbol{\omega} \in (0,1)$ to $\boldsymbol{\chi} \in (-\infty, +\infty)$ and becomes asymptotically linear with positive slope $\partial \boldsymbol{\chi} / \partial \boldsymbol{\mu} > 0$ as wealth increases.
```

### Figure 4: MoM Consumption Function

[](#fig:mom-consumption-function) ({ref}`Figure 4 <fig:IntExpFOCInvPesReaOptNeed45>` in the paper) shows MoM consumption between optimist and pessimist bounds, plus a **tighter upper bound** derived from $\bar{\boldsymbol{\kappa}}$ near the borrowing constraint {cite:p}`Carroll2001MPCBound,MaToda2021SavingRateRich,CarrollToche2009`. The cusp intersection is given by {eq}`eq:mNrmCusp`.

```python
# | label: fig:mom-consumption-function

# Figure 4: MoM Consumption Function
plot_consumption_bounds(
    solution=IndShockMoMApproxSol,
    title="Figure 4: MoM Consumption Function",
    subtitle="MoM Consumption Respects Theoretical Bounds",
    m_max=3.0,
    show_tight_bound=True,
)
```

**Bound Preservation**: MoM consumption stays within theoretical bounds and below tight bound.

### Figure 5: Direct Method Comparison

[](#fig:direct-comparison) compares EGM and MoM precautionary gaps against the high-precision truth. Both approximations use identical 5-point sparse grids; the difference in extrapolation behavior is attributable solely to the method.

```python
# | label: fig:direct-comparison

# Figure 5: Direct Method Comparison
plot_precautionary_gaps(
    truth_solution=IndShockTruthSol,
    approx_solutions=[IndShockEGMApproxSol, IndShockMoMApproxSol],
    title="Figure 5: Direct Method Comparison",
    subtitle="EGM vs MoM Extrapolation Performance",
)
```

```{important} The Decisive Advantage
Given identical sparse grids, EGM produces negative precautionary saving while MoM maintains positive values matching the truth.
```

## Method of Moderation Framework

### Figure 6: Moderation Ratio Function $\boldsymbol{\omega}(m)$

::::{admonition} Definition: The Moderation Ratio
:class: note

The **moderation ratio** (Eq. {eq}`eq:modRte`):

$$
\boldsymbol{\omega}(m) = \frac{\hat{\mathbf{c}}(m) - \underline{\mathbf{c}}(m)}{\bar{\mathbf{c}}(m) - \underline{\mathbf{c}}(m)} \in (0,1)
$$

This ratio is strictly between 0 and 1 due to prudence {cite:p}`CarrollKimball1996`. At $\boldsymbol{\omega} = 0$ the realist behaves like the pessimist (maximum precautionary saving); at $\boldsymbol{\omega} = 1$ like the optimist (no precautionary saving). [](#fig:moderation-ratio) plots this ratio across wealth levels.
::::

```python
# | label: fig:moderation-ratio

# Figure 6: Moderation Ratio Function
plot_moderation_ratio(
    solution=IndShockMoMApproxSol,
    title=r"Figure 6: Consumption Moderation Ratio $\omega(m)$",
    subtitle="Wealth-Dependent Moderation Between Bounds",
    m_max=50,
    grid_type=GridType.CONSUMPTION,
)
```

```{note} Economic Interpretation of $\boldsymbol{\omega}(m)$
$\boldsymbol{\omega} \to 1$ at high wealth (approaches optimist), $\boldsymbol{\omega} \to 0$ at low wealth (approaches pessimist). This monotonic pattern ensures proper economic behavior across the wealth distribution.
```

### Figure 7: The Logit Transformation

::::{admonition} Definition: The Logit Transformation
:class: note

The **logit transformation** (Eq. {eq}`eq:chi`) maps the bounded ratio to an unbounded space:

$$
\boldsymbol{\chi}(\boldsymbol{\mu}) = \log\left(\frac{\boldsymbol{\omega}(\boldsymbol{\mu})}{1 - \boldsymbol{\omega}(\boldsymbol{\mu})}\right)
$$

where $\boldsymbol{\mu} = \log(m - \underline{m})$. As [](#fig:logit-transformation) shows, $\boldsymbol{\chi}$ is nearly linear, making it well-suited for interpolation.
::::

```python
# The logit transformation: 𝛘 = log(𝛚/(1-𝛚))
# The inverse (expit): 𝛚 = 1/(1 + exp(-𝛘))
# These transformations are demonstrated in the symbolic notebook.
print("See method-of-moderation-symbolic.ipynb for SymPy demonstrations")
```

```{important} Why Asymptotic Linearity Matters
As $\boldsymbol{\mu} \to \infty$, $\boldsymbol{\chi}$ becomes linear with slope $\partial \boldsymbol{\chi} / \partial \boldsymbol{\mu} > 0$. This prevents extrapolation errors, ensures smooth convergence to the optimist bound, and maintains numerical stability.
```

```{note} Properties of $\boldsymbol{\chi}(\boldsymbol{\mu})$
Unbounded domain $(-\infty, \infty)$, monotonically increasing, asymptotically linear. $\boldsymbol{\chi} > 0$ indicates behavior closer to optimist; $\boldsymbol{\chi} < 0$ closer to pessimist.
```

## Function Properties and Bounds

### Figure 8: MoM MPC Bounded by Theory

The **MPC** ($\partial c / \partial m$) is bounded between $\underline{\boldsymbol{\kappa}}$ (optimist) and $\bar{\boldsymbol{\kappa}}$ (at the borrowing constraint), as detailed in {ref}`the paper <a-tighter-upper-bound>` and Eq. {eq}`eq:MPCModeration` {cite:p}`Carroll2001MPCBound`. [](#fig:mpc-bounds) confirms MoM respects these bounds.

```{tip} Policy Applications
Bounded MPC estimates prevent nonsensical policy multipliers in DSGE models. MoM ensures economically meaningful MPCs for policy analysis.
```

```python
# | label: fig:mpc-bounds

# Figure 8: MoM MPC Bounds
plot_mom_mpc(
    solution=IndShockMoMApproxSol,
    title="Figure 8: MoM MPC Bounded by Theory",
    subtitle="MoM MPC Stays Within Theoretical Bounds",
)
```

```{hint} MPC Economic Interpretation
MoM MPC declines with $m$: poor consumers spend windfalls immediately (MPC $\to \bar{\boldsymbol{\kappa}}$), wealthy consumers save them (MPC $\to \underline{\boldsymbol{\kappa}}$), reflecting diminishing marginal utility.
```

### Figure 9: Value Functions Bounded by Theory

The **value function** $\mathbf{v}(m)$ is also bounded by optimist and pessimist solutions {cite:p}`Aiyagari1994,Huggett1993`. [](#fig:value-functions) compares truth, EGM, and MoM value functions.

```python
# | label: fig:value-functions

# Figure 9: Value Functions
plot_value_functions(
    truth_solution=IndShockTruthSol,
    title="Figure 9: Value Functions Bounded by Economic Theory",
    subtitle="Value Function: True Solution vs Sparse Approximations",
    egm_solution=IndShockEGMApproxSol,
    mom_solution=IndShockMoMApproxSol,
)
```

```{note} Value Function Interpretation
Both approximations use 5 sparse grid points. MoM stays closer to truth while respecting bounds. The optimist-pessimist gap represents the **cost of uncertainty**.
```

```{hint} Economic Insight
Uncertainty matters most at low wealth where buffers are small; the optimist-pessimist gap narrows with wealth as assets provide natural insurance.
```

### Figure 10: Inverse Value Functions ${\scriptsize \boldsymbol{\Lambda}}(m)$

The **inverse value function** ${\scriptsize \boldsymbol{\Lambda}}(m) = \mathbf{u}^{-1}(\mathbf{v}(m))$ gives the consumption equivalent of lifetime utility. It is more linear than $\mathbf{v}(m)$ near the borrowing constraint, making it better suited for interpolation. [](#fig:inverse-value-functions) compares the three solutions.

```python
# | label: fig:inverse-value-functions

# Figure 10: Inverse Value Functions
plot_value_functions(
    truth_solution=IndShockTruthSol,
    title="Figure 10: Inverse Value Functions",
    subtitle="Inverse Value Function: Consumption-Equivalent Utility",
    inverse=True,
    egm_solution=IndShockEGMApproxSol,
    mom_solution=IndShockMoMApproxSol,
)
```

```{note} Inverse Value Function Interpretation
The inverse transformation converts utility to consumption units: ${\scriptsize \boldsymbol{\Lambda}}(5) = 0.8$ means 5 units of wealth provides lifetime utility equivalent to consuming 0.8 forever. HARK uses this more linear representation for interpolation.
```

### Figure 11: Value Function Moderation Ratio

MoM applies to any bounded function. The **inverse value function moderation ratio** (Eq. {eq}`eq:valModRteReal`):

$$
\boldsymbol{\Omega}(m) = \frac{\hat{{\scriptsize \boldsymbol{\Lambda}}}(m) - \underline{{\scriptsize \boldsymbol{\Lambda}}}(m)}{\bar{{\scriptsize \boldsymbol{\Lambda}}}(m) - \underline{{\scriptsize \boldsymbol{\Lambda}}}(m)}
$$

follows the same pattern as the consumption ratio, as shown in [](#fig:value-moderation-ratio).

```python
# | label: fig:value-moderation-ratio

# Figure 11: Value Function Moderation Ratio
plot_moderation_ratio(
    solution=IndShockMoMApproxSol,
    title=r"Figure 11: Value Function Moderation Ratio $\Omega(m)$",
    subtitle="Value Function Moderation Between Bounds",
    m_max=10,
    grid_type=GridType.VALUE,
)
```

```{hint} Value Function Moderation Interpretation
$\boldsymbol{\Omega} \to 1$ at high wealth (low uncertainty cost), $\boldsymbol{\Omega} \to 0$ at low wealth (high uncertainty cost). The pattern confirms uncertainty's welfare cost diminishes with wealth.
```

### Figure 12: Cusp Point Visualization

The **cusp point** (Eq. {eq}`eq:mNrmCusp`) is where optimist and tighter upper bounds intersect:

$$
m^* = \underline{m} + \frac{\underline{\boldsymbol{\kappa}} \cdot \Delta h}{\bar{\boldsymbol{\kappa}} - \underline{\boldsymbol{\kappa}}}
$$

Below the cusp, the tighter bound ($\bar{\boldsymbol{\kappa}}$ slope) constrains; above, the optimist bound constrains. See `IndShockMoMCuspConsumerType` for the three-piece implementation.

```python
# | label: fig:cusp-point

# Figure 12: Cusp Point Visualization
plot_cusp_point(
    solution=IndShockMoMApproxSol,
    title="Figure 12: Cusp Point and Upper Bound Envelope",
    subtitle="Where Optimist and Tighter Bounds Intersect",
    m_max=8,
)
```

```{hint} Cusp Point Interpretation
Below cusp: MPC near $\bar{\boldsymbol{\kappa}}$, tighter bound constrains. Above cusp: behavior approaches optimist ($\underline{\boldsymbol{\kappa}}$), optimist bound constrains. The envelope is the minimum of both bounds.
```

## Further Extensions: Stochastic Rate of Return

With i.i.d. returns, {cite:t}`Samuelson1969` and {cite:t}`Merton1969,Merton1971` show the consumption function remains linear for consumers without labor income. MoM extends directly by substituting the stochastic-return MPC {cite:p}`BBZ2016SkewedWealth,CRRA-RateRisk`. Serially correlated returns remain for future research.

### Figure 13: Stochastic Returns Comparison

[](#fig:stochastic-bounds) compares bounds under deterministic and stochastic returns using `IndShockMoMStochasticRConsumerType`.

```python
# | label: fig:stochastic-bounds

# Solve model with stochastic returns (mean-preserving spread)
stoch_params = params.copy()
stoch_params["RiskyAvg"] = params["Rfree"][0]  # Same mean as Rfree (scalar)
stoch_params["RiskyStd"] = (
    0.20  # 20% standard deviation (must satisfy β*E[R^{1-ρ}] < 1)
)

IndShockStochR = IndShockMoMStochasticRConsumerType(**stoch_params)
IndShockStochR.solve()
IndShockStochRSol = IndShockStochR.solution[0]

# Figure 13: Stochastic Returns Comparison
plot_stochastic_bounds(
    solution=IndShockStochRSol,
    title="Figure 13: Deterministic vs Stochastic Return Bounds",
    subtitle="Effect of Return Uncertainty on Consumption Bounds",
    m_max=10,
)
```

```{hint} Stochastic Returns Interpretation
Deterministic optimist uses $\underline{\boldsymbol{\kappa}} = 1 - (\beta \text{R})^{1/\rho}$; stochastic optimist uses $\underline{\boldsymbol{\kappa}} = 1 - (\beta \mathbf{\mathbb{E}}[\mathbf{R}^{1-\rho}])^{1/\rho}$. Return uncertainty raises MPC and narrows the feasible region. See {ref}`stochastic-returns-mgf-derivation` for the MGF derivation.
```

## Summary

MoM solves the extrapolation problem by interpolating a **moderation ratio** via an asymptotically linear **logit transformation**, ensuring solutions respect theoretical bounds by construction.

**Key Advantages**:

* **Theoretical Consistency**: Prevents negative precautionary saving.
* **Numerical Stability**: Robust solutions via bounded transformations.
* **Computational Efficiency**: Builds on EGM with minimal overhead.

For complete theoretical development see {ref}`the-method-of-moderation`.


## Symbolic Mathematics

For symbolic manipulation of the Method of Moderation equations using SymPy, see:

- **`method-of-moderation-symbolic.ipynb`** — Interactive symbolic demonstrations
- **`metadata/equations.py`** — Full SymPy module with Unicode symbols
- **`metadata/equations.json`** — JSON export for programmatic access
