---
# ARCHIVED: Original version preserved for reference.
# Current authors, affiliations, and macros are defined in myst.yml and macros.yml
title: The Method of Moderation
authors:
  - name: Christopher D Carroll
    email: ccarroll@jhu.edu
    orcid: 0000-0003-3732-9312
    affiliations:
      - jhu
  - name: Karsten Chipeniuk
    affiliations:
      - rbnz
  - name: Kiichi Tokuoka
    email: ktokuoka@imf.org
    affiliations:
      - ecb
      - imf
  - name: Weifeng Wu
    affiliations:
      - fannie_mae
affiliations:
  - id: jhu
    name: Johns Hopkins University
    department: Department of Economics
    address: Baltimore, MD
    url: http://econ.jhu.edu/people/ccarroll/
  - id: rbnz
    name: Reserve Bank of New Zealand
  - id: ecb
    name: European Central Bank
  - id: imf
    name: International Monetary Fund
    address: Washington, DC
  - id: fannie_mae
    name: Fannie Mae
    address: Washington, DC
date: 2024-11-23
bibliography:
  - references.bib
acknowledgments: |
  The views presented in this paper are those of the authors, and should not be attributed to the International Monetary Fund, its Executive Board, or management, or to the European Central Bank.
math:
  # Time and period notation
  "\\prd": "t"
  "\\trmT": "T"
  "\\EndStg": "T"
  "\\EndPrd": "T"
  "\\EndPrdNxt": "T+1"
  "\\prdT": "T"
  "\\prdLst": "T-1"

  # Level variables (unnormalized)
  "\\aLvl": "A"
  "\\mLvl": "M"
  "\\cLvl": "C"
  "\\bLvl": "B"
  "\\yLvl": "Y"
  "\\pLvl": "P"

  # Normalized variables
  "\\mNrm": "m"
  "\\cNrm": "c"
  "\\aNrm": "a"
  "\\kNrm": "k"
  "\\bNrm": "b"
  "\\hNrm": "h"
  "\\yNrm": "y"
  "\\pNrm": "p"

  # Functions
  "\\uFunc": "u"
  "\\vFunc": "v"
  "\\cFunc": "c"
  "\\cFuncAbove": "\\bar{c}"
  "\\cFuncBelow": "\\underline{c}"
  "\\chiFunc": "\\chi"
  "\\vInv": "\\Lambda"

  # Economic parameters
  "\\DiscFac": "\\beta"
  "\\CRRA": "\\rho"
  "\\MPC": "\\kappa"
  "\\MPCmin": "\\underline{\\kappa}"
  "\\MPCmax": "\\bar{\\kappa}"
  "\\MPCFunc": "\\kappa"
  "\\tranShkEmp": "\\theta"
  "\\tranShkEmpMin": "\\theta_{\\min}"
  "\\permShk": "\\psi"
  "\\Rfree": "R"
  "\\Risky": "\\tilde{R}"
  "\\risky": "r"
  "\\PermGroFac": "G"
  "\\RNrmByG": "\\mathcal{R}"

  # Bounds and modifiers
  "\\Min": "\\underline"
  "\\Max": "\\overline"
  "\\aboveMin": "\\blacktriangle"
  "\\hEndMin": "\\underline{h}"
  "\\mtCusp": "m^\\#"
  "\\Aprx": "\\check"
  "\\Koppa": "Q"
  "\\koppa": "q"

  # Mathematical operators and symbols
  "\\Ex": "\\mathbb{E}"
  "\\ExEndPrd": "\\mathbb{E}_t"
  "\\ExEndStg": "\\mathbb{E}_t"
  "\\Nrml": "\\mathcal{N}"
  "\\std": "\\sigma"
  "\\PDVCoverc": "\\mathcal{C}"
  "\\vctr": "\\mathbf"
  "\\eprem": "\\pi"
  "\\rfree": "r"
abstract: |
  In a risky world, a pessimist assumes the worst will happen.
  Someone who ignores risk altogether is an optimist.
  Consumption decisions are mathematically simple for both the pessimist and the optimist because both behave as if they live in a riskless world.
  A realist (someone who wants to respond optimally to risk) faces a much more difficult problem, but (under standard conditions) will choose a level of spending somewhere between pessimist's and the optimist's.
  We use this fact to redefine the space in which the realist searches for optimal consumption rules.
  The resulting solution accurately represents the numerical consumption rule over the entire interval of feasible wealth values with remarkably few computations.
keypoints:
  - Dynamic Stochastic Optimization
parts:
  jel_codes: D14; C61; G11
---

# The Method of Moderation

## Introduction

Solving a consumption, investment, portfolio choice, or similar intertemporal
optimization problem using numerical methods generally requires the modeler to
choose how to represent a policy or value function. In the stochastic case,
where analytical solutions are generally not available, a common approach is to
use low-order polynominal splines that exactly match the function (and maybe
some derivatives) at a finite set of gridpoints, and then to assume that
interpolated or extrapolated versions of that spline represent the function well
at the continuous infinity of unmatched points.

This paper argues that a better approach in the standard consumption problem is
to rely upon the fact that without uncertainty, the optimal consumption function
has a simple analytical solution. The key insight is that, under standard
assumptions, the consumer who faces an uninsurable labor income risk will
consume less than a consumer with the same path for expected income but who does
not perceive any uncertainty as being attached to that future income. The
'realistic' consumer, who _does_ perceive the risks, will engage in
'precautionary saving,' so the perfect foresight riskless solution provides an
upper bound to the solution that will actually be optimal. A lower bound is
provided by the behavior of a consumer who has the subjective belief that the
future level of income will be the worst that it can possibly be. This consumer,
too, behaves according to the convenient analytical perfect foresight solution,
but his certainty is that of a pessimist perfectly confident in his pessimism.

Using results from {cite:t}`BufferStockTheory`, we show how to use these upper
and lower bounds to tightly constrain the shape and characteristics of the
solution to problem of the 'realist.' Imposition of these constraints can
clarify and speed the solution of the realist's problem.

After showing how to use the method in the baseline case, we show how refine it
to encompass an even tighter theoretical bound, and how to extend it to solve a
problem in which the consumer faces both labor income risk and rate-of-return
risk.

## The Realist's Problem

We assume that truly optimal behavior in the problem facing the consumer who
understands all his risks is captured by

```{math}
:label: eq:MaxProb
\max~\Ex_{\prd}\left[\sum_{n=0}^{\trmT-\prd}\DiscFac^{n} \uFunc(\cLvl_{\prd+n})\right].
```

subject to

```{math}
:label: eq:DBCLevel
\begin{aligned}
\aLvl_{\prd}  &= \mLvl_{\prd}-\cLvl_{\prd} \\
\bLvl_{\prd+1}  &= \aLvl_{\prd} \Rfree_{\prd+1} \\
\yLvl_{\prd+1}  &= \pLvl_{\prd+1}\tranShkEmp_{\prd+1} \\
\mLvl_{\prd+1}  &= \bLvl_{\prd+1} + \yLvl_{\prd+1}
\end{aligned}
```

where

$$
\begin{aligned}
\DiscFac &\text{ - pure time discount factor} \\
\aLvl_{\prd} &\text{ - assets after all actions have been accomplished in period } t \\
\bLvl_{\prd+1} &\text{ - `bank balances' (nonhuman wealth) at the beginning of } t+1 \\
\cLvl_{\prd} &\text{ - consumption in period } t \\
\mLvl_{\prd} &\text{ - `market resources' available for consumption (`cash-on-hand')} \\
\pLvl_{\prd+1} &\text{ - `permanent labor income' in period } t+1 \\
\Rfree_{\prd+1} &\text{ - interest factor } (1+\rfree_{\prd+1}) \text{ from period } t \text{ to } t+1 \\
\yLvl_{\prd+1} &\text{ - noncapital income in period } t+1.
\end{aligned}
$$

and the exogenous variables evolve according to

```{math}
:label: eq:ExogVars
\begin{aligned}
\pLvl_{\prd+1} &= \PermGroFac_{\prd+1}\pLvl_{\prd} && \text{~~ -- permanent labor income dynamics} \\
\log ~ \tranShkEmp_{\prd+n} &\sim ~\Nrml(-\std_{\tranShkEmp}^{2}/2,\std_{\tranShkEmp}^{2}) && \text{~~ -- lognormal transitory shocks}~\forall~n>0 .
\end{aligned}
```

It turns out (see {cite:t}`SolvingMicroDSOPs` for a proof) that this problem can
be rewritten in a more convenient form in which choice and state variables are
normalized by the level of permanent income, e.g., using nonbold font for
normalized variables, $\mNrm_{\prd}=\cLvl_{\prd}/\pLvl_{\prd}$. When that is
done, the Bellman equation for the transformed version of the consumer's problem
is

```{math}
:label: eq:vNormed
\begin{aligned}
\vFunc_{\prd}(\mNrm_{\prd}) &= \max_{{\cNrm}_{\prd}} ~~ \uFunc(\cNrm_{\prd})+\DiscFac \Ex_{\prd}[ \PermGroFac_{\prd+1}^{1-\CRRA}\vFunc_{\prd+1}(\mNrm_{\prd+1})] \\
&\text{s.t.} \\
\aNrm_{\prd} &= \mNrm_{\prd}-\cNrm_{\prd} \\
\kNrm_{\prd+1} &= \aNrm_{\prd} \\
\bNrm_{\prd+1} &= \underbrace{\left(\Rfree/\PermGroFac_{\prd+1}\right)}_{\equiv \RNrmByG_{\prd+1}}\kNrm_{\prd+1} \\
\mNrm_{\prd+1} &= \bNrm_{\prd+1}+\tranShkEmp_{\prd+1},
\end{aligned}
```

and because we have not imposed a liquidity constraint, the solution satisfies
the Euler equation

```{math}
:label: eq:cEuler
\uFunc^{\cNrm}(\cNrm_{\prd}) = \ExEndPrd[\DiscFac \Rfree \PermGroFac_{\prd+1}^{-\CRRA}\uFunc^{\cNrm}(\cNrm_{\prd+1})].
```

For the remainder of the paper we will assume that permanent income
$\pLvl_{\prd}$ grows by a constant factor $\PermGroFac$ and is not subject to
stochastic shocks. (The generalization to the case with permanent shocks is
straightforward.)

## Benchmark: The Method of Endogenous Gridpoints

For comparison to our new solution method, we use the endogenous gridpoints
solution to the microeconomic problem presented in {cite:t}`carrollEGM`. That
method computes the level of consumption at a set of gridpoints for market
resources $\mNrm$ that are determined endogenously using the Euler equation. The
consumption function is then constructed by linear interpolation among the
gridpoints thus found.

{cite:t}`SolvingMicroDSOPs` describes a specific calibration of the model and
constructs a solution using five gridpoints chosen to capture the structure of
the consumption function reasonably well at values of $\mNrm$ near the
infinite-horizon target value. (See those notes for details).

Unfortunately, this endogenous gridpoints solution is not very well-behaved
outside the original range of gridpoints targeted by the solution method.
(Though other common solution methods are no better outside their own predefined
ranges). Figure {ref}`fig:ExtrapProblem` demonstrates the point by plotting the
amount of precautionary saving implied by a linear extrapolation of our
approximated consumption rule (the consumption of the perfect foresight consumer
$\cFuncAbove_{\prd-1}$ minus our approximation to optimal consumption under
uncertainty, $\Aprx{\cFunc}_{\prd-1}$). Although theory proves that
precautionary saving is always positive, the linearly extrapolated numerical
approximation eventually predicts negative precautionary saving (at the point in
the figure where the extrapolated locus crosses the horizontal axis).

```{figure} ./images/extrapolation-problem.png
:label: fig:ExtrapProblem
:alt: Graph showing that precautionary saving, approximated with linear extrapolation, incorrectly becomes negative for large market resources.
:align: center
:width: 80%

For Large Enough $m_{\prd-1}$, Predicted Precautionary Saving is Negative (Oops!)
```

This error cannot be fixed by extending the upper gridpoint; in the presence of
serious uncertainty, the consumption rule will need to be evaluated outside of
_any_ prespecified grid (because starting from the top gridpoint, a large enough
realization of the uncertain variable will push next period's realization of
assets above that top; a similar argument applies below the bottom gridpoint).
While a judicious extrapolation technique can prevent this problem from being
fatal (for example by carefully excluding negative precautionary saving), the
problem is often dealt with using inelegant methods whose implications for the
accuracy of the solution are difficult to gauge.

## The Method of Moderation

### The Optimist, the Pessimist, and the Realist

#### The Consumption Function

As a preliminary to our solution, define $\hNrm_{\EndStg}$ as end-of-period
human wealth (the present discounted value of future labor income) for a perfect
foresight version of the problem of a 'risk optimist:' a period-$t$ consumer who
believes with perfect confidence that the shocks will always take their expected
value of 1, $\tranShkEmp_{\prd+n} = \Ex[\tranShkEmp]=1~\forall~n>0$. The
solution to a perfect foresight problem of this kind takes the form^[For a
derivation, see {cite:t}`BufferStockTheory`; $\MPCmin_{\prd}$ is defined therein
as the MPC of the perfect foresight consumer with horizon $T-t$.]

```{math}
:label: eq:cFuncAbove
\cFuncAbove_{\prd}(\mNrm_{\prd}) = (\mNrm_{\prd} + \hNrm_{\EndStg})\MPCmin_{\prd}
```

for a constant minimal marginal propensity to consume $\MPCmin_{\prd}$ given
below.

We similarly define $\hEndMin_{\EndStg}$ as 'minimal human wealth,' the present
discounted value of labor income if the shocks were to take on their worst
possible value in every future period
$\tranShkEmp_{\prd+n} = \tranShkEmpMin ~\forall~n>0$ (which we define as
corresponding to the beliefs of a 'pessimist').

We will call a 'realist' the consumer who correctly perceives the true
probabilities of the future risks and optimizes accordingly.

A first useful point is that, for the realist, a lower bound for the level of
market resources is $\Min{m}_{\prd} = -\hEndMin_{\EndStg}$, because if
$m_{\prd}$ equalled this value then there would be a positive finite chance
(however small) of receiving $\tranShkEmp_{\prd+n}=\tranShkEmpMin$ in every
future period, which would require the consumer to set $c_{\prd}$ to zero in
order to guarantee that the intertemporal budget constraint holds (this is the
multiperiod generalization of the discussion explaining the derivation of the
'natural borrowing constraint' for period $\trmT-1$, $\Min{a}_{\prd-1}$). Since
consumption of zero yields negative infinite utility, the solution to realist
consumer's problem is not well defined for values of
$m_{\prd} < \Min{m}_{\prd}$, and the limiting value of the realist's $c_{\prd}$
is zero as $m_{\prd} \downarrow \Min{m}_{\prd}$.

Given this result, it will be convenient to define 'excess' market resources as
the amount by which actual resources exceed the lower bound, and 'excess' human
wealth as the amount by which mean expected human wealth exceeds guaranteed
minimum human wealth:

$$
\begin{aligned}
\aboveMin \mNrm_{\prd} &= m_{\prd}+\overbrace{\hEndMin_{\EndStg}}^{=-\Min{m}_{\prd}} \\
\aboveMin \hNrm_{\EndStg} &= \hNrm_{\EndStg}-\hEndMin_{\EndStg}.
\end{aligned}
$$

We can now transparently define the optimal consumption rules for the two
perfect foresight problems, those of the 'optimist' and the 'pessimist.' The
'pessimist' perceives human wealth to be equal to its minimum feasible value
$\hEndMin_{\EndStg}$ with certainty, so consumption is given by the perfect
foresight solution

$$
\begin{aligned}
\cFuncBelow_{\prd}(m_{\prd}) &= (m_{\prd}+\hEndMin_{\EndStg})\MPCmin_{\prd} \\
&= \aboveMin \mNrm_{\prd}\MPCmin_{\prd} .
\end{aligned}
$$

The 'optimist,' on the other hand, pretends that there is no uncertainty about
future income, and therefore consumes

$$
\begin{aligned}
\cFuncAbove_{\prd}(m_{\prd}) &= (m_{\prd} +\hEndMin_{\EndStg} - \hEndMin_{\EndStg} + \hNrm_{\EndStg} )\MPCmin_{\prd} \\
&= (\aboveMin \mNrm_{\prd} + \aboveMin \hNrm_{\EndStg})\MPCmin_{\prd} \\
&= \cFuncBelow_{\prd}(m_{\prd})+\aboveMin \hNrm_{\EndStg} \MPCmin_{\prd} .
\end{aligned}
$$

It seems obvious that the spending of the realist will be strictly greater than
that of the pessimist and strictly less than that of the optimist. Figure
{ref}`fig:IntExpFOCInvPesReaOptNeedHiPlot` illustrates the proposition for the
consumption rule in period $\trmT-1$.

```{figure} ./images/moderation-illustrated.png
:label: fig:IntExpFOCInvPesReaOptNeedHiPlot
:alt: Graph showing the realist's consumption function is bounded by the pessimist's (lower) and optimist's (upper) consumption functions.
:align: center
:width: 80%

Moderation Illustrated: $\cFuncBelow < \cFunc < \cFuncAbove$
```

The proof is more difficult than might be imagined, but the necessary work is
done in {cite:t}`BufferStockTheory` so we will take the proposition as a fact
and proceed by manipulating the inequality:

$$
\begin{array}{rcl}
\aboveMin \mNrm_{\prd} \MPCmin_{\prd} < & \cFunc_{\prd}(\Min{m}_{\prd}+\aboveMin \mNrm_{\prd}) & < (\aboveMin \mNrm_{\prd}+\aboveMin \hNrm_{\EndStg})\MPCmin_{\prd} \\
- \aboveMin \mNrm_{\prd} \MPCmin_{\prd} > & -\cFunc_{\prd}(\Min{m}_{\prd}+\aboveMin \mNrm_{\prd}) & > -(\aboveMin \mNrm_{\prd}+\aboveMin \hNrm_{\EndStg})\MPCmin_{\prd} \\
\aboveMin \hNrm_{\EndStg} \MPCmin_{\prd} > & \bar{\cFunc}_{\prd}(\Min{m}_{\prd}+\aboveMin \mNrm_{\prd})-\cFunc_{\prd}(\Min{m}_{\prd}+\aboveMin \mNrm_{\prd}) & > 0 \\
1 > & \underbrace{\left(\frac{\bar{\cFunc}_{\prd}(\Min{m}_{\prd}+\aboveMin \mNrm_{\prd})-\cFunc_{\prd}(\Min{m}_{\prd}+\aboveMin \mNrm_{\prd})}{\aboveMin \hNrm_{\EndStg} \MPCmin_{\prd}}\right)}_{\equiv \Max{\koppa}_{\prd}} & > 0
\end{array}
$$

where the fraction in the middle of the last inequality is the ratio of actual
precautionary saving (the numerator is the difference between perfect-foresight
consumption and optimal consumption in the presence of uncertainty) to the
maximum conceivable amount of precautionary saving (the amount that would be
undertaken by the pessimist who consumes nothing out of any future income beyond
the perfectly certain component).

Defining $\mu_{\prd} = \log \aboveMin \mNrm_{\prd}$ (which can range from
$-\infty$ to $\infty$), the object in the middle of the last inequality is

```{math}
:label: eq:koppa
\Max{\koppa}_{\prd}(\mu_{\prd}) \equiv  \left(\frac{\bar{\cFunc}_{\prd}(\Min{m}_{\prd}+e^{\mu_{\prd}})-\cFunc_{\prd}(\Min{m}_{\prd}+e^{\mu_{\prd}})}{\aboveMin \hNrm_{\EndStg} \MPCmin_{\prd}}\right),
```

and we now define

```{math}
:label: eq:chi
\begin{aligned}
\Max{\chiFunc}_{\prd}(\mu_{\prd}) &= \log \left(\frac{1-\Max{\koppa}_{\prd}(\mu_{\prd})}{\Max{\koppa}_{\prd}(\mu_{\prd})}\right) \\
&= \log \left(1/\Max{\koppa}_{\prd}(\mu_{\prd})-1\right)
\end{aligned}
```

which has the virtue that it is linear in the limit as $\mu_{\prd}$ approaches
$+\infty$.

Given $\Max{\chiFunc}$, the consumption function can be recovered from

```{math}
:label: eq:cFuncHi
\Max{\cFunc}_{\prd} = \bar{\cFunc}_{\prd}-\overbrace{\left(\frac{1}{1+\exp(\Max{\chiFunc}_{\prd})}\right)}^{=\Max{\koppa}_{\prd}} \aboveMin \hNrm_{\EndStg} \MPCmin_{\prd}.
```

Thus, the procedure is to calculate $\Max{\chiFunc}_{\prd}$ at the points
$\vctr{\mu}_{\prd}$ corresponding to the log of the $\aboveMin \vctr{m}_{\prd}$
points defined above, and then using these to construct an interpolating
approximation $\Aprx{\Max{\chiFunc}}_{\prd}$ from which we indirectly obtain our
approximated consumption rule $\Aprx{\Max{\cFunc}}_{\prd}$ by substituting
$\Aprx{\Max{\chiFunc}}_{\prd}$ for $\Max{\chiFunc}$ in equation
{eq}`eq:cFuncHi`.

Because this method relies upon the fact that the problem is easy to solve if
the decision maker has unreasonable views (either in the optimistic or the
pessimistic direction), and because the correct solution is always between these
immoderate extremes, we call our solution procedure the 'method of moderation.'

Results are shown in Figure {ref}`fig:ExtrapProblemSolved`; a reader with very
good eyesight might be able to detect the barest hint of a discrepancy between
the Truth and the Approximation at the far righthand edge of the figure -- a
stark contrast with the calamitous divergence evident in Figure
{ref}`fig:ExtrapProblem`.

```{figure} ./images/method-of-moderation-results.png
:label: fig:ExtrapProblemSolved
:alt: Graph showing that the Method of Moderation produces an accurate extrapolated consumption function that does not predict negative precautionary saving.
:align: center
:width: 80%

Extrapolated $\Aprx{\Max{\cFunc}}_{\prd-1}$ Constructed Using the Method of Moderation
```

#### The Value Function

Often it is useful to know the value function as well as the consumption rule.
Fortunately, many of the tricks used when solving for the consumption rule have
a direct analogue in approximation of the value function.

Consider the perfect foresight (or "optimist's") problem in period $\trmT-1$.
Using the fact that in a perfect foresight model the growth factor for
consumption is $(\Rfree \DiscFac)^{1/\CRRA}$, we can use the fact that
$\cNrm_{\prd} = (\Rfree \DiscFac)^{1/\CRRA} \cNrm_{\prd-1}$ to calculate the
value function in period $\trmT-1$:

$$
\begin{aligned}
\bar{\vFunc}_{\prd-1}(m_{\prd-1}) &\equiv  \uFunc(\cNrm_{\prd-1})+\DiscFac \uFunc(\cNrm_{\prd}) \\
&= \uFunc(\cNrm_{\prd-1})\left(1+\DiscFac ((\DiscFac\Rfree)^{1/\CRRA})^{1-\CRRA}\right) \\
&= \uFunc(\cNrm_{\prd-1})\left(1+(\DiscFac\Rfree)^{1/\CRRA}/\Rfree\right) \\
&= \uFunc(\cNrm_{\prd-1})\underbrace{\mbox{PDV}_{\prd}^{T}(\cNrm)/\cNrm_{\prd-1}}_{\equiv \PDVCoverc_{\prd-1}^{T}}
\end{aligned}
$$

where $\PDVCoverc_{\prd}^{T}=\mbox{PDV}_{\prd}^{T}(\cNrm)$ is the present
discounted value of consumption, normalized by current consumption. Using the
fact demonstrated in {cite:t}`BufferStockTheory` that
$\PDVCoverc_{\prd}=\MPC^{-1}_{\prd}$, a similar function can be constructed
recursively for earlier periods, yielding the general expression

```{math}
:label: eq:vFuncPF
\begin{aligned}
\bar{\vFunc}_{\prd}(m_{\prd}) &= \uFunc(\bar{\cNrm}_{\prd}(\mNrm))\PDVCoverc_{\prd}^{T} \\
&= \uFunc(\bar{c}_{\prd}) \MPCmin_{\prd}^{-1} \\
&= \uFunc((\aboveMin \mNrm_{\prd}+\aboveMin \hNrm_{\EndStg})\MPCmin_{\prd}) \MPCmin_{\prd}^{-1} \\
&= \uFunc(\aboveMin \mNrm_{\prd}+\aboveMin \hNrm_{\EndStg})\MPCmin_{\prd}^{-\CRRA}
\end{aligned}
```

This can be transformed as

$$
\begin{aligned}
\bar{\vInv}_{\prd} &\equiv  \left((1-\CRRA)\bar{\vFunc}_{\prd}\right)^{1/(1-\CRRA)} \\
&= \cNrm_{\prd}(\PDVCoverc_{\prd}^{T})^{1/(1-\CRRA)} \\
&= (\aboveMin \mNrm_{\prd}+\aboveMin \hNrm_{\EndStg})\MPCmin_{\prd}^{-\CRRA/(1-\CRRA)}
\end{aligned}
$$

and since $\PDVCoverc_{\prd}^{T}$ is a constant while the consumption function
is linear, $\bar{\vInv}_{\prd}$ will also be linear.

We apply the same transformation to the value function for the problem with
uncertainty (the "realist's" problem):

$$
\vInv_{\prd} = \left((1-\CRRA)\vFunc_{\prd}(m_{\prd})\right)^{1/(1-\CRRA)}
$$

and an excellent approximation to the value function can be obtained by
calculating the values of $\vInv$ at the same gridpoints used by the consumption
function approximation, and interpolating among those points.

However, as with the consumption approximation, we can do even better if we
realize that the $\bar{\vInv}$ function for the optimist's problem is an upper
bound for the $\vInv$ function in the presence of uncertainty, and the value
function for the pessimist is a lower bound. Analogously to {eq}`eq:koppa`,
define an upper-case

```{math}
:label: eq:KoppaUpper
\hat{\Koppa}_{\prd}(\mu_{\prd}) = \left(\frac{\bar{\vInv}_{\prd}(\Min{m}_{\prd}+e^{\mu_{\prd}})-\vInv_{\prd}(\Min{m}_{\prd}+e^{\mu_{\prd}})}{\aboveMin \hNrm_{\EndStg} \MPCmin_{\prd} (\PDVCoverc_{\prd}^{T})^{1/(1-\CRRA)}}\right)
```

and an upper-case version of the $\chiFunc$ equation in {eq}`eq:chi`:

```{math}
:label: eq:ChiUpper
\begin{aligned}
\hat{\Chi}_{\prd}(\mu_{\prd}) &= \log \left(\frac{1-\hat{\Koppa}_{\prd}(\mu_{\prd})}{\hat{\Koppa}_{\prd}(\mu_{\prd})}\right) \\
&= \log \left(1/\hat{\Koppa}_{\prd}(\mu_{\prd})-1\right)
\end{aligned}
```

and if we approximate these objects then invert them (as above with the
$\Max{\koppa}$ and $\Max{\chiFunc}$ functions) we obtain a very high-quality
approximation to our inverted value function at the same points for which we
have our approximated value function:

$$
\hat{\vInv}_{\prd} = \bar{\vInv}_{\prd}-\overbrace{\left(\frac{1}{1+\exp(\hat{\Chi}_{\prd})}\right)}^{=\hat{\Koppa}_{\prd}} \aboveMin \hNrm_{\EndStg} \MPCmin_{\prd} (\PDVCoverc_{\prd}^{T})^{1/(1-\CRRA) }
$$

from which we obtain our approximation to the value function as

$$
\begin{aligned}
\hat{\vFunc}_{\prd} &= \uFunc(\hat{\vInv}_{\prd}) \\
\hat{\vFunc}^{m}_{\prd} &= \uFunc^{c}(\hat{\vInv}_{\prd}) \hat{\vInv}^{m} .
\end{aligned}
$$

Although a linear interpolation that matches the level of $\vInv$ at the
gridpoints is simple, a Hermite interpolation that matches both the level and
the derivative of the $\bar{\vInv}_{\prd}$ function at the gridpoints has the
considerable virtue that the $\bar{\vFunc}_{\prd}$ derived from it numerically
satisfies the envelope theorem at each of the gridpoints for which the problem
has been solved.

## Extensions

### A Tighter Upper Bound

{cite:t}`BufferStockTheory` derives an upper limit $\MPCmax_{\prd}$ for the MPC
as $m_{\prd}$ approaches its lower bound. Using this fact plus the strict
concavity of the consumption function yields the proposition that

$$
\cFunc_{\prd}(\underline{m}_{\prd}+\blacktriangle \mNrm_{\prd}) < \MPCmax_{\prd} \blacktriangle \mNrm_{\prd}.
$$

The solution method described above does not guarantee that approximated
consumption will respect this constraint between gridpoints, and a failure to
respect the constraint can occasionally cause computational problems in solving
or simulating the model. Here, we describe a method for constructing an
approximation that always satisfies the constraint.

Defining $\mtCusp$ as the 'cusp' point where the two upper bounds intersect:

$$
\begin{aligned}
\left(\blacktriangle \mtCusp+\blacktriangle \hNrm_{\EndStg}\right)\MPCmin_{\prd} &=  \MPCmax_{\prd} \blacktriangle \mtCusp \\
\blacktriangle \mtCusp &=  \frac{\MPCmin_{\prd}\blacktriangle \hNrm_{\EndStg}}{(1-\MPCmin_{\prd})\MPCmax_{\prd}} \\
\mtCusp &=  \frac{\MPCmin_{\prd}\hNrm_{\EndStg}-\hEndMin_{\EndStg}}{(1-\MPCmin_{\prd})\MPCmax_{\prd}},
\end{aligned}
$$

we want to construct a consumption function for
$m_{\prd} \in (\underline{m}_{\prd}, \mtCusp]$ that respects the tighter upper
bound:

$$
\begin{array}{rcl}
\blacktriangle \mNrm_{\prd} \MPCmin_{\prd} < & \cFunc_{\prd}(\underline{m}_{\prd}+\blacktriangle \mNrm_{\prd}) & < \MPCmax_{\prd} \blacktriangle \mNrm_{\prd} \\
\blacktriangle \mNrm_{\prd}(\MPCmax_{\prd}- \MPCmin_{\prd}) > & \MPCmax_{\prd} \blacktriangle \mNrm_{\prd}-\cFunc_{\prd}(\underline{m}_{\prd}+\blacktriangle \mNrm_{\prd}) & > 0 \\
1 > & \left(\frac{\MPCmax_{\prd} \blacktriangle \mNrm_{\prd}-\cFunc_{\prd}(\underline{m}_{\prd}+\blacktriangle \mNrm_{\prd})}{\blacktriangle \mNrm_{\prd}(\MPCmax_{\prd}- \MPCmin_{\prd})}\right) & > 0.
\end{array}
$$

Again defining $\mu_{\prd} =\log \blacktriangle \mNrm_{\prd}$, the object in the
middle of the inequality is

```{math}
:label: eq:koppaLo
\underline{\koppa}_{\prd}(\mu_{\prd}) \equiv  \frac{\MPCmax_{\prd}-\cFunc_{\prd}(\underline{m}_{\prd}+e^{\mu_{\prd}})e^{-\mu_{\prd}}}{\MPCmax_{\prd}-\MPCmin_{\prd}} .
```

As $m_{\prd}$ approaches $-\underline{m}_{\prd}$,
$\underline{\koppa}_{\prd}(\mu_{\prd})$ converges to zero, while as $m_{\prd}$
approaches $+\infty$, $\underline{\koppa}_{\prd}(\mu_{\prd})$ approaches $1$.

As before, we can derive an approximated consumption function; call it
$\check{\underline{\cFunc}}_{\prd}$. This function will clearly do a better job
approximating the consumption function for low values of $\mNrm_{\prd}$ while
the previous approximation will perform better for high values of
$\mNrm_{\prd}$.

For middling values of $\mNrm$ it is not clear which of these functions will
perform better. However, an alternative is available which performs well. Define
the highest gridpoint below $\mtCusp$ as $\bar{\check{\mNrm}}_{\prd}^{\#}$ and
the lowest gridpoint above $\mtCusp$ as $\underline{\hat{\mNrm}}_{\prd}^{\#}$.
Then there will be a unique interpolating polynomial that matches the level and
slope of the consumption function at these two points. Call this function
$\tilde{\cFunc}_{\prd}(\mNrm)$.

Using indicator functions that are zero everywhere except for specified
intervals,

$$
\begin{aligned}
\mathbf{1}_{\text{Lo}}(\mNrm) &= 1 \text{~if $          \mNrm \leq  \bar{\check{\mNrm}}_{\prd}^{\#}$} \\
\mathbf{1}_{\text{Mid}}(\mNrm) &= 1 \text{~if $\bar{\check{\mNrm}}_{\prd}^{\#}          < \mNrm <   \underline{\hat{\mNrm}}_{\prd}^{\#}$} \\
\mathbf{1}_{\text{Hi}}(\mNrm) &= 1 \text{~if $\underline{\hat{\mNrm}}_{\prd}^{\#}           \leq \mNrm$}
\end{aligned}
$$

we can define a well-behaved approximating consumption function

$$
\check{\cFunc}_{\prd} = \mathbf{1}_{\text{Lo}} \check{\underline{\cFunc}}_{\prd} + \mathbf{1}_{\text{Mid}} \check{\tilde{\cFunc}}_{\prd}+\mathbf{1}_{\text{Hi}} \check{\overline{\cFunc}}_{\prd}.
$$

This just says that, for each interval, we use the approximation that is most
appropriate. The function is continuous and once-differentiable everywhere, and
is therefore well behaved for computational purposes.

We now construct an upper-bound value function implied for a consumer whose
spending behavior is consistent with the refined upper-bound consumption rule.

For $\mNrm_{\prd} \geq \mNrm_{\prd}^{\#}$, this consumption rule is the same as
before, so the constructed upper-bound value function is also the same. However,
for values $\mNrm_{\prd} < \mNrm_{\prd}^{\#}$ matters are slightly more
complicated.

Start with the fact that at the cusp point,

$$
\begin{aligned}
\bar{\vFunc}_{\prd}(\mtCusp) &= \uFunc(\bar{\cNrm}_{\prd}(\mtCusp))\mathcal{C}_{\prd}^T \\
&=  \uFunc(\blacktriangle \mtCusp  \MPCmax_{\prd})\mathcal{C}_{\prd}^{T} .
\end{aligned}
$$

But for _all_ $\mNrm_{\prd}$,

$$
\bar{\vFunc}_{\prd}(\mNrm) = \uFunc(\bar{\cNrm}_{\prd}(\mNrm))+ \bar{\mathcal{T}}(\mNrm-\bar{\cNrm}_{\prd}(\mNrm)),
$$

and we assume that for the consumer below the cusp point consumption is given by
$\MPCmax_{\prd} \blacktriangle \mNrm_{\prd}$ so for $\mNrm_{\prd}< \mtCusp$

$$
\bar{\vFunc}_{\prd}(\mNrm) = \uFunc( \MPCmax_{\prd} \blacktriangle \mNrm)+ \bar{\mathcal{T}}((1-\MPCmax_{\prd})\blacktriangle \mNrm),
$$

which is easy to compute because
$\bar{\mathcal{T}}(\aNrm_{\prd}) = \DiscFac \bar{\vFunc}_{\prd+1}(\aNrm_{\prd}\RNrmByG+1)$
where $\bar{\vFunc}_{\prd}$ is as defined above because a consumer who ends the
current period with assets exceeding the lower bound will not expect to be
constrained next period. (Recall again that we are merely constructing an object
that is guaranteed to be an _upper bound_ for the value that the 'realist'
consumer will experience.) At the gridpoints defined by the solution of the
consumption problem can then construct

$$
\bar{\vInv}_{\prd}(\mNrm) = ((1-\CRRA)\bar{\vFunc}_{\prd}(\mNrm))^{1/(1-\CRRA)}
$$

which yields the appropriate vector for constructing $\check{\Chi}$ and
$\check{\Koppa}$. The rest of the procedure is analogous to that performed for
the consumption rule and is thus omitted for brevity.

```{figure} ./images/tighter-upper-bound.png
:label: fig:tighterupper
:alt: A diagram showing the true consumption function bounded above by both the optimist's consumption rule and a tighter linear bound originating from the natural borrowing constraint.
:align: center
:width: 80%

A Tighter Upper Bound
```

### Stochastic Rate of Return

Thus far we have assumed that the interest factor is constant at $\Rfree$.
Extending the previous derivations to allow for a perfectly forecastable
time-varying interest factor $\Rfree_{\prd}$ would be trivial. Allowing for a
stochastic interest factor is less trivial.

The easiest case is where the interest factor is i.i.d.,

```{math}
:label: eq:distRisky
\log \Risky_{t+n} \sim \mathcal{N}(r + \pi - \sigma^{2}_{\risky}/2,\sigma^{2}_{\risky}) ~\forall~n>0
```

because in this case {cite:t}`merton:restat` and {cite:t}`samuelson:portfolio`
showed that for a consumer without labor income (or with perfectly forecastable
labor income) the consumption function is linear, with an MPC[^crra-rate-risk].

[^crra-rate-risk]: See {cite:t}`CRRA-RateRisk` for a derivation.

```{math}
:label: eq:MPCExact
\text{MPC} = 1- \left(\DiscFac  \Ex_{\prd}[\Risky_{\prd+1}^{1-\CRRA}]\right)^{1/\CRRA}
```

and in this case the previous analysis applies once we substitute this MPC for
the one that characterizes the perfect foresight problem without rate-of-return
risk.

The more realistic case where the interest factor has some serial correlation is
more complex. We consider the simplest case that captures the main features of
empirical interest rate dynamics: An AR(1) process. Thus the specification is

$$
r_{\prd+1}-\bar{r} = (r_{\prd}-\bar{r}) \gamma + \epsilon_{\prd+1}
$$

where $\bar{r}$ is the long-run mean log interest factor, $0 < \gamma < 1$ is
the AR(1) serial correlation coefficient, and $\epsilon_{\prd+1}$ is the
stochastic shock.

The consumer's problem in this case now has two state variables, $\mNrm_{\prd}$
and $r_{\prd}$, and is described by

```{math}
:label: eq:vtNormRisky
\begin{aligned}
\vFunc_{\prd}(\mNrm_{\prd},r_{\prd}) &= \max_{\cNrm_{\prd}} ~ \uFunc(\cNrm_{\prd})+ \Ex_{\prd}[\DiscFac_{\prd+1}\PermGroFac_{\prd+1}^{1-\CRRA}\vFunc_{\prd+1}(\mNrm_{\prd+1},r_{\prd+1})] \\
&\text{s.t.} \\
\aNrm_{\prd} &= \mNrm_{\prd}-\cNrm_{\prd} \\
r_{\prd+1}-\bar{r} &= (r_{\prd}-\bar{r})\gamma + \epsilon_{\prd+1} \\
\Risky_{\prd+1} &= \exp(r_{\prd+1}) \\
\mNrm_{\prd+1} &= \underbrace{\left(\Risky_{\prd+1}/\PermGroFac_{\prd+1}\right)}_{\equiv R^p_{\prd+1}}\aNrm_{\prd}+\tranShkEmp_{\prd+1}.
\end{aligned}
```

We approximate the AR(1) process by a Markov transition matrix using standard
techniques. The stochastic interest factor is allowed to take on 11 values
centered around the steady-state value $r$ and chosen using standard procedures.
Given this Markov transition matrix, _conditional_ on the Markov AR(1) state the
consumption functions for the 'optimist' and the 'pessimist' will still be
linear, with identical MPC's that are computed numerically. Given these MPC's,
the (conditional) realist's consumption function can be computed for each Markov
state, and the converged consumption rules constitute the solution contingent on
the dynamics of the stochastic interest rate process.

In principle, this refinement should be combined with the previous one; further
exposition of this combination is omitted here because no new insights spring
from the combination of the two techniques.

## Conclusion

The method proposed here is not universally applicable. For example, the method
cannot be used for problems for which upper and lower bounds to the 'true'
solution are not known. But many problems do have obvious upper and lower
bounds, and in those cases (as in the consumption example used in the paper),
the method may result in substantial improvements in accuracy and stability of
solutions.
