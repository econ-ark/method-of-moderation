---
# Page-specific frontmatter (inherits title, keywords, bibliography from myst.yml)
abstract: abstract.md
parts:
  jel_codes: D14; C61; G11
exports:
  - format: tex+pdf
    template: arxiv_two_column
    output: ../exports/moderation_letters.pdf
---

# Introduction

Solving a consumption-saving problem using numerical methods requires the modeler to choose how to represent a policy function. In the stochastic case, where analytical solutions are generally not available, a common approach is to use low-order polynomial splines that exactly match the function at a finite set of gridpoints, and then to assume that interpolated or extrapolated versions of that spline represent the function well at the continuous infinity of unmatched points. {cite:t}`carrollEGM` developed the endogenous gridpoints method (EGM), which has become a standard tool for computing consumption at gridpoints determined endogenously using the Euler equation.

Unfortunately, this endogenous gridpoints solution is not very well-behaved outside the original range of gridpoints (though other common solution methods are no better outside their own predefined ranges). {ref}`fig:ExtrapProblem` demonstrates the point.  The figure shows the approximated precautionary component of savings, the amount that a consumer facing income uncertainty saves above and beyond that of a consumer with a deterministic income stream with identical average value.  Although theory proves that precautionary saving is always positive, the linearly extrapolated numerical approximation eventually predicts negative precautionary saving.  However, in the presence of uncertainty, the consumption-saving rule must be evaluated outside _any_ prespecified grid.  This is because large positive shock realizations push next period's assets for a sufficiently wealthy individual beyond the grid boundaries.

:::{figure} #fig:egm-extrapolation-problem
:label: fig:ExtrapProblem
:alt: Graph showing that precautionary saving, approximated with linear extrapolation, incorrectly becomes negative for large market resources.

For Large Enough Resources $\mNrm$, Predicted Precautionary Saving is Negative (Oops!)
:::

This error cannot be fixed by extending the upper gridpoint. While extrapolation techniques can prevent this from being fatal, the problem is often dealt with using inelegant methods whose implications for accuracy are difficult to gauge.

This paper argues that, in the standard consumption problem, a better approach is to rely upon the fact that without uncertainty, the optimal consumption function has a simple analytical solution. The key insight is that, under standard assumptions, the consumer who faces an uninsurable labor income risk will consume less than a consumer with the same path for expected income but who does not perceive any uncertainty. Following {cite:t}`Leland1968`, {cite:t}`Sandmo1970`, and {cite:t}`Kimball1990`, the 'realistic' consumer, who _does_ perceive the risks, will engage in 'precautionary saving,' so the perfect foresight riskless solution provides an upper bound to the solution that will actually be optimal. A lower bound is provided by the behavior of a consumer who has the subjective belief that the future level of income will be the worst that it can possibly be. This consumer, too, behaves according to the convenient analytical perfect foresight solution, but his certainty is that of a pessimist perfectly confident in his pessimism.

We build on bounds for the consumption function and limiting MPCs established by {cite:t}`StachurskiToda2019JET`, {cite:t}`MST2020JET`, {cite:t}`Carroll2001MPCBound`, and {cite:t}`MaToda2021SavingRateRich` in buffer-stock theory. Using results from {cite:t}`CarrollShanker2024`, we show how to use these bounds to constrain the shape and characteristics of the solution to the 'realist' problem characterized by {cite:t}`Carroll1997`. Imposition of these constraints clarifies and speeds the solution of the realist's problem. For comparison, we use the endogenous gridpoints method {cite:p}`carrollEGM` as our benchmark, which computes consumption at gridpoints determined endogenously using the Euler equation.

After showing how to use the method in the baseline case, we show how to refine it to encompass an even tighter theoretical bound, and how to extend it to solve a problem in which the consumer faces both labor income risk and rate-of-return risk.

# The Realist's Problem

Consider a consumer who correctly perceives all risks. The consumer has CRRA utility with risk aversion parameter $\CRRA > 0$:

```{math}
:label: eq:UtilityFunc
\uFunc(\cNrm) = \begin{cases}
\frac{\cNrm^{1-\CRRA}}{1-\CRRA} & \text{if } \CRRA \neq 1 \\
\log \cNrm & \text{if } \CRRA = 1.
\end{cases}
```

This utility function satisfies prudence ($\uFunc''' > 0$), which induces precautionary saving. The consumer maximizes expected lifetime utility:

```{math}
:label: eq:MaxProb
\max_{\cNrm_t,\aNrm_t}~\Ex_{t}\left[\sum_{n=0}^{T-t}\DiscFac^{n} \uFunc(\cNrm_{t+n})\right]
```

subject to $\aNrm_{t} + \cNrm_{t} = \mNrm_{t}$, where $\mNrm$ denotes income or resources and $\aNrm$ denotes assets. We will make use of how the properties of uncertain sources of income impact a perfect foresight household's propensity to consume.  Common sources include interest, wages, dividends, and government transfers.  We focus on resources of the form $\mNrm_{t+1} = \aNrm_{t}\Rfree_{t+1} + \yNrm_{t+1}$, where $\Rfree_{t+1}$ denotes the interest rate, and $\yNrm_{t+1}$ labor income.  Initially we take $\Rfree_{t+1}$ to be deterministic, and relax this later. 

While our method can be adapted to a range of stochastic labor income processes, to fix ideas we suppose income evolves via the Friedman-Muth process ({cite:t}`Friedman1957` distinguished permanent from transitory income; {cite:t}`Muth1960` provided the stochastic framework).  That is, $\yNrm_{t+1} = \pNrm_{t+1}\tranShk_{t+1}$ where $\pNrm$ denotes permanent labor income and $\tranShk_{t+1}$ a transitory component.  Permanent income growth is given by $\pNrm_{t+1} = \pNrm_{t} \PermGroShk_{t+1}$, $\PermGroShk_{t+1} = \PermGroFac_{t+1} \permShk_{t+1}$. Here $\PermGroFac_{t+1}$ is deterministic income growth, while $\permShk_{t_1}$ are permanent shocks with mean unity and bounded support $[\permShkMin, \permShkMax]$ where $0 < \permShkMin \leq 1 \leq \permShkMax < \infty$.  Transitory shocks $\tranShk_{t+1}$ take value $0$ with probability $\WorstProb > 0$ (unemployment) or $\tranShkEmp_{t+1}/(1-\WorstProb)$ otherwise, with $\Ex_t[\tranShkEmp_{t+1}]=1$.

This problem can be rewritten (see {cite:t}`SolvingMicroDSOPs` for a proof) in a more convenient form in which choice and state variables are normalized by the level of permanent income, e.g., replacing $\mNrm_t$ with $\mNrm_t/\pNrm_t$. When that is done, the Bellman equation for the transformed version of the consumer's problem is

```{math}
:label: eq:vNormed
\vFunc_{t}(\mNrm_{t}) = \max_{\cNrm_{t},\aNrm_t} ~~ \left(\uFunc(\cNrm_{t})+\DiscFac \Ex_{t}[ \PermGroShk_{t+1}^{1-\CRRA}\vFunc_{t+1}(\mNrm_{t+1})]\right)
```

with Euler equation $\uPrime(\cNrm_{t}) = \DiscFac \Rfree \Ex_{t}[ \PermGroShk_{t+1}^{-\CRRA} \uPrime(\cNrm_{t+1})]$.

{cite:t}`CarrollShanker2024` gives conditions for a finite solution of the problem with a Friedman-Muth process.  Consider the case of time-invariant $\PermGroFac_t$, $\psi_t$, and $\Rfree_t$, and define the absolute patience factor $\AbsPatFac\equiv(\DiscFac\Rfree)^{1/\CRRA}$.  Then a finite solution requires: (i) finite-value-of-autarky condition (FVAC) $0<\DiscFac\PermGroFac^{1-\CRRA}\Ex[\permShk^{1-\CRRA}]<1$, (ii) absolute-impatience condition (AIC) $\AbsPatFac<1$, (iii) return-impatience condition (RIC) $\AbsPatFac/\Rfree<1$, (iv) growth-impatience condition (GIC) $\AbsPatFac/\PermGroFac<1$, and (v) finite-human-wealth condition (FHWC) $\PermGroFac/\Rfree<1$. These patience conditions ensure the consumption bounds and limiting MPCs used in our method.

For expositional simplicity, in the numerical results that follow, we assume $\CRRA \neq 1$ and set $\PermGroFac=1$, $\permShk=1$ (no permanent income growth or shocks) and focus on the next-to-last period of a finite horizon problem.  The method extends to general cases, including the infinite-horizoon formulation.

# The Method of Moderation

## The Optimist, the Pessimist, and the Realist

As a preliminary to our solution, define $\hNrmOpt$[^optimist-human-wealth] as end-of-period human wealth (the present discounted value of future labor income) for a perfect foresight version of the problem of a 'risk optimist:' a consumer who believes with perfect confidence that the shocks will always take their expected value of 1, $\tranShk_{t+n} = \Ex[\tranShk]=1~\forall~n>0$. The solution to a perfect foresight problem of this kind takes the form

```{math}
:label: eq:cFuncOpt
\cFuncOpt(\mNrm) = (\mNrm + \hNrmOpt)\MPCmin
```

for a constant minimal marginal propensity to consume $\MPCmin$.[^mpc-min-definition] We similarly define $\hNrmPes$[^pessimist-human-wealth] as 'minimal human wealth,' the present discounted value of labor income if the shocks were to take on their worst value in every future period $\tranShk_{t+n} = \tranShkMin$ $\forall~n>0$.  We refer to a consumer whose expects to encounter this sequence of shocks as a 'pessimist'. Their consumption decision rule is given by

```{math}
:label: eq:cFuncPes
\cFuncPes(\mNrm) = (\mNrm+\hNrmPes)\MPCmin.
```

We will call a 'realist' the consumer who correctly perceives the true probabilities of the future risks and optimizes accordingly.

[^optimist-human-wealth]:
    Setting $\tranShk_{t+n}=1$ (the optimist's assumption), human wealth in infinite-horizon is $\hNrmOpt = \PermGroFac/(\Rfree-\PermGroFac)$ if $\Rfree>\PermGroFac$. When $\PermGroFac=1$, $\hNrmOpt = 1/(\Rfree-1)$.

[^mpc-min-definition]:
    The MPC of the perfect foresight consumer: infinite-horizon $\MPCmin=1-\AbsPatFac/\Rfree$.

[^pessimist-human-wealth]:
    Setting $\tranShk_{t+n}=\tranShkMin~\forall~n>0$, minimal human wealth is $\hNrmPes=\tranShkMin\PermGroFac/(\Rfree-\PermGroFac)$ if $\Rfree>\PermGroFac$. When $\tranShkMin=0$, $\hNrmPes=0$.

A first useful point is that, for the realist, a lower bound for the level of market resources is the natural borrowing constraint $\mNrmMin = -\hNrmPes$ derived by {cite:t}`Aiyagari1994` and {cite:t}`Huggett1993`, because if $\mNrm$ equalled this value then there would be a positive finite chance (however small) of receiving $\tranShk_{t+n} = \tranShkMin$ in every future period, which would require the consumer to set $\cNrm$ to zero in order to guarantee that the intertemporal budget constraint holds. Since consumption of zero yields infinite marginal utility, {cite:t}`Zeldes1989` and {cite:t}`Deaton1991` show that the solution to the realist consumer's problem is not well defined for values of $\mNrm \leq \mNrmMin$, and the limiting value of the realist's $\cNrm$ is zero as $\mNrm \downarrow \mNrmMin$ (established in {cite:t}`CarrollShanker2024`).

It is convenient to define 'excess' market resources as the amount by which actual resources exceed the lower bound, and 'excess' human wealth as the amount by which mean expected human wealth exceeds guaranteed minimum human wealth:

```{math}
:label: eq:ExcessDef
\begin{aligned}
\mNrmEx &= \mNrm+\overbrace{\hNrmPes}^{=-\mNrmMin} \\
\hNrmEx &= \hNrmOpt-\hNrmPes.
\end{aligned}
```

We now rewrite the optimal consumption rules for the two perfect foresight problems in terms of excess resources and human wealth. The 'pessimist' perceives human wealth to be equal to its minimum feasible value $\hNrmPes$ with certainty, and so consumes a constant fraction of current excess resources

```{math}
:label: eq:cFuncPesExcess
\cFuncPes(\mNrm) = \mNrmEx\MPCmin.
```

The 'optimist,' on the other hand, pretends that there is no uncertainty about future income, and therefore consumes the same fraction out of current excess resources _and_ excess human wealth

```{math}
:label: eq:cFuncOptExcess
\cFuncOpt(\mNrm) = (\mNrmEx + \hNrmEx)\MPCmin = \cFuncPes(\mNrm)+\hNrmEx \MPCmin.
```

## The Moderation Ratio

Whereas the pessimist believes that they need to save enough to smooth consumption through guaranteed adverse future income outcomes, a realist needs only save enough to insure against this eventuality (while reoptimizing each period as information becomes available).  However, the adverse outcome remains a possibility even for a high-resource individual, necessitating some degree of precautionary saving.  It therefore seems clear that the spending of the realist will be strictly greater than that of the pessimist and strictly less than that of the optimist, as shown in {ref}`fig:IntExpFOCInvPesReaOptNeedHi`. 

:::{figure} #fig:truth-bounded-by-theory
:label: fig:IntExpFOCInvPesReaOptNeedHi
:alt: Graph showing the realist's consumption function is bounded by the pessimist's (lower) and optimist's (upper) consumption functions.

Moderation Illustrated: $\cFuncPes < \cFuncApprox < \cFuncOpt$
:::

The proof is more difficult than might be imagined, but the necessary work is done in {cite:t}`CarrollShanker2024` so we will take the proposition as a fact:

$$
\cFuncPes(\mNrmMin+\mNrmEx) < \cFuncReal(\mNrmMin+\mNrmEx) < \cFuncOpt(\mNrmMin+\mNrmEx)
$$

Subtracting $\cFuncPes(\mNrmMin+\mNrmEx)$ in each of these inequalities and using equations {eq}`eq:cFuncPesExcess` and {eq}`eq:cFuncOptExcess` gives

$$
\begin{array}{rcl}
0 < & \cFuncReal(\mNrmMin+\mNrmEx)-\cFuncPes(\mNrmMin+\mNrmEx) & < \hNrmEx \MPCmin \\
0 < & \underbrace{\left(\frac{\cFuncReal(\mNrmMin+\mNrmEx)-\cFuncPes(\mNrmMin+\mNrmEx)}{\hNrmEx \MPCmin}\right)}_{\equiv \modRte} & < 1
\end{array}
$$

where the fraction in the middle of the last inequality is the moderation ratio measuring how close the realist's consumption is to the optimist's behavior (the numerator is the gap between the realist and pessimist) relative to the maximum possible gap between optimist and pessimist. When $\modRte=0$, the realist behaves like the pessimist (maximum precautionary saving); when $\modRte=1$, the realist behaves like the optimist (no precautionary saving). {cite:t}`CarrollKimball1996` and {cite:t}`CarrollShanker2024` establish that under bounded shocks, $\modRte\in(0,1)$ strictly for all $\mNrm > \mNrmMin$. Defining $\logmNrmEx = \log \mNrmEx$ (which can range from $-\infty$ to $\infty$), the object in the middle of the last inequality is

```{math}
:label: eq:modRte
\modRte(\logmNrmEx) \equiv  \left(\frac{\cFuncReal(\mNrmMin+e^{\logmNrmEx})-\cFuncPes(\mNrmMin+e^{\logmNrmEx})}{\hNrmEx \MPCmin}\right),
```

and we now define

```{math}
:label: eq:chi
\begin{aligned}
\logitModRte(\logmNrmEx) &= \log \left(\frac{\modRte(\logmNrmEx)}{1-\modRte(\logmNrmEx)}\right) \\
&= \log(\modRte(\logmNrmEx)) - \log(1-\modRte(\logmNrmEx))
\end{aligned}
```

which has the virtue that it is _asymptotically linear_ in the limit as $\logmNrmEx$ approaches $+\infty$.[^asymptotically-linear] The method uses standard transformations for unbounded domains: logit maps $\modRte \in (0,1)$ to $\logitModRte \in (-\infty, \infty)$ with inverse sigmoid $\modRte = 1/(1+\exp(-\logitModRte))$; log maps $(\mNrm - \mNrmMin) \in (0, \infty)$ to $\logmNrmEx \in (-\infty, \infty)$. As $\modRte \to 1$ (realist approaches optimist), $\logitModRte \to +\infty$; as $\modRte \to 0$ (realist approaches pessimist), $\logitModRte \to -\infty$.

[^asymptotically-linear]:
    Under the GIC, $\logitModRte(\logmNrmEx)$ is asymptotically linear with slope $\asympSlope \geq 0$ as $\logmNrmEx \to +\infty$. We extrapolate $\logitModRte$ linearly using the boundary slope, preserving $\modRte\in(0,1)$ and hence $\cFuncPes < \cFuncApprox < \cFuncOpt$ throughout the extrapolation domain.

Given $\logitModRte$, the consumption function can be recovered from

```{math}
:label: eq:cFuncHi
\cFuncReal = \cFuncPes+\overbrace{\frac{1}{1+\exp(-\logitModRte)}}^{=\modRte} \hNrmEx \MPCmin.
```

Thus, the method of moderation (MoM) is to calculate $\logitModRte$ at the points $\logmNrmEx$ corresponding to the log of the $\mNrmEx$ points defined above, and then using these to construct an interpolating approximation $\logitModRteApprox$ from which we indirectly obtain our approximated consumption rule $\cFuncApprox$ (an approximation to the true $\cFuncReal$) by substituting $\logitModRteApprox$ for $\logitModRte$ in equation {eq}`eq:cFuncHi`.

Because this method relies upon the fact that the problem is easy to solve if the decision maker has unreasonable views (either in the optimistic or the pessimistic direction), and because the correct solution is always between these immoderate extremes, we call our solution procedure the 'method of moderation.'

Results are shown in {ref}`fig:ExtrapProblemSolved`; a reader with very good eyesight might be able to detect the barest hint of a discrepancy between the Truth and the Approximation at the far righthand edge of the figure, a stark contrast with the calamitous divergence evident in {ref}`fig:ExtrapProblem`.

:::{figure} #fig:mom-solution
:label: fig:ExtrapProblemSolved
:alt: Graph showing that the Method of Moderation produces an accurate extrapolated consumption function that does not predict negative precautionary saving.

Extrapolated $\cFuncApprox$ Constructed Using the Method of Moderation
:::

# Extensions

## A Tighter Upper Bound

{cite:t}`CarrollShanker2024` derives an explicit formula for the MPC at the natural borrowing constraint: $\MPCmax = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)$ where $\WorstProb$ is the unemployment probability derived by {cite:t}`CarrollToche2009`. This provides a tighter upper bound near the constraint, extending the explicit limiting MPC formulas established in buffer-stock theory by {cite:t}`MaToda2021SavingRateRich`. Strict concavity of the consumption function implies $\cFuncReal(\mNrm) < \MPCmax \mNrmEx$ for low wealth, while the optimist's bound $\cFuncReal(\mNrm) < \cFuncOpt(\mNrm) = (\mNrmEx+\hNrmEx)\MPCmin$ is tighter for high wealth.

:::{figure} #fig:mom-consumption-function
:label: fig:IntExpFOCInvPesReaOptNeed45
:alt: A diagram showing the true consumption function bounded above by both the optimist's consumption rule and a tighter linear bound originating from the natural borrowing constraint.

A Tighter Upper Bound
:::

As shown in {ref}`fig:IntExpFOCInvPesReaOptNeed45`, the two upper bounds intersect at the cusp point:

```{math}
:label: eq:mNrmCusp
\mNrmCusp = -\hNrmPes + \frac{\MPCmin(\hNrmOpt-\hNrmPes)}{\MPCmax-\MPCmin}
```

This intersection occurs in the feasible region since $\MPCmax > \MPCmin$ under the stated conditions (the MPC is highest when wealth is lowest).

For $\mNrm < \mNrmCusp$, define the low-resource moderation ratio using the tighter bound:

```{math}
:label: eq:modRteLoTightUpBd
\modRteLoTightUpBd(\logmNrmEx) = \frac{\cFuncReal(\mNrmMin+e^{\logmNrmEx})e^{-\logmNrmEx}-\MPCmin}{\MPCmax-\MPCmin}
```

This ratio measures how far consumption per unit of wealth exceeds the optimist's MPC, relative to the maximum possible excess. Applying the logit transformation and interpolating as before yields consumption satisfying both upper bounds throughout.

For computational robustness, construct a three-piece approximation: below the cusp using the tight bound, near the cusp using Hermite interpolation matching levels and slopes at adjacent gridpoints, above the cusp using the original optimist bound. This ensures continuous, differentiable consumption functions respecting all theoretical constraints.

The MoM also contributes to literature which aims to improve the precisison of dynamic stochastic optimization solutions, such as {cite:t}`Chipeniuk2020`. Table 1 demonstrates the accuracy gains obtained with the method between each pair of grid points $m_j,m_{j+1}$, as well as for the extrapolation of the consumption function to $\overline{m}=30$.  Displayed is the average absolute difference between the true consumption function and each approximation.  In each region the MoM produces an approximation which is more than an order of magnitude more accurate than the basic EGM.

| Method   | $[m_0,m_1]$ | $[m_1,m_2]$  | $[m_2,m_3]$ | $[m_3,m_4]$ | $[m_4,\overline{m}]$|
| :-------- | :-------: | :--------: | :-------: | :--------: | :-------: |
| EGM  | 8.5(-2)   | 1.8(-4) | 2.5(-5) | 7.3(-6) | 1.1(-1) |
| MoM | 2.9(-3)    | 4.3(-9) | 6.6(-7) | 1.3(-7) | 2.4(-3) |

Table 1: Approximation errors. Orders of magnitude in parentheses.

## Value Function

Often it is useful to know the value function as well as the consumption rule. Fortunately, many of the tricks used when solving for the consumption rule have a direct analogue in approximation of the value function. Define the inverse value function transformation $\vInvOpt = ((1-\CRRA)\vFuncOpt)^{1/(1-\CRRA)}$, which under perfect foresight equals $(\mNrmEx+\hNrmEx)\MPCmin^{-\CRRA/(1-\CRRA)}$ (linear in market resources). The value moderation ratio $\valModRteReal$ measures proximity to the optimist's value, with logit transformation $\logitValModRteReal$ applied as before. Interpolate $\logitValModRteReal$ at gridpoints and invert to obtain $\vFuncReal = \uFunc(\vInvReal)$.

## Hermite Interpolation

The numerical accuracy of the method of moderation depends critically on the quality of function approximation between gridpoints {cite:p}`Santos2000`. Our bracketing approach complements work that bounds numerical errors in dynamic economic models {cite:p}`JuddMaliarMaliar2017`. Although linear interpolation that matches the level of $\cFuncReal$ at the gridpoints is simple, Hermite interpolation {cite:p}`Fritsch1980` offers a considerable advantage.

The moderation ratio derivative measures how quickly the realist approaches the optimist as resources increase.  Differentiating {eq} `eq:modRte` with respect to $\logmNrmEx$ we obtain

```{math}
:label: eq:modRteMu
\frac{\partial \modRte}{\partial \logmNrmEx} = \frac{\mNrmEx (\partial \cFuncReal/\partial \mNrm - \MPCmin)}{\MPCmin \hNrmEx}.
```

Rearranging this yields a moderation form for the marginal propensity to consume:

```{math}
:label: eq:MPCModeration
\frac{\partial \cFuncReal}{\partial \mNrm} = (1-\MPCmod)\,\MPCmin + \MPCmod\,\MPCmax
```

where

```{math}
:label: eq:MPCModerationWeight
\MPCmod = \frac{\MPCmin}{\MPCmax-\MPCmin} \cdot \frac{\hNrmEx}{\mNrmEx} \cdot \partial \modRte / \partial \logmNrmEx.
```

Theory guarantees $\MPCmin \leq \partial \cFuncReal/\partial \mNrm \leq \MPCmax$ at gridpoints where the Euler equation holds. At very high wealth, $\MPCmod \to 0$ and the MPC approaches $\MPCmin$; near the borrowing constraint, $\MPCmod \to 1$ and the MPC approaches $\MPCmax$.

For Hermite interpolation, compute $\modRteMu$ at gridpoints, then derive $\logitModRteMu = \modRteMu/[\modRte(1-\modRte)]$ for slope data. By matching both the level and the derivative of the $\cFuncReal$ function at the gridpoints, {cite:t}`BenvenisteScheinkman1979` and {cite:t}`MilgromSegal2002` show that the consumption rule derived from such interpolation numerically satisfies the Euler equation at each gridpoint for which the problem has been solved. These techniques extend naturally to the value function approximation.

 For monotone cubic Hermite schemes {cite:p}`Fritsch1980,FritschButland1984,deBoor2001`, theoretical slopes may be adjusted to enforce monotonicity {cite:p}`Hyman1983`. The Fritsch-Carlson algorithm modifies slopes at local extrema, while Fritsch-Butland uses harmonic mean weighting. Both preserve the shape-preserving property essential for consumption functions that must be strictly increasing.

## Stochastic Rate of Return

For i.i.d. returns with $\log \Risky \sim \Nrml(r + \equityPrem - \std^{2}_{\risky}/2,\std^{2}_{\risky})$, {cite:t}`Samuelson1969,Merton1969,Merton1971` showed that for a consumer without labor income (or with perfectly forecastable labor income) the consumption function is linear, with an MPC $= 1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$. See {cite:t}`CRRA-RateRisk,BBZ2016SkewedWealth,CKW2021Aggregation` for extensions. Simply substitute this stochastic MPC for $\MPCmin$ throughout our formulas. The pessimist and optimist still perceive their income streams with certainty, but both face the same stochastic return; thus the Merton-Samuelson result applies to them and their consumption functions remain linear. The realist, however, faces both labor income uncertainty and rate-of-return risk, so the moderation ratio captures the combined precautionary response to both sources of uncertainty. All moderation ratio calculations proceed identically. Extensions to serially correlated returns require tracking the return state as an additional state variable, complicating the analysis but not fundamentally altering the approach. As {ref}`fig:StochasticBounds` shows, consumption remains bounded between the pessimist and the optimist, each of which consume slightly less in the face of return uncertainty.  

:::{figure} #fig:stochastic-bounds
:label: fig:StochasticBounds
:alt: Comparison of consumption bounds under deterministic and stochastic rates of return.

Effect of Return Uncertainty on Consumption Bounds
:::

# Conclusion

The method proposed here is not universally applicable. For example, the method cannot be used for problems for which upper and lower bounds to the 'true' solution are not known. But many problems do have obvious upper and lower bounds, and in those cases (as in the consumption example used in the paper), the method may result in substantial improvements in accuracy and stability of solutions.  The method of moderation is efficient because the transformed moderation ratio is better-behaved than consumption, requiring fewer gridpoints.
