---
title: The Method of Moderation
bibliography:
  - references.bib
abstract: |
  In a risky world, a pessimist assumes the worst will happen. Someone who ignores risk altogether is an optimist. Consumption decisions are mathematically simple for both the pessimist and the optimist because both behave as if they live in a riskless world. A consumer who is a realist (that is, who wants to respond optimally to risk) faces a much more difficult problem, but (under standard conditions) will choose a level of spending somewhere between that of the pessimist and the optimist. We use this fact to redefine the space in which the realist searches for optimal consumption rules. The resulting solution accurately represents the numerical consumption rule over the entire interval of feasible wealth values with remarkably few computations.
keywords:
  - Dynamic Stochastic Optimization
parts:
  jel_codes: D14; C61; G11
exports:
  - format: tex+pdf
    template: arxiv_two_column
    output: moderation_letters.pdf
---

# Introduction

Solving a consumption-saving problem using numerical methods requires the modeler to choose how to represent a policy function. In the stochastic case, where analytical solutions are generally not available, a common approach is to use low-order polynomial splines that exactly match the function at a finite set of gridpoints, and then to assume that interpolated or extrapolated versions of that spline represent the function well at the continuous infinity of unmatched points. {cite:t}`carrollEGM` developed the endogenous gridpoints method, which has become a standard tool for computing consumption at gridpoints determined endogenously using the Euler equation.

Unfortunately, these approximation methods are not very well-behaved outside the original range of gridpoints. This is a serious problem: in the presence of uncertainty, the consumption rule will need to be evaluated outside of _any_ prespecified grid, because starting from the top gridpoint, a large enough realization of the uncertain variable will push next period's realization of assets above that top. Linear extrapolation of consumption can produce theoretically inconsistent predictions. For example, although theory proves that precautionary saving is always positive, a linearly extrapolated numerical approximation eventually predicts negative precautionary saving (at the point in the figure where the extrapolated locus crosses the horizontal axis).

```{figure} ../images/ExtrapProblemPlot
:label: fig:ExtrapProblem
:alt: Graph showing that precautionary saving, approximated with linear extrapolation, incorrectly becomes negative for large market resources.
:align: center
:width: 80%

For Large Enough $\mNrm_{T-1}$, Predicted Precautionary Saving is Negative (Oops!)
```

This error cannot be fixed by extending the upper gridpoint; in the presence of serious uncertainty, the consumption rule will need to be evaluated outside of _any_ prespecified grid (because starting from the top gridpoint, a large enough realization of the uncertain variable will push next period's realization of assets above that top; a similar argument applies below the bottom gridpoint). While an extrapolation technique can prevent this from being fatal (for example by excluding negative precautionary saving), the problem is often dealt with using inelegant methods whose implications for accuracy are difficult to gauge.

This paper argues that, at least in the context of the standard consumption problem, a better approach is to rely upon the fact that in the absence of uncertainty, the optimal consumption function has a simple analytical solution. The key insight is that, under standard assumptions, the consumer who faces an uninsurable labor income risk will consume less than a consumer with the same path for expected income who does not perceive any uncertainty. Following {cite:t}`Leland1968`, {cite:t}`Sandmo1970`, and {cite:t}`Kimball1990`, the 'realistic' consumer, who _does_ perceive the risks, will engage in 'precautionary saving,' so the perfect foresight riskless solution provides an upper bound to the solution that will actually be optimal. A lower bound is provided by the behavior of a consumer who has the subjective belief that the future level of income will be the worst that it can possibly be. This consumer, too, behaves according to the convenient analytical perfect foresight solution, but his certainty is that of a pessimist who is extremely overconfident in his pessimism.

We build on bounds for the consumption function and limiting MPCs established by {cite:t}`StachurskiToda2019JET`, {cite:t}`MST2020JET`, {cite:t}`Carroll2001MPCBound`, and {cite:t}`MaToda2021SavingRateRich` in buffer-stock theory. Using results from {cite:t}`CarrollShanker2024`, we show how to use these bounds to constrain the shape and characteristics of the solution to the 'realist' problem characterized by {cite:t}`Carroll1997`. Imposition of these constraints clarifies and speeds the solution of the realist's problem.

# The Model

Consider a consumer who correctly perceives all risks. The consumer with CRRA utility $\uFunc(\cNrm) = \cNrm^{1-\CRRA}/(1-\CRRA)$ (where prudence $\uFunc''' > 0$ induces precautionary saving) maximizes expected lifetime utility:

```{math}
:label: eq:MaxProb
\max~\Ex_{t}\left[\sum_{n=0}^{T-t}\DiscFac^{n} \uFunc(\cLvl_{t+n})\right]
```

subject to $\aLvl_{t} = \mLvl_{t}-\cLvl_{t}$, $\pLvl_{t+1} = \pLvl_{t} \PermGroShk_{t+1}$, $\yLvl_{t+1} = \pLvl_{t+1}\tranShk_{t+1}$, $\mLvl_{t+1} = \aLvl_{t}\Rfree_{t+1} + \yLvl_{t+1}$, where $\mLvl$ denotes market resources, $\pLvl$ permanent labor income, $\aLvl$ assets, and $\yLvl$ noncapital income. Income evolves via the Friedman-Muth process ({cite:t}`Friedman1957` distinguished permanent from transitory income; {cite:t}`Muth1960` provided the stochastic framework): $\PermGroShk_{t+1} = \PermGroFac_{t+1} \permShk_{t+1}$ with permanent shocks $\permShk$ having mean unity and bounded support $[\permShkMin, \permShkMax]$ where $0 < \permShkMin \leq 1 \leq \permShkMax < \infty$, and transitory shocks $\tranShk_{t+1}$ taking value $0$ with probability $\WorstProb > 0$ (unemployment) or $\tranShkEmp_{t+1}/(1-\WorstProb)$ otherwise, with $\Ex[\tranShkEmp]=1$.

This problem can be rewritten (see {cite:t}`SolvingMicroDSOPs` for a proof) in a more convenient form in which choice and state variables are normalized by the level of permanent income, e.g., $\mNrm=\mLvl/\pLvl$. When that is done, the Bellman equation for the transformed version of the consumer's problem is

```{math}
:label: eq:vNormed
\vFunc_{t}(\mNrm_{t}) = \max_{\cNrm_{t}} ~~ \uFunc(\cNrm_{t})+\DiscFac \Ex_{t}[ \PermGroShk_{t+1}^{1-\CRRA}\vFunc_{t+1}(\mNrm_{t+1})]
```

with Euler equation $\uPrime(\cNrm_{t}) = \DiscFac \Rfree \Ex_{t}[ \PermGroShk_{t+1}^{-\CRRA} \uPrime(\cNrm_{t+1})]$.

For expositional simplicity, we assume $\CRRA \neq 1$ and set $\PermGroFac=1$, $\permShk=1$ (no permanent income growth or shocks), working with the infinite-horizon formulation; results extend to general cases. Parameter restrictions (below) are required.

We define the absolute patience factor $\AbsPatFac\equiv(\DiscFac\Rfree)^{1/\CRRA}$. {cite:t}`CarrollShanker2024` shows that a finite solution requires: (i) finite-value-of-autarky condition (FVAC) $0<\DiscFac\PermGroFac^{1-\CRRA}\Ex[\permShk^{1-\CRRA}]<1$, (ii) absolute-impatience condition (AIC) $\AbsPatFac<1$, (iii) return-impatience condition (RIC) $\AbsPatFac/\Rfree<1$, (iv) growth-impatience condition (GIC) $\AbsPatFac/\PermGroFac<1$, and (v) finite-human-wealth condition (FHWC) $\PermGroFac/\Rfree<1$. These patience conditions ensure the consumption bounds and limiting MPCs used in our method.

# The Method of Moderation

## The Optimist, the Pessimist, and the Realist

As a preliminary to our solution, define $\hNrmOpt$ as end-of-period human wealth (the present discounted value of future labor income) for a perfect foresight version of the problem of a 'risk optimist:' a consumer who believes with perfect confidence that the shocks will always take their expected value of 1, $\tranShk_{t+n} = \Ex[\tranShk]=1~\forall~n>0$. The solution to a perfect foresight problem of this kind takes the form

```{math}
:label: eq:cFuncOpt
\cFuncOpt(\mNrm) = (\mNrm + \hNrmOpt)\MPCmin
```

for a constant minimal marginal propensity to consume $\MPCmin$. We similarly define $\hNrmPes$ as 'minimal human wealth,' the present discounted value of labor income if the shocks were to take on their worst value in every future period $\tranShk_{t+n} = \tranShkMin$ $\forall~n>0$ (which we define as corresponding to the beliefs of a 'pessimist'). We will call a 'realist' the consumer who correctly perceives the true probabilities of the future risks and optimizes accordingly.

A first useful point is that, for the realist, a lower bound for the level of market resources is the natural borrowing constraint $\mNrmMin = -\hNrmPes$ derived by {cite:t}`Aiyagari1994` and {cite:t}`Huggett1993`, because if $\mNrm$ equalled this value then there would be a positive finite chance (however small) of receiving $\tranShk_{t+n} = \tranShkMin$ in every future period, which would require the consumer to set $\cNrm$ to zero in order to guarantee that the intertemporal budget constraint holds. Since consumption of zero yields negative infinite utility, {cite:t}`Zeldes1989` and {cite:t}`Deaton1991` show that the solution to the realist consumer's problem is not well defined for values of $\mNrm < \mNrmMin$, and the limiting value of the realist's $\cNrm$ is zero as $\mNrm \downarrow \mNrmMin$, where $\modRte(\logmNrmEx) \to 0$ as $\logmNrmEx \to -\infty$ (established in {cite:t}`CarrollShanker2024`).

It is convenient to define 'excess' market resources as the amount by which actual resources exceed the lower bound, and 'excess' human wealth as the amount by which mean expected human wealth exceeds guaranteed minimum human wealth:

```{math}
:label: eq:ExcessDef
\begin{aligned}
\mNrmEx &= \mNrm+\overbrace{\hNrmPes}^{=-\mNrmMin} \\
\hNrmEx &= \hNrmOpt-\hNrmPes.
\end{aligned}
```

We now define the optimal consumption rules for the two perfect foresight problems. The 'pessimist' perceives human wealth to be equal to its minimum feasible value $\hNrmPes$ with certainty, so consumption is given by the perfect foresight solution

```{math}
:label: eq:cFuncPes
\cFuncPes(\mNrm) = (\mNrm+\hNrmPes)\MPCmin = \mNrmEx\MPCmin.
```

The 'optimist,' on the other hand, pretends that there is no uncertainty about future income, and therefore consumes

```{math}
:label: eq:cFuncOptExcess
\cFuncOpt(\mNrm) = (\mNrmEx + \hNrmEx)\MPCmin = \cFuncPes(\mNrm)+\hNrmEx \MPCmin.
```

## The Moderation Ratio

It seems obvious that the spending of the realist will be strictly greater than that of the pessimist and strictly less than that of the optimist.

```{figure} ../images/IntExpFOCInvPesReaOptNeedHiPlot
:label: fig:IntExpFOCInvPesReaOptNeedHi
:alt: Graph showing the realist's consumption function is bounded by the pessimist's (lower) and optimist's (upper) consumption functions.
:align: center
:width: 80%

Moderation Illustrated: $\cFuncPes < \cFuncApprox < \cFuncOpt$
```

The proof is more difficult than might be imagined, but the necessary work is done in {cite:t}`CarrollShanker2024` so we will take the proposition as a fact and proceed by manipulating the inequality:

$$
\begin{array}{rcl}
\cFuncOpt(\mNrmMin+\mNrmEx) > & \cFuncReal(\mNrmMin+\mNrmEx) & > \cFuncPes(\mNrmMin+\mNrmEx) \\
-\cFuncPes(\mNrmMin+\mNrmEx) < & -\cFuncReal(\mNrmMin+\mNrmEx) & < -\cFuncOpt(\mNrmMin+\mNrmEx) \\
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

which has the virtue that it is _asymptotically linear_ in the limit as $\logmNrmEx$ approaches $+\infty$. The method uses standard transformations for unbounded domains: logit maps $\modRte \in (0,1)$ to $\logitModRte \in (-\infty, \infty)$ with inverse sigmoid $\modRte = 1/(1+\exp(-\logitModRte))$; log maps $(\mNrm - \mNrmMin) \in (0, \infty)$ to $\logmNrmEx \in (-\infty, \infty)$. As $\modRte \to 1$ (realist approaches optimist), $\logitModRte \to +\infty$; as $\modRte \to 0$ (realist approaches pessimist), $\logitModRte \to -\infty$.

Given $\logitModRte$, the consumption function can be recovered from

```{math}
:label: eq:cFuncHi
\cFuncReal = \cFuncPes+\overbrace{\frac{1}{1+\exp(-\logitModRte)}}^{=\modRte} \hNrmEx \MPCmin.
```

Thus, the procedure is to calculate $\logitModRte$ at the points $\logmNrmEx$ corresponding to the log of the $\mNrmEx$ points defined above, and then using these to construct an interpolating approximation $\logitModRteApprox$ from which we indirectly obtain our approximated consumption rule $\cFuncApprox$ (an approximation to the true $\cFuncReal$) by substituting $\logitModRteApprox$ for $\logitModRte$ in equation {eq}`eq:cFuncHi$.

Because this method relies upon the fact that the problem is easy to solve if the decision maker has unreasonable views (either in the optimistic or the pessimistic direction), and because the correct solution is always between these immoderate extremes, we call our solution procedure the 'method of moderation.'

Results are shown in {ref}`fig:ExtrapProblemSolved`; a reader with very good eyesight might be able to detect the barest hint of a discrepancy between the Truth and the Approximation at the far righthand edge of the figure, a stark contrast with the calamitous divergence evident in {ref}`fig:ExtrapProblem`.

```{figure} ../images/ExtrapProblemSolvedPlot
:label: fig:ExtrapProblemSolved
:alt: Graph showing that the Method of Moderation produces an accurate extrapolated consumption function that does not predict negative precautionary saving.
:align: center
:width: 80%

Extrapolated $\cFuncApprox_{T-1}$ Constructed Using the Method of Moderation
```

## A Tighter Upper Bound

{cite:t}`CarrollShanker2024` derives an explicit formula for the MPC at the natural borrowing constraint: $\MPCmax = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)$ where $\WorstProb$ is the unemployment probability derived by {cite:t}`CarrollToche2009`. This provides a tighter upper bound near the constraint, extending the explicit limiting MPC formulas established in buffer-stock theory by {cite:t}`MaToda2021SavingRateRich`. Strict concavity of the consumption function implies $\cFuncReal(\mNrm) < \MPCmax \mNrmEx$ for low wealth, while the optimist's bound $\cFuncReal(\mNrm) < \cFuncOpt(\mNrm) = (\mNrmEx+\hNrmEx)\MPCmin$ is tighter for high wealth.

The two upper bounds intersect at the cusp point:

```{math}
:label: eq:mNrmCusp
\mNrmCusp = -\hNrmPes + \frac{\MPCmin(\hNrmOpt-\hNrmPes)}{\MPCmax-\MPCmin}
```

This intersection occurs in the feasible region since $\MPCmax > \MPCmin$ under the stated conditions (the MPC is highest when wealth is lowest).

For $\mNrm < \mNrmCusp$, define the low-region moderation ratio using the tighter bound:

```{math}
:label: eq:modRteLoTightUpBd
\modRteLoTightUpBd(\logmNrmEx) = \frac{\cFuncReal(\mNrmMin+e^{\logmNrmEx})e^{-\logmNrmEx}-\MPCmin}{\MPCmax-\MPCmin}
```

This ratio measures how far consumption per unit of wealth exceeds the optimist's MPC, relative to the maximum possible excess. Applying the logit transformation and interpolating as before yields consumption satisfying both upper bounds throughout.

For computational robustness, construct a three-piece approximation: below the cusp using the tight bound, near the cusp using Hermite interpolation matching levels and slopes at adjacent gridpoints, above the cusp using the original optimist bound. This ensures continuous, differentiable consumption functions respecting all theoretical constraints.

```{figure} ../images/IntExpFOCInvPesReaOptNeed45Plot
:label: fig:IntExpFOCInvPesReaOptNeed45
:alt: A diagram showing the true consumption function bounded above by both the optimist's consumption rule and a tighter linear bound originating from the natural borrowing constraint.
:align: center
:width: 80%

A Tighter Upper Bound
```

## Hermite Interpolation

The numerical accuracy of the method of moderation depends critically on the quality of function approximation between gridpoints. Although linear interpolation that matches the level of $\cFuncReal$ at the gridpoints is simple, Hermite interpolation offers a considerable advantage. Differentiating {eq}`eq:cFuncHi` with respect to $\mNrm$ yields a moderation form for the marginal propensity to consume:

```{math}
:label: eq:MPCModeration
\frac{\partial \cFuncReal}{\partial \mNrm} = (1-\MPCmod)\,\MPCmin + \MPCmod\,\MPCmax
```

where the weight $\MPCmod = (\hNrmEx/\mNrmEx) \cdot \MPCmin/(\MPCmax-\MPCmin) \cdot \modRteMu$. Theory guarantees $\MPCmin \leq \partial \cFuncReal/\partial \mNrm \leq \MPCmax$ at gridpoints where the Euler equation holds. At very high wealth, $\MPCmod \to 0$ and the MPC approaches $\MPCmin$; near the borrowing constraint, $\MPCmod \to 1$ and the MPC approaches $\MPCmax$.

For Hermite interpolation, compute $\modRteMu$ at gridpoints, then derive $\logitModRteMu = \modRteMu/[\modRte(1-\modRte)]$ for slope data. By matching both the level and the derivative of the $\cFuncReal$ function at the gridpoints, {cite:t}`BenvenisteScheinkman1979` and {cite:t}`MilgromSegal2002` show that the consumption rule derived from such interpolation numerically satisfies the Euler equation at each gridpoint for which the problem has been solved.

# Extensions

## Value Function

Often it is useful to know the value function as well as the consumption rule. Fortunately, many of the tricks used when solving for the consumption rule have a direct analogue in approximation of the value function. Define the inverse value function transformation $\vInvOpt = ((1-\CRRA)\vFuncOpt)^{1/(1-\CRRA)}$, which under perfect foresight equals $(\mNrmEx+\hNrmEx)\MPCmin^{-\CRRA/(1-\CRRA)}$ (linear in market resources). The value moderation ratio $\valModRteReal$ measures proximity to the optimist's value, with logit transformation $\logitValModRteReal$ applied as before. Interpolate $\logitValModRteReal$ at gridpoints and invert to obtain $\vFuncReal = \uFunc(\vInvReal)$.

## Stochastic Rate of Return

For i.i.d. returns with $\log \Risky \sim \Nrml(r + \equityPrem - \std^{2}_{\risky}/2,\std^{2}_{\risky})$, {cite:t}`Samuelson1969,Merton1969,Merton1971` showed that for a consumer without labor income (or with perfectly forecastable labor income) the consumption function is linear, with an MPC $= 1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$. See {cite:t}`CRRA-RateRisk,BBZ2016SkewedWealth` for extensions. Simply substitute this stochastic MPC for $\MPCmin$ throughout our formulas. The pessimist and optimist still perceive their income streams with certainty, but both face the same stochastic return. All moderation ratio calculations proceed identically. Extensions to serially correlated returns require tracking the return state as an additional state variable, complicating the analysis but not fundamentally altering the approach.

# Conclusion

The method proposed here is not universally applicable. For example, the method cannot be used for problems for which upper and lower bounds to the 'true' solution are not known. But many problems do have obvious upper and lower bounds, and in those cases (as in the consumption example used in the paper), the method may result in substantial improvements in accuracy and stability of solutions.

