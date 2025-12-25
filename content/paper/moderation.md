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
    output: moderation.pdf
---

# Introduction

Solving a consumption, investment, portfolio choice, or similar intertemporal
optimization problem using numerical methods generally requires the modeler to
choose how to represent a policy or value function. In the stochastic case,
where analytical solutions are generally not available, a common approach is to
use low-order polynomial splines that exactly match the function (and maybe some
derivatives) at a finite set of gridpoints, and then to assume that interpolated
or extrapolated versions of the matching polynomial are a good representation elsewhere.

This paper argues that, at least in the context of a standard consumption problem,
a better approach is available, which relies upon the fact that in the absence of
uncertainty, the optimal consumption function has a simple analytical solution. The key insight is that, under standard
assumptions, the consumer who faces an uninsurable labor income risk will
consume less than a consumer with the same path for expected income but who does
not perceive any uncertainty as being attached to that future income. The
'realistic' consumer, who _does_ perceive the risks, will engage in
'precautionary saving' {cite:p}`Leland1968,Sandmo1970,Kimball1990`, so the perfect foresight riskless solution provides an
upper bound to the solution that will actually be optimal. A lower bound is
provided by the behavior of a consumer who has the subjective belief that the
future level of income will be the worst that it can possibly be. This consumer,
too, behaves according to the convenient analytical perfect foresight solution,
but his certainty is that of a pessimist perfectly confident in his pessimism.

We build on bounds for the consumption function and limiting MPCs established in buffer-stock theory and related work {cite:p}`StachurskiToda2019JET,MST2020JET`. Using results from {cite:t}`CarrollShanker2024`, we show how to use these upper
and lower bounds to tightly constrain the shape and characteristics of the
solution to the 'realist's problem (that is, the solution to the problem of a consumer who correctly perceives the risks to future income and behaves rationally in response) {cite:p}`Carroll1997`. Imposition of these constraints can
clarify and speed the solution of the realist's problem.

After showing how to use the method in the baseline case, we show how to refine
it to encompass an even tighter theoretical bound, and how to extend it
to solve a problem in which the consumer faces both labor income risk and rate-of-return risk.

# The Realist's Problem

Consider a consumer who correctly perceives all risks. The consumer's problem is to maximize expected lifetime utility:

```{math}
:label: eq:MaxProb
\max~\Ex_{t}\left[\sum_{n=0}^{T-t}\DiscFac^{n} \uFunc(\cLvl_{t+n})\right]
```

where the utility function is CRRA with risk aversion parameter $\CRRA > 0$:

```{math}
:label: eq:UtilityFunc
\uFunc(\cNrm) = \begin{cases}
\frac{\cNrm^{1-\CRRA}}{1-\CRRA} & \text{if } \CRRA \neq 1 \\
\log \cNrm & \text{if } \CRRA = 1.
\end{cases}
```

This utility function satisfies prudence ($\uFunc''' > 0$), which ensures the consumer exhibits precautionary saving in response to income uncertainty. The optimization is subject to the budget constraints

```{math}
:label: eq:DBCLevel
\begin{aligned}
\aLvl_{t}  &= \mLvl_{t}-\cLvl_{t} \\
\pLvl_{t+1}  &= \pLvl_{t} \PermGroShk_{t+1}  \\
\yLvl_{t+1}  &= \pLvl_{t+1}\tranShk_{t+1} \\
\mLvl_{t+1}  &= \aLvl_{t}\Rfree_{t+1} + \yLvl_{t+1}
\end{aligned}
```

where the variables are defined as

$$
\begin{aligned}
\DiscFac &\text{ - pure time discount factor} \\
\aLvl_{t} &\text{ - assets at the end of period } t \\
\cLvl_{t} &\text{ - consumption in period } t \\
\mLvl_{t} &\text{ - `market resources' available for consumption} \\
\pLvl_{t+1} &\text{ - `permanent labor income' in period } t+1 \\
\Rfree_{t+1} &\text{ - gross interest rate from period } t \text{ to } t+1 \\
\yLvl_{t+1} &\text{ - noncapital income in period } t+1.
\end{aligned}
$$

The exogenous variables evolve according to the _Friedman-Muth Income Process_[^friedman-muth-formalization]:

```{math}
:label: eq:ExogVars
\begin{aligned}
\PermGroShk_{t+1} &= \PermGroFac_{t+1} \permShk_{t+1}  \\
\tranShk_{t+1} &= \begin{cases}
0 & \text{with probability } \WorstProb > 0 \\
\frac{\tranShkEmp_{t+1}}{1-\WorstProb} & \text{with probability } (1-\WorstProb)
\end{cases}
\end{aligned}
```

where $\PermGroFac_{t+1}$ is the deterministic permanent income growth factor, and the permanent shocks to income $\permShk_{t+1}$ are independently and identically distributed with mean $\Ex[\permShk_{t+1}] = 1$ and support $[\permShkMin, \permShkMax]$ where $0 < \permShkMin \leq 1 \leq \permShkMax < \infty$,[^bounded-support] and the transitory shocks to income $\tranShkEmp_{t+1}$ are independently distributed with mean $\Ex[\tranShkEmp_{t+1}] = 1$ and bounded support $\tranShkEmpMin \leq \tranShkEmp_{t+1} \leq \tranShkEmpMax$ where $0 \leq \tranShkEmpMin \leq 1 \leq \tranShkEmpMax < \infty$.

[^bounded-support]:
    Bounded support ($\permShkMax < \infty$) ensures existence of well-defined upper and lower bounds for the consumption function. Results in the literature also exist for unbounded distributions with finite moments (e.g., lognormal shocks), but establishing the moderation ratio bounds used in this paper requires bounded support. Extensions to unbounded shocks are beyond the scope of this paper.

[^friedman-muth-formalization]:
    {cite:t}`Friedman1957` distinguished permanent (long-term earning capacity) from transitory (temporary fluctuations) income components. {cite:t}`Muth1960` provided the stochastic framework for modeling these as random processes. Our specification combines both insights to model unemployment risk and permanent income growth.

It turns out (see {cite:t}`SolvingMicroDSOPs` for a proof) that this problem can
be rewritten in a more convenient form in which choice and state variables are
normalized by the level of permanent income, e.g., using nonbold font for normalized variables, $\mNrm_{t}=\mLvl_{t}/\pLvl_{t}$. When that is done, the
Bellman equation for the transformed version of the consumer's problem is

```{math}
:label: eq:vNormed
\begin{aligned}
\vFunc_{t}(\mNrm_{t}) &= \max_{\cNrm_{t}} ~~ \uFunc(\cNrm_{t})+\DiscFac \Ex_{t}[ \PermGroShk_{t+1}^{1-\CRRA}\vFunc_{t+1}(\mNrm_{t+1})] \\
&\text{s.t.} \\
\aNrm_{t} &= \mNrm_{t}-\cNrm_{t} \\
\mNrm_{t+1} &= \left(\Rfree/\PermGroShk_{t+1}\right)\aNrm_{t}+\tranShk_{t+1},
\end{aligned}
```

and because we have not imposed a liquidity constraint, the solution satisfies
the Euler equation

```{math}
:label: eq:cEuler
\uPrime(\cNrm_{t}) = \DiscFac \Rfree \Ex_{t}[ \PermGroShk_{t+1}^{-\CRRA} \uPrime(\cNrm_{t+1})].
```

We define the absolute patience factor as $\AbsPatFac\equiv(\DiscFac\Rfree)^{1/\CRRA}$. A finite solution requires {cite:p}`CarrollShanker2024`: (i) the finite-value-of-autarky condition (FVAC) $0<\DiscFac\PermGroFac^{1-\CRRA}\Ex[\permShk^{1-\CRRA}]<1$; (ii) the absolute-impatience condition (AIC) $\AbsPatFac<1$; (iii) the return-impatience condition (RIC) $\AbsPatFac/\Rfree<1$; (iv) the growth-impatience condition (GIC) $\AbsPatFac/\PermGroFac<1$; and (v) the finite-human-wealth condition (FHWC) $\PermGroFac/\Rfree<1$. These conditions ensure existence of upper and lower bounds on consumption {cite:p}`Carroll2001MPCBound,StachurskiToda2019JET` and pin down limiting MPCs {cite:p}`MaToda2021SavingRateRich`.

For expositional simplicity in what follows, we set $\PermGroFac=1$ and assume $\permShk_{t+n}=1$ with probability 1 for all $n>0$ (no permanent income growth or shocks), and drop time subscripts except where context requires, working with the infinite-horizon formulation.[^crra-not-one] Under these simplifications, FVAC becomes $0<\DiscFac<1$, the GIC coincides with the AIC, and the FHWC reduces to $\Rfree>1$. All results apply equally to finite-horizon models via backward recursion from terminal period $T$, and to models with permanent income growth by appropriately adjusting the patience conditions above.

[^crra-not-one]:
    For notational simplicity, we henceforth assume $\CRRA \neq 1$. Most subsequent derivations involving transformations of the value function (such as $\vInvOpt$ and $\vInvReal$) contain expressions with denominators $(1-\CRRA)$ that are undefined when $\CRRA=1$. The case $\CRRA=1$ (logarithmic utility) requires parallel derivations that exploit the simplifications arising from $\log$ utility; the economic insights remain analogous.

# Benchmark: The Method of Endogenous Gridpoints

For comparison to our new solution method, we use the endogenous gridpoints
solution to the microeconomic problem presented in {cite:t}`carrollEGM`. That
method computes the level of consumption at a set of gridpoints for market
resources $\mNrm$ that are determined endogenously using the Euler equation. The
consumption function is then constructed by linear interpolation among the
gridpoints thus found. Extensions of this method include multi-dimensional problems {cite:p}`BarillasFV2007`, occasionally binding constraints {cite:p}`HintermaierKoeniger2010`, non-smooth and non-concave problems {cite:p}`Fella2014`, discrete-continuous choice models {cite:p}`IskhakovRustSchjerning2017`, and comprehensive treatments of theory and practice {cite:p}`White2015`. For shock discretization in numerical solutions, standard methods include {cite:p}`TauchenHussey1991,tauchen1986`.

{cite:t}`SolvingMicroDSOPs` describes a specific calibration of the model and
constructs a solution using five gridpoints chosen to capture the structure of
the consumption function reasonably well at values of $\mNrm$ near the
infinite-horizon target value (See those notes for details).[^grid-size-note]

[^grid-size-note]:
    The five gridpoints are used for illustration; production codes typically use 30-80 gridpoints for accurate solutions.

Unfortunately, this endogenous gridpoints solution is not very well-behaved
outside the original range of gridpoints targeted by the solution method.
(Though other common solution methods are no better outside their own predefined
ranges). {ref}`fig:ExtrapProblem` demonstrates the point by plotting the amount
of precautionary saving implied by a linear extrapolation of our approximated
consumption rule (the consumption of the perfect foresight consumer
$\cFuncOpt_{T-1}$ minus our approximation to optimal consumption under
uncertainty, $\cFuncApprox_{T-1}$). Although theory proves that precautionary
saving is always positive, the linearly extrapolated numerical approximation
eventually predicts negative precautionary saving (at the point in the figure
where the extrapolated locus crosses the horizontal axis).

```{figure} ../images/ExtrapProblemPlot
:label: fig:ExtrapProblem
:alt: Graph showing that precautionary saving, approximated with linear extrapolation, incorrectly becomes negative for large market resources.
:align: center
:width: 80%

For Large Enough $\mNrm_{T-1}$, Predicted Precautionary Saving is Negative (Oops!)
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

# The Method of Moderation

## The Optimist, the Pessimist, and the Realist

As a preliminary to our solution, define $\hNrmOpt$[^optimist-human-wealth]
as end-of-period human wealth (the present discounted value of future labor
income) for a perfect foresight version of the problem of a 'risk optimist:' a
consumer who believes with perfect confidence that the shocks will
always take their expected value of 1,
$\tranShk_{t+n} = \Ex[\tranShk]=1~\forall~n>0$. The solution to a perfect
foresight problem of this kind takes the form[^derivation]

[^derivation]:
    For a derivation, see {cite:t}`CarrollShanker2024`; $\MPCmin$ is defined therein as the MPC of the perfect foresight consumer with horizon $T-t$.

[^optimist-human-wealth]:
    Setting $\tranShk_{t+n}=1$ (the optimist's assumption), human wealth has three equivalent forms: (i) backward recursion: $\hNrmOpt_{T} = 0$, $\hNrmOpt_{t} = (\PermGroFac/\Rfree)(1 + \hNrmOpt_{t+1})$ for $t = T-1, T-2, \ldots$; (ii) forward sum: $\hNrmOpt_{t} = \sum_{n=1}^{T-t}(\PermGroFac/\Rfree)^{n}$; (iii) infinite-horizon: $\hNrmOpt = \PermGroFac/(\Rfree-\PermGroFac)$ if $\Rfree>\PermGroFac$. When $\PermGroFac=1$, $\hNrmOpt_{t} = \sum_{n=1}^{T-t}(1/\Rfree)^{n}$ and $\hNrmOpt = 1/(\Rfree-1)$.

```{math}
:label: eq:cFuncOpt
\cFuncOpt(\mNrm) = (\mNrm + \hNrmOpt)\MPCmin
```

for a constant minimal marginal propensity to consume $\MPCmin$ (defined in footnote [^mpc-min-definition]). We similarly define
$\hNrmPes$[^pessimist-human-wealth] as 'minimal human wealth,' the present
discounted value of labor income if the shocks were to take on their worst
value in every future period
$\tranShk_{t+n} = \tranShkMin$ $\forall~n>0$ (which we define as corresponding to the beliefs of a 'pessimist').

[^mpc-min-definition]:
    The MPC of the perfect foresight consumer with horizon $T-t$ (optimist's behavior at high wealth). Three forms: (i) backward recursion $\MPCmin_{t}=\MPCmin_{t+1}/(\MPCmin_{t+1}+\AbsPatFac/\Rfree)$ with $\MPCmin_T=1$; (ii) forward sum $\MPCmin_{t}=(\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n})^{-1}$; (iii) infinite-horizon $\MPCmin=1-\AbsPatFac/\Rfree$.

[^pessimist-human-wealth]:
    Setting $\tranShk_{t+n}=\tranShkMin~\forall~n>0$ (the pessimist's assumption), where $\tranShkMin \geq 0$ is the worst income realization (zero for unemployment, positive for underemployment), minimal human wealth can be calculated three ways: (i) backward recursion: $\hNrmPes_{T}=0$, $\hNrmPes_{t}=(\PermGroFac/\Rfree)(\tranShkMin + \hNrmPes_{t+1})$ for $t = T-1, T-2, \ldots$; (ii) forward sum: $\hNrmPes_{t}=\tranShkMin\sum_{n=1}^{T-t}(\PermGroFac/\Rfree)^{n}$; (iii) infinite-horizon: $\hNrmPes=\tranShkMin\PermGroFac/(\Rfree-\PermGroFac)$ if $\Rfree>\PermGroFac$. When $\tranShkMin=0$, $\hNrmPes=0$.

We will call a 'realist' the consumer who correctly perceives the true
probabilities of the future risks and optimizes accordingly.

A first useful point is that, for the realist, a lower bound for the level of
market resources is $\mNrmMin = -\hNrmPes$ {cite:p}`Aiyagari1994,Huggett1993`, because if $\mNrm$
equalled this value then there would be a positive finite chance (however small)
of receiving $\tranShk_{t+n} = \tranShkMin$ in every future period, which
would require the consumer to set $\cNrm$ to zero in order to guarantee that the
intertemporal budget constraint holds. Since consumption of zero yields negative
infinite utility, the solution to the realist consumer's problem is not well
defined for values of $\mNrm < \mNrmMin$ {cite:p}`Zeldes1989,Deaton1991`, and the limiting value of the
realist's $\cNrm$ is zero as $\mNrm \downarrow \mNrmMin$, where $\modRte(\logmNrmEx) \to 0$ as $\logmNrmEx \to -\infty$ (established in {cite:t}`CarrollShanker2024`).

Given this result, it will be convenient to define 'excess' market resources as
the amount by which actual resources exceed the lower bound, and 'excess' human
wealth as the amount by which mean expected human wealth exceeds guaranteed
minimum human wealth:[^delta-notation]

```{math}
:label: eq:ExcessDef
\begin{aligned}
\mNrmEx &= \mNrm+\overbrace{\hNrmPes}^{=-\mNrmMin} \\
\hNrmEx &= \hNrmOpt-\hNrmPes.
\end{aligned}
```

[^delta-notation]:
    Here $\Delta$ denotes excess above minimum, not a time difference.

We can now transparently define the optimal consumption rules for the two
perfect foresight problems, those of the 'optimist' and the 'pessimist.' The
'pessimist' perceives human wealth to be equal to its minimum feasible value
$\hNrmPes$ with certainty, so consumption is given by the perfect foresight
solution

```{math}
:label: eq:cFuncPes
\begin{aligned}
\cFuncPes(\mNrm) &= (\mNrm+\hNrmPes)\MPCmin \\
&= \mNrmEx\MPCmin .
\end{aligned}
```

The 'optimist,' on the other hand, pretends that there is no uncertainty about
future income, and therefore consumes

```{math}
:label: eq:cFuncOptExcess
\begin{aligned}
\cFuncOpt(\mNrm) &= (\mNrm +\hNrmPes - \hNrmPes + \hNrmOpt )\MPCmin \\
&= (\mNrmEx + \hNrmEx)\MPCmin \\
&= \cFuncPes(\mNrm)+\hNrmEx \MPCmin .
\end{aligned}
```

## The Consumption Function

It seems obvious that the spending of the realist will be strictly greater than
that of the pessimist and strictly less than that of the optimist.
{ref}`fig:IntExpFOCInvPesReaOptNeedHi` illustrates the proposition for the
consumption rule in period $T-1$.

```{figure} ../images/IntExpFOCInvPesReaOptNeedHiPlot
:label: fig:IntExpFOCInvPesReaOptNeedHi
:alt: Graph showing the realist's consumption function is bounded by the pessimist's (lower) and optimist's (upper) consumption functions.
:align: center
:width: 80%

Moderation Illustrated: $\cFuncPes < \cFuncApprox < \cFuncOpt$
```

The proof is more difficult than might be imagined, but the necessary work is
done in {cite:t}`CarrollShanker2024`[^proof] so we will take the proposition as
a fact and proceed by manipulating the inequality:

[^proof]: Under bounded shocks with strictly positive support (nondegenerate lower and upper bounds), the consumption function is strictly increasing and concave, and the moderation ratio lies in $(0,1)$; see {cite:t}`CarrollShanker2024`.

$$
\begin{array}{rcl}
\cFuncOpt(\mNrmMin+\mNrmEx) > & \cFuncReal(\mNrmMin+\mNrmEx) & > \cFuncPes(\mNrmMin+\mNrmEx) \\
-\cFuncOpt(\mNrmMin+\mNrmEx) < & -\cFuncReal(\mNrmMin+\mNrmEx) & < -\cFuncPes(\mNrmMin+\mNrmEx) \\
0 < & \cFuncReal(\mNrmMin+\mNrmEx)-\cFuncPes(\mNrmMin+\mNrmEx) & < \hNrmEx \MPCmin \\
0 < & \underbrace{\left(\frac{\cFuncReal(\mNrmMin+\mNrmEx)-\cFuncPes(\mNrmMin+\mNrmEx)}{\hNrmEx \MPCmin}\right)}_{\equiv \modRte} & < 1
\end{array}
$$

where the fraction in the middle of the last inequality is the moderation ratio
measuring how close the realist's consumption is to the optimist's behavior
(the numerator is the gap between the realist and pessimist) relative to the
maximum possible gap between optimist and pessimist. When $\modRte=0$, the
realist behaves like the pessimist (maximum precautionary saving); when
$\modRte=1$, the realist behaves like the optimist (no precautionary saving).[^modRte-ratio] Defining
$\logmNrmEx = \log \mNrmEx$ (which can range from $-\infty$ to
$\infty$), the object in the middle of the last inequality is

[^modRte-ratio]:
    Under bounded shocks {cite:p}`CarrollKimball1996,CarrollShanker2024`, $\modRte\in(0,1)$ strictly for all $\mNrm > \mNrmMin$. Equivalently, $\cFuncReal = \cFuncPes + \modRte\hNrmEx \MPCmin$, ensuring the realist consumes strictly between the pessimist and optimist.

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

which has the virtue that it is _asymptotically linear_ in the limit as
$\logmNrmEx$ approaches $+\infty$.[^asymptotically-linear] As
$\modRte \to 1$ (realist approaches optimist), $\logitModRte \to +\infty$;
as $\modRte \to 0$ (realist approaches pessimist), $\logitModRte \to -\infty$.[^ml-logit]

[^ml-logit]:
    The method uses standard ML transformations for unbounded domains: logit maps $\modRte \in (0,1)$ to $\logitModRte \in (-\infty, \infty)$ with inverse sigmoid $\modRte = 1/(1+\exp(-\logitModRte))$; log maps $(\mNrm - \mNrmMin) \in (0, \infty)$ to $\logmNrmEx \in (-\infty, \infty)$. These transformations enable accurate interpolation and are familiar to ML practitioners.

[^asymptotically-linear]:
    Under the GIC, $\logitModRte(\logmNrmEx)$ is asymptotically linear with slope $\asympSlope = \lim_{\logmNrmEx \to +\infty} \frac{\partial \logitModRte}{\partial \logmNrmEx} \geq 0$ as $\logmNrmEx \to +\infty$ (may equal zero in theory, but strictly positive on finite grids). Practical implications: (i) we extrapolate $\logitModRte$ linearly using the positive boundary slope; (ii) this preserves $\modRte\in(0,1)$ and hence $\cFuncPes < \cFuncApprox < \cFuncOpt$ throughout the extrapolation domain, even if the theoretical limiting slope vanishes.

Given $\logitModRte$, the consumption function can be recovered from

```{math}
:label: eq:cFuncHi
\cFuncReal = \cFuncPes+\overbrace{\frac{1}{1+\exp(-\logitModRte)}}^{=\modRte} \hNrmEx \MPCmin.
```

Thus, the procedure is to calculate $\logitModRte$ at the points
$\logmNrmEx$ corresponding to the log of the $\mNrmEx$ points
defined above, and then using these to construct an interpolating approximation
$\logitModRteApprox$ from which we indirectly obtain our approximated
consumption rule $\cFuncApprox$ (an approximation to the true $\cFuncReal$) by substituting $\logitModRteApprox$ for
$\logitModRte$ in equation {eq}`eq:cFuncHi`.

Because this method relies upon the fact that the problem is easy to solve if
the decision maker has unreasonable views (either in the optimistic or the
pessimistic direction), and because the correct solution is always between these
immoderate extremes, we call our solution procedure the 'method of moderation.'

Results are shown in {ref}`fig:ExtrapProblemSolved`; a reader with very good
eyesight might be able to detect the barest hint of a discrepancy between the
Truth and the Approximation at the far righthand edge of the figure - a stark
contrast with the calamitous divergence evident in {ref}`fig:ExtrapProblem`.

```{figure} ../images/ExtrapProblemSolvedPlot
:label: fig:ExtrapProblemSolved
:alt: Graph showing that the Method of Moderation produces an accurate extrapolated consumption function that does not predict negative precautionary saving.
:align: center
:width: 80%

Extrapolated $\cFuncApprox_{T-1}$ Constructed Using the Method of Moderation
```

## The Value Function

Often it is useful to know the value function as well as the consumption rule.
Fortunately, many of the tricks used when solving for the consumption rule have
a direct analogue in approximation of the value function.

Consider the perfect foresight (or "optimist's") problem in period $T-1$. Using
the fact that in a perfect foresight model the growth factor for consumption is
constant, we can use $\cNrm_{t} = \AbsPatFac \cdot \cNrm_{t-1}$ to calculate the value function in
period $T-1$:

$$
\begin{aligned}
\vFuncOpt_{T-1}(\mNrm_{T-1}) &\equiv  \uFunc(\cNrm_{T-1})+\DiscFac \uFunc(\cNrm_{T}) \\
&= \uFunc(\cNrm_{T-1})\left(1+\DiscFac \AbsPatFac^{1-\CRRA}\right) \\
&= \uFunc(\cNrm_{T-1})\left(1+\AbsPatFac/\Rfree\right) \\
&= \uFunc(\cNrm_{T-1})\underbrace{\PDV_{T-1}^{T}(\cNrm)/\cNrm_{T-1}}_{\equiv \PDVCoverc_{T-1}^{T}}
\end{aligned}
$$

where $\PDVCoverc_{t}^{T}=\PDV_{t}^{T}(\cNrm)/\cNrm_{t}$ is the present discounted value
of consumption, normalized by current consumption. Using the fact demonstrated
in {cite:t}`CarrollShanker2024` that
$\PDVCoverc_{t}^{T}=\MPCmin_{t}^{-1}$[^identity], a similar function can be
constructed recursively for earlier periods, yielding the general expression

[^identity]:
    Under perfect foresight, consumption grows at rate $\AbsPatFac$: $\cLvl_{t+n}=\cLvl_{t}\AbsPatFac^{n}$. Discounting yields $(\PDV_{t}^{T}(\cLvl)/\cLvl_{t})=\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n}=\MPCmin_{t}^{-1}$, so $\PDVCoverc_{t}^{T}=\MPCmin_{t}^{-1}$ (unchanged for normalized variables). In the infinite-horizon limit, we write simply $\PDVCoverc = \MPCmin^{-1}$.

```{math}
:label: eq:vFuncPF
\begin{aligned}
\vFuncOpt(\mNrm) &= \uFunc(\cFuncOpt(\mNrm))\PDVCoverc \\
&= \uFunc(\cFuncOpt(\mNrm))\MPCmin^{-1} \\
&= \uFunc((\mNrmEx+\hNrmEx)\MPCmin) \MPCmin^{-1} \\
&= \left[(\mNrmEx+\hNrmEx)^{1-\CRRA}/(1-\CRRA)\right] \cdot \left[\MPCmin^{1-\CRRA} \cdot \MPCmin^{-1}\right] \\
&= \uFunc(\mNrmEx+\hNrmEx)\MPCmin^{-\CRRA}.
\end{aligned}
```

This can be transformed as

$$
\begin{aligned}
\vInvOpt &\equiv  \left((1-\CRRA)\vFuncOpt\right)^{1/(1-\CRRA)}   \\
&= \cNrm\,\PDVCoverc^{1/(1-\CRRA)} \\
&= (\mNrmEx+\hNrmEx)\MPCmin^{-\CRRA/(1-\CRRA)}.
\end{aligned}
$$

We apply the same transformation to the value function for the problem with
uncertainty (the "realist's" problem):

$$
\vInvReal = \left((1-\CRRA)\vFuncReal(\mNrm)\right)^{1/(1-\CRRA)}
$$

and an excellent approximation to the value function can be obtained by
calculating the values of $\vInvReal$ at the same gridpoints used by the
consumption function approximation, and interpolating among those points.

However, as with the consumption approximation, we can do even better if we
realize that the $\vInvOpt$ function for the optimist's problem is an upper
bound for the $\vInv$ function in the presence of uncertainty, and the value
function for the pessimist is a lower bound. Analogously to {eq}`eq:modRte`,
define an upper-case

```{math}
:label: eq:valModRteReal
\valModRteReal(\logmNrmEx) = \left(\frac{\vInvReal(\mNrmMin+e^{\logmNrmEx})-\vInvPes(\mNrmMin+e^{\logmNrmEx})}{\hNrmEx \MPCmin \,\PDVCoverc^{1/(1-\CRRA)}}\right)
```

and an upper-case version of the $\logitModRte$ equation in {eq}`eq:chi`:

```{math}
:label: eq:ChiUpper
\begin{aligned}
\logitValModRteReal(\logmNrmEx) &= \log \left(\frac{\valModRteReal(\logmNrmEx)}{1-\valModRteReal(\logmNrmEx)}\right) \\
&= \log(\valModRteReal(\logmNrmEx)) - \log(1-\valModRteReal(\logmNrmEx))
\end{aligned}
```

and if we approximate these objects then invert them (as above with the $\modRte$
and $\logitModRte$ functions) we obtain a very high-quality approximation to our
inverted value function at the same points for which we have our approximated
value function:

```{math}
:label: eq:vInvHi
\vInvReal = \vInvPes+\overbrace{\left(\frac{1}{1+\exp(-\logitValModRteReal)}\right)}^{=\valModRteReal} \hNrmEx \MPCmin \,\PDVCoverc^{1/(1-\CRRA) }
```

from which we obtain our approximation to the value function as $\vFuncReal = \uFunc(\vInvReal)$.

# Extensions

## A Tighter Upper Bound

{cite:t}`CarrollShanker2024` derives an upper limit
$\MPCmax$[^mpc-max-definition] for the MPC as $\mNrm$ approaches its
lower bound, extending the explicit limiting MPC formulas established in buffer-stock theory {cite:p}`MaToda2021SavingRateRich`. Using this fact plus the strict concavity of the consumption
function yields the proposition that

$$
\cFuncReal(\mNrmMin+\mNrmEx) < \MPCmax \mNrmEx.
$$

[^mpc-max-definition]:
    The MPC at the natural borrowing constraint under maximum income uncertainty. For Friedman-Muth with unemployment probability $\WorstProb = \Pr(\tranShk = 0)$ {cite:p}`CarrollToche2009`: (i) backward recursion $\MPCmax_{t} = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree) (1 + \MPCmax_{t+1})$ with $\MPCmax_T = 1$; (ii) forward sum $\MPCmax_{t} = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree) \sum_{n=0}^{T-t}\left(\WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)\right)^{n}$; (iii) infinite-horizon $\MPCmax = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)$.

The solution method described above does not guarantee that approximated
consumption will respect this constraint between gridpoints, and a failure to
respect the constraint can occasionally cause computational problems in solving
or simulating the model. Here, we describe a method for constructing an
approximation that always satisfies the constraint.

Defining $\mNrmCusp$ as the _cusp_ point where the two upper bounds intersect
(where $\mNrmCuspEx\equiv\mNrmCusp-\mNrmMin$):

```{math}
:label: eq:mNrmCusp
\begin{array}{rclcll}
\bigl(\mNrmCuspEx + \hNrmEx\bigr)\,\MPCmin &= & \MPCmax\,\mNrmCuspEx & & \\
\mNrmCuspEx &= & \dfrac{\MPCmin\,\hNrmEx}{\MPCmax-\MPCmin} & & \\
\mNrmCusp &= & -\hNrmPes + \dfrac{\MPCmin\,\bigl(\hNrmOpt-\hNrmPes\bigr)}{\MPCmax-\MPCmin},
\end{array}
```

this intersection occurs in the feasible region (i.e., $\mNrmCusp > \mNrmMin$) since $\MPCmax > \MPCmin$ under the stated conditions, ensuring $\mNrmCuspEx > 0$. We want to construct a consumption function for
$\mNrm \in (\mNrmMin, \mNrmCusp]$ that respects the tighter upper
bound:

$$
\begin{array}{rcl}
\mNrmEx \MPCmin < & \cFuncReal(\mNrmMin+\mNrmEx) & < \MPCmax \mNrmEx \\
0 < & \cFuncReal(\mNrmMin+\mNrmEx) - \mNrmEx \MPCmin & < \mNrmEx(\MPCmax- \MPCmin) \\
0 < & \left(\frac{\cFuncReal(\mNrmMin+\mNrmEx) - \mNrmEx \MPCmin}{\mNrmEx(\MPCmax- \MPCmin)}\right) & < 1.
\end{array}
$$

Again defining $\logmNrmEx =\log \mNrmEx$, the object in the middle of
the inequality is

```{math}
:label: eq:modRteLoTightUpBd
\modRteLoTightUpBd(\logmNrmEx) \equiv  \frac{\cFuncReal(\mNrmMin+e^{\logmNrmEx})e^{-\logmNrmEx}-\MPCmin}{\MPCmax-\MPCmin}.
```

As $\mNrm$ approaches $-\mNrmMin$, $\modRteLoTightUpBd(\logmNrmEx)$
converges to $0$, while as $\mNrm$ approaches $+\infty$,
$\modRteLoTightUpBd(\logmNrmEx)$ approaches $1$.

As before, we can derive an approximated consumption function; call it
$\cFuncLoTightUpBd$. This function will clearly do a better job approximating
the consumption function for low values of $\mNrm$ while the previous
approximation will perform better for high values of $\mNrm$.

For middling values of $\mNrm$ it is not clear which of these functions will
perform better. However, an alternative is available which performs well. Define
the highest gridpoint below $\mNrmCusp$ as $\mNrmLoTightUpBd$ and the lowest
gridpoint above $\mNrmCusp$ as $\mNrmHiTightUpBd$. Then there will be a unique
interpolating polynomial that matches the level and slope of the consumption
function at these two points. Call this function $\cFuncMidTightUpBd(\mNrm)$.

Using indicator functions that are zero everywhere except for specified
intervals,

$$
\begin{aligned}
\indLo(\mNrm)  &= 1 \text{~if $\mNrm \leq \mNrmLoTightUpBd \phantom{< \mNrm < \mNrmHiTightUpBd \leq \mNrm}$} \\
\indMid(\mNrm) &= 1 \text{~if $\phantom{\mNrm \leq}~ \mNrmLoTightUpBd < \mNrm < \mNrmHiTightUpBd \phantom{\leq \mNrm}$} \\
\indHi(\mNrm)  &= 1 \text{~if $\phantom{\mNrm \leq ~\mNrmLoTightUpBd < \mNrm <} \mNrmHiTightUpBd \leq \mNrm$}
\end{aligned}
$$

we can define a well-behaved approximating consumption function

$$
\cFuncApprox = \indLo \cFuncLoTightUpBd + \indMid \cFuncMidTightUpBd+\indHi \cFuncHiTightUpBd.
$$

This just says that, for each interval, we use the approximation that is most
appropriate. The function is continuous and once-differentiable everywhere, and
is therefore well behaved for computational purposes.

We now construct an upper-bound value function implied for a consumer whose
spending behavior is consistent with the refined upper-bound consumption rule.

For $\mNrm \geq \mNrmCusp$, this consumption rule is the same as
before, so the constructed upper-bound value function is also the same. However,
for values $\mNrm < \mNrmCusp$ matters are slightly more complicated.

Start with the fact that at the cusp point,

$$
\begin{aligned}
\vFuncOpt(\mNrmCusp) &= \uFunc(\cFuncOpt(\mNrmCusp))\PDVCoverc \\
&=  \uFunc(\mNrmCuspEx \MPCmax)\PDVCoverc .
\end{aligned}
$$

But for _all_ $\mNrm$,

$$
\vFuncOpt(\mNrm) = \uFunc(\cFuncOpt(\mNrm))+ \wFuncCont(\mNrm-\cFuncOpt(\mNrm)),
$$

and we assume that for the consumer below the cusp point consumption is given by
$\MPCmax \mNrmEx$ so for $\mNrm< \mNrmCusp$

$$
\vFuncOpt(\mNrm) = \uFunc( \MPCmax \mNrmEx)+ \wFuncCont((1-\MPCmax)\mNrmEx),
$$

which is easy to compute because
$\wFuncCont(\aNrm_{t}) = \DiscFac \vFuncOpt_{t+1}(\aNrm_{t}\RNrmByG_{t+1}+1)$ where
$\vFuncOpt$ is as defined above because a consumer who ends the current
period with assets exceeding the lower bound will not expect to be constrained
next period. (Recall again that we are merely constructing an object that is
guaranteed to be an _upper bound_ for the value that the 'realist' consumer will
experience.) At the gridpoints defined by the solution of the consumption
problem can then construct

$$
\vInvOpt(\mNrm) = ((1-\CRRA)\vFuncOpt(\mNrm))^{1/(1-\CRRA)}
$$

which yields the appropriate vector for constructing $\logitValModRteApprox$ and
$\valModRteApprox$. The rest of the procedure is analogous to that performed for the
consumption rule and is thus omitted for brevity.

```{figure} ../images/IntExpFOCInvPesReaOptNeed45Plot
:label: fig:IntExpFOCInvPesReaOptNeed45
:alt: A diagram showing the true consumption function bounded above by both the optimist's consumption rule and a tighter linear bound originating from the natural borrowing constraint.
:align: center
:width: 80%

A Tighter Upper Bound
```

## Hermite Interpolation

The numerical accuracy of the method of moderation depends critically on the quality of function approximation between gridpoints {cite:p}`Santos2000`. Our bracketing approach complements work that bounds numerical errors in dynamic economic models {cite:p}`JuddMaliarMaliar2017`. Although linear interpolation that matches the level of $\cFuncReal$ at the gridpoints is simple, Hermite interpolation {cite:p}`Fritsch1980,FritschButland1984,Hyman1983` offers a considerable advantage. By matching both the level and the derivative of the $\cFuncReal$ function at the gridpoints, the consumption rule derived from such interpolation numerically satisfies the Euler equation at each gridpoint for which the problem has been solved {cite:p}`BenvenisteScheinkman1979,MilgromSegal2002`.

The theoretical foundation for this approach rests on the moderation ratio $\modRte$. This ratio captures how close the realist's consumption is to the optimist's behavior relative to the gap between optimist and pessimist. Since its log-gap argument $\logmNrmEx$ moves with cash-on-hand relative to human wealth, the derivative measures how quickly the realist approaches the optimist as available resources increase:

```{math}
:label: eq:modRteMu
\frac{\partial \modRte}{\partial \logmNrmEx} = \frac{\mNrmEx (\partial \cFuncReal/\partial \mNrm - \MPCmin)}{\MPCmin \hNrmEx}.
```

For numerical stability and interpretation on an unbounded scale, we apply the transformation defined in {ref}`eq:chi` to the moderation ratio. The derivative of this transformation is:

```{math}
:label: eq:logitModRteMu
\frac{\partial \logitModRte}{\partial \logmNrmEx} = \frac{\partial \modRte / \partial \logmNrmEx}{\modRte(1 - \modRte)}.
```

This expression provides the slope data for cubic Hermite interpolation.[^hermite-slopes] Differentiating {ref}`eq:cFuncHi` yields a moderation form for the MPC:[^mpc-derivation]

```{math}
:label: eq:MPCModeration
\frac{\partial \cFuncReal}{\partial \mNrm} = (1-\MPCmod)\,\MPCmin + \MPCmod\,\MPCmax
```

where

```{math}
:label: eq:MPCModerationWeight
\MPCmod = \frac{\MPCmin}{\MPCmax-\MPCmin} \cdot \frac{\hNrmEx}{\mNrmEx} \cdot \partial \modRte / \partial \logmNrmEx.
```

The weight $\MPCmod \in [0,1]$ at gridpoints by construction, since equation {eq}`eq:MPCModeration` can be rewritten as $\MPCmod = (\partial \cFuncReal/\partial \mNrm - \MPCmin)/(\MPCmax - \MPCmin)$ and theory guarantees $\MPCmin \leq \partial \cFuncReal/\partial \mNrm \leq \MPCmax$ at points where the Euler equation is solved. Between gridpoints, the interpolated moderation ratio derivative $\modRteMu$ should preserve this property when the grid is sufficiently refined. The weight reflects precautionary intensity: it increases when market resources are low relative to human wealth ($\hNrmEx/\mNrmEx$ large) and when the moderation ratio responds sharply to changes in log excess resources ($\modRteMu$ large). As $\mNrmEx \to \infty$, $\MPCmod \to 0$ and the MPC approaches the optimist's; as $\mNrmEx \to 0$, $\MPCmod \to 1$ and the MPC approaches the pessimist's.

[^mpc-derivation]:
    Differentiating {ref}`eq:cFuncHi` with respect to $\mNrm$ and applying the chain rule: $\partial \cFuncReal/\partial \mNrm = \MPCmin \left(1 + \hNrmEx/\mNrmEx \cdot \modRteMu\right)$. The moderation form follows by factoring: $\MPCmod(\MPCmax-\MPCmin) = \MPCmin \cdot \hNrmEx/\mNrmEx \cdot \modRteMu$. Note that $\modRteMu = \modRte(1-\modRte) \cdot \logitModRteMu$ from the chain rule, where $\modRte(1-\modRte) = \partial\modRte/\partial\logitModRte$ is the standard sigmoid derivative, but working directly with $\modRteMu$ provides superior numerical stability since the moderation ratio $\modRte \in [0,1]$ is naturally bounded.

We can apply analogous techniques to the value function. Under perfect foresight, consumption grows at a constant rate, making $\PDVCoverc$ constant. This implies that the inverse value function for the optimist has a constant slope with respect to cash-on-hand:

```{math}
:label: eq:vInvOptDeriv
\begin{aligned}
\vInvOptDeriv &= \PDVCoverc^{1/(1-\CRRA)} \MPCmin \\
&= \MPCmin^{-\CRRA/(1-\CRRA)}.
\end{aligned}
```

The result in {ref}`eq:vInvOptDeriv` has important implications for the structure of the value function.[^vInvOpt-linearity]

Consider the value analogue of the moderation ratio, which compares the realist's value to the optimist's. The derivative of this ratio with respect to the log-gap argument is:

```{math}
:label: eq:valModRteRealDerivmu
\frac{\partial \valModRteReal}{\partial \logmNrmEx} = \frac{\mNrmEx (\vInvRealDeriv - \vInvOptDeriv)}{\hNrmEx \vInvOptDeriv}
```

where $\vInvOptDeriv = \MPCmin^{-\CRRA/(1-\CRRA)}$ from {ref}`eq:vInvOptDeriv` and $\vInvRealDeriv$ is the derivative of the realist's inverse value function. Note that for the pessimist's inverse value function, the derivative equals $\vInvOptDeriv$ since both are linear perfect foresight solutions.

Applying the same transformation to the value-based moderation ratio converts the bounded ratio into an unconstrained slope:

```{math}
:label: eq:logitValModRteRealDerivmu
\frac{\partial \logitValModRteReal}{\partial \logmNrmEx} = \frac{\partial \valModRteReal / \partial \logmNrmEx}{\valModRteReal(1-\valModRteReal)}.
```

Since $\vInvReal$ and $\vFuncReal$ are functional inverses, their derivatives are linked by chain-rule relationships. The first derivative is:

```{math}
:label: eq:vInvRealDeriv
\vInvRealDeriv = \left( (1-\CRRA) \vFuncReal(\mNrm)\right)^{-1+1/(1-\CRRA)}  \vFuncRealDeriv(\mNrm).
```

The first- and second-derivative connections are:

```{math}
:label: eq:vFuncRealDerivatives
\begin{aligned}
\vFuncRealDeriv &= \uPrime(\vInvReal) \, \vInvRealDeriv \\
\vFuncRealDerivSecond &= \uDoublePrime(\vInvReal) \, (\vInvRealDeriv)^2 + \uPrime(\vInvReal) \, \vInvRealDerivSecond.
\end{aligned}
```

Moreover, if we use the double-derivative calculated in {ref}`eq:vFuncRealDerivatives` to produce a higher-order Hermite polynomial, our approximation will also match the marginal propensity to consume at the gridpoints.[^hermite-higher-order]

These results provide the theoretical foundation for constructing high-quality cubic Hermite interpolants that preserve both the economic structure and numerical accuracy of the model between gridpoints.

[^hermite-higher-order]:
    This would guarantee that the consumption function generated from the value function would match both the level of consumption and the marginal propensity to consume at the gridpoints, making the numerical differences between the newly constructed consumption function and the highly accurate one constructed earlier negligible within the grid.

[^hermite-slopes]:
    For cubic Hermite interpolation of the transformed moderation ratio, use node values from the transformation and node slopes $\logitModRteMu$. For improved shape preservation, a monotone cubic Hermite scheme {cite:p}`deBoor2001` can be used (such as the Fritsch-Carlson or Fritsch-Butland algorithms), where theoretical slopes serve as targets that may be adjusted to enforce monotonicity.

[^vInvOpt-linearity]:
    This confirms that $\vInvOpt$ is linear in $\mNrm$ and highlights the role of $\MPCmin$ in scaling marginal utility in the perfect-foresight benchmark. The linearity property simplifies both theoretical analysis and numerical implementation.

## Stochastic Rate of Return

Thus far we have assumed that the interest factor is constant at $\Rfree$.
Extending the previous derivations to allow for a perfectly forecastable
time-varying interest factor $\Risky$ would be trivial. Allowing for a
stochastic interest factor is less trivial.

The easiest case is where the interest factor is i.i.d.,

```{math}
:label: eq:distRisky
\log \Risky_{t+n} \sim \Nrml(r + \equityPrem - \std^{2}_{\risky}/2,\std^{2}_{\risky}) ~\forall~n>0
```

because in this case {cite:t}`Samuelson1969,Merton1969,Merton1971`
showed that for a consumer without labor income (or with perfectly forecastable
labor income) the consumption function is linear, with an MPC.[^crra-rate-risk]

[^crra-rate-risk]:
    The Merton-Samuelson rule implies linear consumption $\cFunc(\mNrm) = \MPC\,\mNrm$ with $\MPC$ from {eq}`eq:MPCExact`. For lognormal $\log \Risky \sim \Nrml(r+\equityPrem-\std_{\risky}^{2}/2,\std_{\risky}^{2})$, the MGF yields $\Ex[\Risky^{1-\CRRA}] = \exp((1-\CRRA)(r+\equityPrem) + (1-\CRRA)(1-2\CRRA)\std_{\risky}^{2}/2)$. This extends to our framework by substituting stochastic-return MPC for perfect foresight MPC. See {cite:t}`CRRA-RateRisk,BBZ2016SkewedWealth`.

```{math}
:label: eq:MPCExact
\MPC = 1- \left(\DiscFac  \Ex[\Risky_{t+1}^{1-\CRRA}]\right)^{1/\CRRA}
```

and in this case the previous analysis applies once we substitute this MPC for
the one that characterizes the perfect foresight problem without rate-of-return
risk. The more realistic case where the interest factor has some serial
correlation is more complex, and thus left for future work.

In principle, this refinement should be combined with the previous one; further
exposition of this combination is omitted here because no new insights spring
from the combination of the two techniques.

# Conclusion

The method proposed here is not universally applicable. For example, the method
cannot be used for problems for which upper and lower bounds to the 'true'
solution are not known. But many problems do have obvious upper and lower
bounds, and in those cases (as in the consumption example used in the paper),
the method may result in substantial improvements in accuracy and stability of
solutions.
