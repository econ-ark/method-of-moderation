---
# Page-specific frontmatter (inherits bibliography from myst.yml)
title: Appendix to "The Method of Moderation"
short_title: Appendix
abstract: |
  This appendix provides detailed mathematical derivations and technical results supporting the Method of Moderation. Topics include: value function transformations and their relationship to the inverse value function; explicit formulas for minimal and maximal marginal propensities to consume; cusp point calculations for tighter upper bounds; Hermite interpolation slope formulas and MPC derivations; patience conditions ensuring well-defined solutions; and extensions to stochastic returns with explicit formulas for consumption problems with an exogenous risky return.
keywords:
  - Dynamic Stochastic Optimization
  - Consumption-Saving Models
  - Numerical Methods
parts:
  jel_codes: C63; D81; E21
---



# Patience Conditions Details

Each patience condition from the main text controls a distinct way the problem could misbehave. The FVAC $0<\DiscFac\PermGroFac^{1-\CRRA}\Ex[\permShk^{1-\CRRA}]<1$ guarantees that even autarky, saving nothing and consuming income as it arrives, delivers finite expected discounted utility, so the consumer has a reason to value resources at all. The AIC $\AbsPatFac<1$ rules out indefinite deferral of consumption: under certainty the marginal utility of consuming now exceeds the discounted marginal utility of consuming later. Two further conditions bound wealth from opposite directions. The RIC $\AbsPatFac/\Rfree<1$ holds asset growth below the patience-adjusted discount rate, so wealth cannot explode; the GIC $\AbsPatFac/\PermGroFac<1$ holds consumption growth below permanent-income growth, which is what pins down a finite target wealth ratio. Finally, the FHWC $\PermGroFac/\Rfree<1$ keeps the present value of future labor income finite. Where these conditions fail, behavior changes qualitatively: when all hold, the consumer runs a buffer stock around a target wealth ratio; when the AIC fails, consumption grows without bound; when the GIC fails but the RIC still holds, wealth grows without bound {cite:p}`Carroll1997,SolvingMicroDSOPs,CarrollShanker2024`.

# Human Wealth Formulas

The optimist's human wealth (assuming $\tranShk_{t+n}=1~\forall~n>0$) can be computed three ways: backward recursion $\hNrmOpt_{T} = 0$, $\hNrmOpt_{t} = (\PermGroFac/\Rfree)(1 + \hNrmOpt_{t+1})$; forward sum $\hNrmOpt_{t} = \sum_{n=1}^{T-t}(\PermGroFac/\Rfree)^{n}$; or infinite-horizon $\hNrmOpt = \PermGroFac/(\Rfree-\PermGroFac)$ when $\Rfree>\PermGroFac$. With $\PermGroFac=1$, $\hNrmOpt = 1/(\Rfree-1)$.

The pessimist's human wealth (assuming $\tranShk_{t+n}=\tranShkMin~\forall~n>0$) follows similarly: backward recursion $\hNrmPes_{T}=0$, $\hNrmPes_{t}=(\PermGroFac/\Rfree)(\tranShkMin + \hNrmPes_{t+1})$; forward sum $\hNrmPes_{t}=\tranShkMin\sum_{n=1}^{T-t}(\PermGroFac/\Rfree)^{n}$; or infinite-horizon $\hNrmPes=\tranShkMin\PermGroFac/(\Rfree-\PermGroFac)$. When $\tranShkMin=0$ (unemployment), $\hNrmPes=0$.

# Marginal Propensity to Consume Formulas

The minimal MPC (perfect foresight consumer with horizon $T-t$) has three forms {cite:p}`Carroll2001MPCBound`: backward recursion $\MPCmin_{t}=\MPCmin_{t+1}/(\MPCmin_{t+1}+\AbsPatFac/\Rfree)$ with $\MPCmin_T=1$; forward sum $\MPCmin_{t}=(\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n})^{-1}$; or infinite-horizon $\MPCmin=1-\AbsPatFac/\Rfree = 1-(\Rfree \DiscFac)^{1/\CRRA}/\Rfree$.

The maximal MPC {cite:p}`CarrollToche2009` satisfies backward recursion $\MPCmax_{t}=\MPCmax_{t+1}/(\MPCmax_{t+1}+\WorstProb^{1/\CRRA}\AbsPatFac/\Rfree)$ with $\MPCmax_T=1$; forward sum $\MPCmax_{t}=(\sum_{n=0}^{T-t}(\WorstProb^{1/\CRRA}\AbsPatFac/\Rfree)^{n})^{-1}$; or infinite-horizon $\MPCmax = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)$.

# A Tighter Upper Bound

The method in the main text does not guarantee that the approximation respects $\cFuncReal(\mNrm) < \MPCmax \mNrmEx$, where $\MPCmax$ is the MPC at the natural borrowing constraint; near the constraint the optimist's bound is loose because it is calibrated to the low MPC that prevails at high wealth. {cite:t}`CarrollShanker2024` derives the maximal MPC $\MPCmax = 1 - \WorstProb^{1/\CRRA}(\AbsPatFac/\Rfree)$, where $\WorstProb$ is the unemployment probability of {cite:t}`CarrollToche2009`, extending the limiting-MPC formulas of {cite:t}`MaToda2021SavingRateRich`. Strict concavity implies $\cFuncReal(\mNrm) < \MPCmax \mNrmEx$ for low wealth, while the optimist's bound $\cFuncReal(\mNrm) < (\mNrmEx+\hNrmEx)\MPCmin$ is tighter for high wealth.

:::{figure} #fig:mom-consumption-function
:label: fig:IntExpFOCInvPesReaOptNeed45
:alt: A diagram showing the true consumption function bounded above by both the optimist's consumption rule and a tighter linear bound originating from the natural borrowing constraint.

A Tighter Upper Bound
:::

As {ref}`fig:IntExpFOCInvPesReaOptNeed45` shows, the two upper bounds intersect at the cusp point $\mNrmCusp$ where

```{math}
:label: eq:mNrmCuspFull
\begin{array}{rclcll}
\bigl(\mNrmCuspEx + \hNrmEx\bigr)\,\MPCmin &= & \MPCmax\,\mNrmCuspEx & & \\
\mNrmCuspEx &= & \dfrac{\MPCmin\,\hNrmEx}{\MPCmax-\MPCmin} & & \\
\mNrmCusp &= & -\hNrmPes + \dfrac{\MPCmin\,\bigl(\hNrmOpt-\hNrmPes\bigr)}{\MPCmax-\MPCmin},
\end{array}
```

where $\mNrmCuspEx\equiv\mNrmCusp-\mNrmMin > 0$ since $\MPCmax > \MPCmin$. For $\mNrm \in (\mNrmMin, \mNrmCusp]$, the tighter upper bound yields

$$
\begin{array}{rcl}
\mNrmEx \MPCmin < & \cFuncReal(\mNrmMin+\mNrmEx) & < \MPCmax \mNrmEx \\
0 < & \cFuncReal(\mNrmMin+\mNrmEx) - \mNrmEx \MPCmin & < \mNrmEx(\MPCmax- \MPCmin) \\
0 < & \left(\frac{\cFuncReal(\mNrmMin+\mNrmEx) - \mNrmEx \MPCmin}{\mNrmEx(\MPCmax- \MPCmin)}\right) & < 1.
\end{array}
$$

This motivates the low-resource moderation ratio, defined for $\mNrm \in (\mNrmMin,\mNrmCusp]$ as

```{math}
:label: eq:modRteLoTightUpBd
\modRteLoTightUpBd(\logmNrmEx) = \frac{\cFuncReal(\mNrmMin+e^{\logmNrmEx})e^{-\logmNrmEx}-\MPCmin}{\MPCmax-\MPCmin}.
```

Since $e^{-\logmNrmEx} = 1/\mNrmEx$, the right-hand side equals $(\cFuncReal/\mNrmEx - \MPCmin)/(\MPCmax - \MPCmin)$, which lies in $(0,1)$ for $\mNrm \in (\mNrmMin,\mNrmCusp]$: the lower bound is the minimal MPC $\MPCmin$ and the upper bound is the maximal MPC $\MPCmax$, with strict inequality at the upper end following from $\cFuncReal < \MPCmax\,\mNrmEx$. Applying the logit transformation and interpolating as before yields consumption satisfying both upper bounds throughout. For computational robustness we combine the pieces into a three-part approximation: the tighter bound below the cusp, the optimist's bound above, and a Hermite segment (below) bridging the cusp, where the two bounds meet at equal levels but different slopes. Because the Hermite segment is matched to the level and the slope of the adjacent piece at each of its endpoints, the combined consumption function is continuous and differentiable and respects all theoretical constraints.

# Value Function Derivation

Under perfect foresight, consumption grows at constant rate equal to the absolute patience factor $\AbsPatFac$: $\cLvl_{t+n}=\cLvl_{t}\AbsPatFac^{n}$. The present discounted value of consumption, discounting the stream at the return $\Rfree$, satisfies $\PDV_{t}^{T}(\cLvl)=\sum_{n=0}^{T-t}\Rfree^{-n}\cLvl_{t}\AbsPatFac^{n}=\cLvl_{t}\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n}$. Dividing by consumption yields the PDV-to-consumption ratio $\PDVCoverc_{t}^{T}=\PDV_{t}^{T}(\cLvl)/\cLvl_{t}=\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n}=\MPCmin_{t}^{-1}$, which is unchanged for normalized variables. Defining $\PDVCoverc \equiv \lim_{T\to\infty} \PDVCoverc_{t}^{T}$, this yields the key identity $\PDVCoverc = \MPCmin^{-1}$, connecting the infinite-horizon PDV-to-consumption ratio to the minimal MPC.

The optimist's value function satisfies

$$
\begin{aligned}
\vFuncOpt_{T-1}(\mNrm_{T-1}) &\equiv  \uFunc(\cNrm_{T-1})+\DiscFac \uFunc(\cNrm_{T}) \\
&= \uFunc(\cNrm_{T-1})\left(1+\DiscFac \AbsPatFac^{1-\CRRA}\right) \\
&= \uFunc(\cNrm_{T-1})\left(1+\AbsPatFac/\Rfree\right) \\
&= \uFunc(\cNrm_{T-1})\PDVCoverc_{T-1}^{T}
\end{aligned}
$$

The infinite horizon expression becomes

```{math}
:label: eq:vFuncPF
\begin{aligned}
\vFuncOpt(\mNrm) &= \uFunc(\cFuncOpt(\mNrm))\PDVCoverc \\
&= \uFunc(\cFuncOpt(\mNrm))\MPCmin^{-1} \\
&= \uFunc((\mNrmEx+\hNrmEx)\MPCmin) \MPCmin^{-1} \\
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

The transformation $\vInv \equiv \left((1-\CRRA)\vFunc\right)^{1/(1-\CRRA)}$ is the inverse utility $\uFunc^{-1}$; for log utility ($\CRRA=1$) it becomes $\vInv = \exp(\vFunc)$, the $\CRRA\to1$ limit, so the construction carries over to log utility unchanged.

The pessimist's inverse value follows by the same steps, with $\cFuncPes = \mNrmEx\MPCmin$:

$$
\vInvPes = \cFuncPes\,\PDVCoverc^{1/(1-\CRRA)} = \mNrmEx\,\MPCmin^{-\CRRA/(1-\CRRA)},
$$

so that $\vInvOpt - \vInvPes = \hNrmEx\,\MPCmin\,\PDVCoverc^{1/(1-\CRRA)}$, the denominator that normalizes the value moderation ratio below.

For the realist's problem, we define $\vInvReal = \left((1-\CRRA)\vFuncReal(\mNrm)\right)^{1/(1-\CRRA)}$. At each $\mNrm$ the values are ordered $\vFuncPes < \vFuncReal < \vFuncOpt$: the realist's true income process stochastically dominates the pessimist's worst-case income and is dominated, for a risk-averse agent, by the optimist's certain expected income. Because the inverse-value transform is monotonic, the same ordering holds for $\vInvPes < \vInvReal < \vInvOpt$, and we define

```{math}
:label: eq:valModRteReal
\valModRteReal(\logmNrmEx) = \left(\frac{\vInvReal(\mNrmMin+e^{\logmNrmEx})-\vInvPes(\mNrmMin+e^{\logmNrmEx})}{\hNrmEx \MPCmin \,\PDVCoverc^{1/(1-\CRRA)}}\right)
```

and the logit-transformed counterpart:

```{math}
:label: eq:ChiUpper
\begin{aligned}
\logitValModRteReal(\logmNrmEx) &= \log \left(\frac{\valModRteReal(\logmNrmEx)}{1-\valModRteReal(\logmNrmEx)}\right) \\
&= \log(\valModRteReal(\logmNrmEx)) - \log(1-\valModRteReal(\logmNrmEx))
\end{aligned}
```

Inverting these approximations yields

```{math}
:label: eq:vInvHi
\vInvReal = \vInvPes+\overbrace{\left(\frac{1}{1+\exp(-\logitValModRteReal)}\right)}^{=\valModRteReal} \hNrmEx \MPCmin \,\PDVCoverc^{1/(1-\CRRA) }
```

from which the value function approximation is $\vFuncReal = \uFunc(\vInvReal)$.

# Hermite Interpolation

The numerical accuracy of the method of moderation depends critically on the quality of function approximation between gridpoints {cite:p}`Santos2000`. Our bracketing approach complements work that bounds numerical errors in dynamic economic models {cite:p}`JuddMaliarMaliar2017`. Although linear interpolation that matches the level of $\cFuncReal$ at the gridpoints is simple, Hermite interpolation {cite:p}`Fritsch1980` offers a considerable advantage.

The moderation ratio derivative measures how quickly the realist approaches the optimist as resources increase.  Differentiating {eq}`eq:modRte` with respect to $\logmNrmEx$ we obtain

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

{cite:t}`CarrollShanker2024` guarantees $\MPCmin \leq \partial \cFuncReal/\partial \mNrm \leq \MPCmax$ at gridpoints where the Euler equation holds, so $\MPCmod \in [0,1]$ and the expression above is indeed a convex combination of $\MPCmin$ and $\MPCmax$. At very high wealth, $\MPCmod \to 0$ and the MPC approaches $\MPCmin$; near the borrowing constraint, $\MPCmod \to 1$ and the MPC approaches $\MPCmax$.

For Hermite interpolation, compute $\modRteMu$ at gridpoints, then derive $\logitModRteMu = \modRteMu/[\modRte(1-\modRte)]$ for slope data. By matching both the level and the derivative of $\cFuncReal$ at the gridpoints, where the derivative is obtained from the envelope condition {cite:p}`BenvenisteScheinkman1979,MilgromSegal2002` together with the EGM Euler equation, the interpolated consumption rule satisfies the Euler equation exactly at each solved gridpoint. These techniques extend naturally to the value function approximation.

For monotone cubic Hermite schemes {cite:p}`Fritsch1980,FritschButland1984,deBoor2001`, theoretical slopes may be adjusted to enforce monotonicity {cite:p}`Hyman1983`. The Fritsch-Carlson algorithm modifies slopes at local extrema, while Fritsch-Butland uses harmonic mean weighting. Both preserve the shape-preserving property essential for consumption functions that must be strictly increasing.

(stochastic-returns-mgf-derivation)=
# Stochastic Rate of Return

For i.i.d. returns with $\log \Risky \sim \Nrml(r + \equityPrem - \std^{2}_{\risky}/2,\std^{2}_{\risky})$,[^lognormal-params] {cite:t}`Samuelson1969,Merton1969,Merton1971` showed that for a consumer without labor income (or with perfectly forecastable labor income) the consumption function is linear, with an MPC $= 1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$, which is positive under the stochastic return impatience condition $\DiscFac\Ex[\Risky^{1-\CRRA}]<1$ (the i.i.d.-return analogue of the RIC $\AbsPatFac/\Rfree<1$). See {cite:t}`CRRA-RateRisk,BBZ2016SkewedWealth,CKW2021Aggregation` for extensions. The pessimist and the optimist face certain income but the same stochastic return, so the Merton-Samuelson result applies to both and their consumption functions remain linear. The realist faces both labor income and return risk, and the moderation ratio captures their combined precautionary response. In this case the previous analysis applies once we substitute this MPC for the one that characterizes the perfect-foresight problem without rate-of-return risk. As {ref}`fig:StochasticBounds` shows, consumption remains bounded between the pessimist and the optimist, each of which (for $\CRRA>1$) consumes slightly less in the face of return uncertainty; for $\CRRA<1$ the effect reverses.

[^lognormal-params]:
    Here $r$ is the log risk-free rate and $\equityPrem$ is the equity premium (the expected excess log return). This parametrization ensures $\Ex[\Risky] = \exp(r+\equityPrem)$, so that increasing $\std^{2}_{\risky}$ constitutes a mean-preserving spread of the level of the return.

:::{figure} #fig:stochastic-bounds
:label: fig:StochasticBounds
:alt: Comparison of consumption bounds under deterministic and stochastic rates of return.

Effect of Return Uncertainty on Consumption Bounds
:::

The fact that a linear consumption function with an MPC $= 1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$ satisfies the Euler equation with i.i.d. returns and no labor income can be derived by the method of undetermined coefficients.  In particular, assume that $\cFuncOpt(\mNrm) = \mNrm\MPCmin$, with a time-independent MPC $\MPCmin$ to be determined.  Substituting this into the Euler equation, we have

```{math}
:label: eq:stochReturnsEulerEqn
\begin{aligned}
1 &= \DiscFac \Ex_t\left[\Risky_{t+1} \left(\frac{\cNrm_{t+1}}{\cNrm_t}\right)^{-\CRRA}\right]\\
&= \DiscFac \Ex_t\left[\Risky_{t+1} \left(\frac{\mNrm_{t+1}}{\mNrm_t}\right)^{-\CRRA}\right]
\end{aligned}
```

where the second equality uses the assumed form of the consumption function.  Since there is no labor income, $\mNrm_{t+1} = \Risky_{t+1}(\mNrm_t - \cNrm_t)$.  Substituting this into the above we obtain

```{math}
:label: eq:stochReturnsEulerEqnContd
1 = \DiscFac \Ex_t\left[\Risky_{t+1} \left(\Risky_{t+1}(1-\MPCmin)\right)^{-\CRRA}\right]
```
Solving for $\MPCmin$ and recalling that returns are i.i.d. gives $\MPCmin=1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$.

In the particular case of lognormal returns, the MPC can be written in closed form.  The moment generating function (MGF) of the normal variable $X = \log \Risky$ provides the key formula. For $X \sim \Nrml(\mu, \sigma^2)$, the MGF is $\Ex[e^{sX}] = \exp(\mu s + \sigma^2 s^2/2)$. Setting $s = 1-\CRRA$ and $\mu = r + \equityPrem - \std_{\risky}^2/2$ yields[^lognormal-returns-intuition]

$$
\Ex[\Risky^{1-\CRRA}] = \exp\left((1-\CRRA)\left(r+\equityPrem - \frac{\std_{\risky}^2}{2}\right) + \frac{(1-\CRRA)^2\std_{\risky}^2}{2}\right).
$$

Simplifying the variance terms: $(1-\CRRA)^2\std_{\risky}^2/2 - (1-\CRRA)\std_{\risky}^2/2 = (1-\CRRA)[(1-\CRRA)-1]\std_{\risky}^2/2 = -\CRRA(1-\CRRA)\std_{\risky}^2/2$, giving the final form

$$
\Ex[\Risky^{1-\CRRA}] = \exp\left((1-\CRRA)\left(r+\equityPrem - \CRRA\std_{\risky}^2/2\right)\right).
$$

[^lognormal-returns-intuition]:
    Here we can interpret $\equityPrem$ as the risk premium, that is, the additional average return from holding a risky asset compared to the risk-free rate $r$.  Adjusting the average log return by the asset volatility ensures that increasing $\std_{\risky}^2$ constitutes a mean-preserving spread of the level of return.
