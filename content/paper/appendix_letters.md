---
# Page-specific frontmatter (inherits bibliography from myst.yml)
title: Appendix to "The Method of Moderation"
short_title: Appendix
abstract: |
  This appendix provides detailed mathematical derivations and technical results supporting the Method of Moderation. Topics include: value function transformations and their relationship to the inverse value function; explicit formulas for minimal and maximal marginal propensities to consume; cusp point calculations for tighter upper bounds; Hermite interpolation slope formulas and MPC derivations; patience conditions ensuring well-defined solutions; and extensions to stochastic returns with explicit formulas for portfolio problems.
keywords:
  - Dynamic Stochastic Optimization
  - Numerical Methods
parts:
  jel_codes: C63; D81; E21
exports:
  - format: tex+pdf
    template: arxiv_two_column
    output: ../exports/appendix_letters.pdf
---



# Appendix: Mathematical Details

## Patience Conditions Details

The patience conditions listed in the main text have clear economic interpretations. The FVAC $0<\DiscFac\PermGroFac^{1-\CRRA}\Ex[\permShk^{1-\CRRA}]<1$ ensures autarky (consuming the permanent component of income each period) has finite expected discounted utility, guaranteeing the consumer values resources. The AIC $\AbsPatFac<1$ prevents indefinite consumption deferral by ensuring the marginal utility of current consumption exceeds the discounted marginal utility of future consumption under certainty. The RIC $\AbsPatFac/\Rfree<1$ ensures asset growth is slower than the patience-adjusted discount rate, preventing unbounded wealth accumulation. The GIC $\AbsPatFac/\PermGroFac<1$ ensures consumption grows slower than permanent income, establishing a target wealth ratio. The FHWC $\PermGroFac/\Rfree<1$ ensures the present value of future labor income is finite. Together, these conditions partition parameter space into regions with qualitatively different behavior: buffer-stock saving with a target wealth ratio (all conditions hold), perpetual borrowing (AIC fails), or unbounded wealth growth (GIC fails but RIC holds) {cite:p}`Carroll1997,SolvingMicroDSOPs,CarrollShanker2024`.

## Human Wealth Formulas

The optimist's human wealth (assuming $\tranShk_{t+n}=1~\forall~n>0$) can be computed three ways: backward recursion $\hNrmOpt_{T} = 0$, $\hNrmOpt_{t} = (\PermGroFac/\Rfree)(1 + \hNrmOpt_{t+1})$; forward sum $\hNrmOpt_{t} = \sum_{n=1}^{T-t}(\PermGroFac/\Rfree)^{n}$; or infinite-horizon $\hNrmOpt = \PermGroFac/(\Rfree-\PermGroFac)$ when $\Rfree>\PermGroFac$. With $\PermGroFac=1$, $\hNrmOpt = 1/(\Rfree-1)$.

The pessimist's human wealth (assuming $\tranShk_{t+n}=\tranShkMin~\forall~n>0$) follows similarly: backward recursion $\hNrmPes_{T}=0$, $\hNrmPes_{t}=(\PermGroFac/\Rfree)(\tranShkMin + \hNrmPes_{t+1})$; forward sum $\hNrmPes_{t}=\tranShkMin\sum_{n=1}^{T-t}(\PermGroFac/\Rfree)^{n}$; or infinite-horizon $\hNrmPes=\tranShkMin\PermGroFac/(\Rfree-\PermGroFac)$. When $\tranShkMin=0$ (unemployment), $\hNrmPes=0$.

## Marginal Propensity to Consume Formulas

The minimal MPC (perfect foresight consumer with horizon $T-t$) has three forms {cite:p}`Carroll2001MPCBound`: backward recursion $\MPCmin_{t}=\MPCmin_{t+1}/(\MPCmin_{t+1}+\AbsPatFac/\Rfree)$ with $\MPCmin_T=1$; forward sum $\MPCmin_{t}=(\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n})^{-1}$; or infinite-horizon $\MPCmin=1-\AbsPatFac/\Rfree = 1-(\Rfree \DiscFac)^{1/\CRRA}/\Rfree$.

The maximal MPC {cite:p}`CarrollToche2009` satisfies backward recursion $\MPCmax_{t} = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree) (1 + \MPCmax_{t+1})$ with $\MPCmax_T = 1$; forward sum $\MPCmax_{t} = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree) \sum_{n=0}^{T-t}\left(\WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)\right)^{n}$; or infinite-horizon $\MPCmax = 1 - \WorstProb^{1/\CRRA} (\AbsPatFac/\Rfree)$.

## Asymptotic Linearity and Extrapolation

Under the GIC, the logit-transformed moderation ratio $\logitModRte(\logmNrmEx)$ is asymptotically linear with slope $\asympSlope = \lim_{\logmNrmEx \to +\infty} \frac{\partial \logitModRte}{\partial \logmNrmEx} \geq 0$ as $\logmNrmEx \to +\infty$. This slope may equal zero in theory, but is strictly positive on finite grids. For practical implementation, extrapolate $\logitModRte$ linearly using the positive boundary slope computed at the highest gridpoint. This linear extrapolation preserves $\modRte\in(0,1)$ and hence $\cFuncPes < \cFuncApprox < \cFuncOpt$ throughout the extrapolation domain, even if the theoretical limiting slope vanishes. The asymptotic linearity property is crucial because it allows the method to accurately represent the consumption function far beyond the range of gridpoints where the Euler equation was solved, without ever violating the theoretical bounds.

## Cusp Point Calculation

The two upper bounds intersect at the cusp point $\mNrmCusp$ where

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

This motivates the definition of the low-resource moderation ratio as in {eq} `eq:modRteLoTightUpBd`.

## Value Function Derivation

Under perfect foresight, consumption grows at constant rate equal to the absolute patience factor $\AbsPatFac$: $\cLvl_{t+n}=\cLvl_{t}\AbsPatFac^{n}$. The present discounted value of consumption satisfies $\PDV_{t}^{T}(\cLvl)=\sum_{n=0}^{T-t}\DiscFac^{n}\cLvl_{t}\AbsPatFac^{n}=\cLvl_{t}\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n}$, where we use $\DiscFac\AbsPatFac^{1-\CRRA}=\AbsPatFac/\Rfree$. Dividing by consumption yields the PDV-to-consumption ratio $\PDVCoverc_{t}^{T}=\PDV_{t}^{T}(\cLvl)/\cLvl_{t}=\sum_{n=0}^{T-t}(\AbsPatFac/\Rfree)^{n}=\MPCmin_{t}^{-1}$, which is unchanged for normalized variables. This yields the key identity $\PDVCoverc_{t}^{T} \to \PDVCoverc = \MPCmin^{-1}$ in the infinite-horizon limit, connecting the PDV-to-consumption ratio to the minimal MPC.

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

For the realist's problem, we define $\vInvReal = \left((1-\CRRA)\vFuncReal(\mNrm)\right)^{1/(1-\CRRA)}$. Using the bounds $\vInvPes < \vInvReal < \vInvOpt$, we define

```{math}
:label: eq:valModRteReal
\valModRteReal(\logmNrmEx) = \left(\frac{\vInvReal(\mNrmMin+e^{\logmNrmEx})-\vInvPes(\mNrmMin+e^{\logmNrmEx})}{\hNrmEx \MPCmin \,\PDVCoverc^{1/(1-\CRRA)}}\right)
```

and:

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

## I.I.D. Stochastic Returns: MPC Derivation

The fact that a linear consumption function with an MPC $= 1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$ satisfies the Euler equation with i.i.d. returns and no labor income can be derived by the method of undetermined coefficients.  In particular, assume that $\cFuncOpt(\mNrm) = \mNrm\MPCmin$, with a time-independent MPC $\MPCmin$ to be determined.  Substituting this into the Euler equation, we have

```{math}
:label: eq:stochReturnsEulerEqn
\begin{aligned}
1 &= \DiscFac \Ex_t\Risky_{t+1} \left(\frac{\cNrm_{t+1}}{\cNrm_t}\right)^{-\CRRA}\\
&= \DiscFac \Ex_t\Risky_{t+1} \left(\frac{\mNrm_{t+1}}{\mNrm_t}\right)^{-\CRRA}
\end{aligned}
```

where the second equality uses the assumed form of the consumption function.  Since there is no labor income, $\mNrm_{t+1} = \Risky_{t+1}(\mNrm_t - \cNrm_t)$.  Substituting this into the above we obtain

```{math}
:label: eq:stochReturnsEulerEqnContd
\begin{aligned}
1 &= \DiscFac \Ex_t\Risky_{t+1} \left(\frac{\Risky_{t+1}(\mNrm_t - \cNrm_t)}{\mNrm_t}\right)^{-\CRRA} \\
&= \DiscFac \Ex_t\Risky_{t+1} \left(\Risky_{t+1}(1-\MPCmin)\right)^{-\CRRA}
\end{aligned}
```
Solving for $\MPCmin$ and recalling that returns are i.i.d. gives $\MPCmin=1- (\DiscFac \Ex[\Risky^{1-\CRRA}])^{1/\CRRA}$.

In the particular case of lognormal returns, the MPC can be written in closed form.  The moment generating function (MGF) for lognormal returns provides the key formula. For $\log \Risky \sim \Nrml(\mu, \sigma^2)$, the MGF is $\Ex[e^{sX}] = \exp(\mu s + \sigma^2 s^2/2)$ where $X = \log \Risky$. Setting $s = 1-\CRRA$ and $\mu = r + \equityPrem - \std_{\risky}^2/2$ yields[^lognormal-returns-intuition]

$$
\Ex[\Risky^{1-\CRRA}] = \exp\left((1-\CRRA)\left(r+\equityPrem - \frac{\std_{\risky}^2}{2}\right) + \frac{(1-\CRRA)^2\std_{\risky}^2}{2}\right).
$$

Simplifying the variance terms: $(1-\CRRA)^2\std_{\risky}^2/2 - (1-\CRRA)\std_{\risky}^2/2 = (1-\CRRA)[(1-\CRRA)-1]\std_{\risky}^2/2 = (1-\CRRA)(-\CRRA)\std_{\risky}^2/2$, giving the final form

$$
\Ex[\Risky^{1-\CRRA}] = \exp((1-\CRRA)(r+\equityPrem) + (1-\CRRA)(1-2\CRRA)\std_{\risky}^2/2).
$$

[^lognormal-returns-intuition]:
    Here we can interpret $\equityPrem$ as the risk premium, that is, the additional average return from holding a risky asset compared to the risk-free rate $r$.  Adjusting the average log return by the asset volatility ensures that increasing $\std_{\risky}^2$ constitutes a mean-preserving spread of the level of return.

## Correlated Returns: Shock Discretization

For serially correlated returns, the return state becomes an additional state variable, requiring two-dimensional interpolation of the moderation ratio.  Moreover, continuous shock distributions require discretization. 

The Tauchen method {cite:p}`tauchen1986` constructs a Markov chain by dividing the state space into bins. The Tauchen-Hussey method {cite:p}`TauchenHussey1991` uses Gaussian quadrature, often requiring fewer states for comparable accuracy. For unemployment shocks, assign probability $\WorstProb$ to zero income and $(1-\WorstProb)$ across positive realizations. Choose gridpoints and shock points via convergence analysis. 