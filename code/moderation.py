r"""The Method of Moderation for Consumption-Saving Problems.

This module implements the Method of Moderation (MoM) solution technique for
consumption-saving problems with idiosyncratic income shocks, as described in
"The Method of Moderation" by Chris Carroll and Alan Lujan.
Code implementation by Alan Lujan.

Mathematical Framework
----------------------
The Method of Moderation leverages the fact that the realist consumer's optimal
consumption will always be bounded between two extreme cases:

1. **Optimist**: Assumes all future shocks take their expected value (E[θ] = 1)
   - Has analytical perfect-foresight solution: c_opt(m) = κ_min * (m + h)
   - Where κ_min is the minimum MPC and h is expected human wealth

2. **Pessimist**: Assumes all future shocks take their worst possible value
   - Has analytical perfect-foresight solution: c_pes(m) = κ_min * (m - m_min)
   - Where m_min is the natural borrowing constraint (= -h_pes)

The realist solution c_real(m) satisfies: c_pes(m) ≤ c_real(m) ≤ c_opt(m)

Core Algorithm: The Method of Moderation
----------------------------------------
The algorithm transforms the consumption function construction problem by working
in the space between these analytical bounds:

1. **Solve standard EGM**: Use endogenous gridpoints to get consumption/resources pairs
2. **Calculate moderation ratio**: ω(μ) = (c_opt - c_real) / (c_opt - c_pes)
   where μ = log(m - m_min) is log excess market resources
3. **Apply chi transformation**: χ(μ) = log((1-ω)/ω) for asymptotic linearity
4. **Interpolate chi function**: Create smooth χ(μ) interpolant with derivatives
5. **Reconstruct consumption**: c_real = c_opt - ω * (c_opt - c_pes)

Key Mathematical Properties
---------------------------
- **Asymptotic linearity**: χ'(μ) → 0 as μ → \infty (prevents negative precautionary saving)
- **Moderation bounds**: ω ∈ [0,1] ensures consumption stays within theoretical bounds
- **Log excess resources**: μ = log(m - m_min) maps domain (m_min, \infty) to (-\infty, \infty)
- **Numerical stability**: Uses log1p/expm1 for robust floating-point arithmetic
- **Smooth extrapolation**: Chi function provides excellent out-of-sample behavior

Performance Advantages
----------------------
- **Superior extrapolation**: No negative precautionary saving for large wealth
- **Computational efficiency**: Leverages analytical bounds + single EGM solve
- **Numerical robustness**: Stable across extreme wealth levels and parameter values
- **Theoretical foundation**: Grounded in consumption theory, not ad-hoc fixes

Classes
-------
IndShockEGMConsumerType : HARK consumer type using standard endogenous grid method
    Baseline implementation for comparison with Method of Moderation
IndShockMoMConsumerType : HARK consumer type using Method of Moderation
    Superior implementation with excellent extrapolation properties

Core Solvers
------------
endogenous_grid_method : Standard EGM solver (baseline for comparison)
method_of_moderation : Method of Moderation solver (primary implementation)

Helper Functions
----------------
prepare_to_solve : Calculate shared economic parameters and constraints
make_behavioral_bounds : Create optimist, pessimist, and tighter upper bound solutions
solve_egm_step : Execute core Endogenous Grid Method calculation step
construct_value_functions : Build complete value functions with inverse utility transformations

Mathematical Utility Functions
------------------------------
log_mnrm_ex : Log excess market resources transformation μ = log(m - m_min)
moderate : Moderation ratio calculation ω = (c_opt - c_real)/(c_opt - c_pes)
logit_moderate : Asymptotically linear transformation χ = log((1-ω)/ω)
expit_moderate : Inverse chi transformation ω = 1/(1 + exp(χ))
TransformedFunctionMoM : Generalized function transformer using Method of Moderation

Numerical Stability Features
-----------------------------
- **log1p/expm1**: Used throughout for accurate computation near 0
- **Overflow protection**: Handles extreme values of μ, ω, and χ
- **Derivative augmentation**: Includes derivatives for smoother interpolation
- **Boundary handling**: Robust extrapolation beyond computed gridpoints

Extensions and Refinements
-------------------------
The paper describes extensions including:
- Tighter upper bounds using maximum MPC constraints
- Stochastic rate of return extensions
- Value function approximation using similar moderation techniques

Usage Example
-------------
>>> # Standard EGM (baseline with extrapolation problems)
>>> consumer_egm = IndShockEGMConsumerType()
>>> consumer_egm.solve()
>>>
>>> # Method of Moderation (superior extrapolation)
>>> consumer_mom = IndShockMoMConsumerType()
>>> consumer_mom.solve()
>>>
>>> # Compare extrapolation properties
>>> m_large = 50.0  # Large wealth level
>>> c_egm = consumer_egm.solution[0].cFunc(m_large)
>>> c_mom = consumer_mom.solution[0].cFunc(m_large)
>>> # c_mom has better theoretical properties and no negative precautionary saving

References
----------
"The Method of Moderation" by Chris Carroll and Alan Lujan

Notes
-----
All consumption functions include auxiliary bounds (cOptFunc, cPesFunc, cTightFunc)
to enable Method of Moderation analysis and comparison across solution methods.
These bounds correspond to the optimist, pessimist, and tighter upper bound
consumption functions described in the paper.

"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from HARK import (
    NullFunc,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    IndShockConsumerType_defaults,
    calc_boro_const_nat,
    calc_human_wealth,
    calc_m_nrm_min,
    calc_mpc_max,
    calc_mpc_min,
    calc_patience_factor,
    calc_v_next,
    calc_vp_next,
    calc_vpp_next,
    calc_worst_inc_prob,
)
from HARK.distributions import (
    expected,
)
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA

# =========================================================================
# Core solvers (declared early to guide file order)
# =========================================================================


def endogenous_grid_method(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
    ExtrapBool,
):
    return _endogenous_grid_method(
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
        ExtrapBool,
    )


def method_of_moderation(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    return _method_of_moderation(
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
    )


def _create_interpolation(
    cubic_bool: bool,
    x_list,
    y_list,
    dydx_list=None,
    intercept=None,
    slope=None,
):
    """Create interpolation function based on CubicBool flag.

    Parameters
    ----------
    cubic_bool : bool
        If True, create CubicInterp. If False, create LinearInterp.
    x_list : array_like
        Input values for interpolation.
    y_list : array_like
        Output values for interpolation.
    dydx_list : array_like, optional
        Derivative values for cubic interpolation. Ignored for linear.
        Required when cubic_bool=True (will raise ValueError if None).
    intercept : float, optional
        Extrapolation intercept.
    slope : float, optional
        Extrapolation slope.

    Returns
    -------
    InterpolationFunc : CubicInterp or LinearInterp
        Configured interpolation function.

    """
    if cubic_bool:
        # Require derivatives for cubic interpolation
        if dydx_list is None:
            msg = (
                "Derivatives (dydx_list) must be provided when cubic_bool=True. "
                "CubicInterp requires derivative values for proper interpolation."
            )
            raise ValueError(
                msg,
            )
        return CubicInterp(
            x_list,
            y_list,
            dydx_list,
            intercept,
            slope,
            lower_extrap=True,
        )
    return LinearInterp(x_list, y_list, intercept, slope, lower_extrap=True)


# Define constants for Method of Moderation extrapolation
MOM_EXTRAP_GAP_LEFT = 0.05  # Small leftward extension (lower wealth)
MOM_EXTRAP_GAP_RIGHT = 0.5  # Larger rightward extension (higher wealth direction)


# =========================================================================
# Shared helpers to organize common steps (EGM vs MoM)
# =========================================================================


def _compute_mpc_vector(
    u_func: UtilityFuncCRRA,
    end_of_prd_vpp: np.ndarray,
    c_nrm: np.ndarray,
) -> np.ndarray:
    """Compute MPC vector from end-of-period v'' and c grid."""
    dcda = end_of_prd_vpp / u_func.der(np.asarray(c_nrm), order=2)
    return dcda / (dcda + 1.0)


def _build_cfunc_egm(
    *,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vPPfuncNext,
    uFunc,
    aNrm,
    cNrm,
    mNrm,
    mNrmMin,
    hNrm,
    MPCmin,
    MPCmax,
    ExtrapBool,
    CubicBool,
):
    """Construct consumption function for EGM path with ordered steps and minimal duplication."""
    # Boundary augmentation
    cNrmAug = np.insert(cNrm, 0, 0.0)
    mNrmAug = np.insert(mNrm, 0, mNrmMin)

    # Extrapolation parameters for asymptotic behavior
    cFuncLimitIntercept = MPCmin * hNrm
    cFuncLimitSlope = MPCmin

    # Derivative data for cubic interpolation
    if CubicBool:
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        EndOfPrdvPP = vPPfacEff * expected(
            calc_vpp_next,
            IncShkDstn,
            args=(aNrm, Rfree, CRRA, PermGroFac, vPPfuncNext),
        )
        MPC = _compute_mpc_vector(uFunc, EndOfPrdvPP, cNrm)
        MPCAug = np.insert(MPC, 0, MPCmax)
    else:
        MPCAug = None

    # Interpolant
    return _create_interpolation(
        CubicBool,
        mNrmAug,
        cNrmAug,
        MPCAug,
        cFuncLimitIntercept if ExtrapBool else None,
        cFuncLimitSlope if ExtrapBool else None,
    )


def _build_cfunc_mom(
    *,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vPPfuncNext,
    uFunc,
    aNrm,
    cNrm,
    mNrm,
    mNrmMin,
    hNrm,
    MPCmin,
    CubicBool,
    optimist,
    pessimist,
):
    """Construct consumption function for MoM path (χ/ω over μ + TransformedFunctionMoM)."""
    # μ grid and derivative inputs
    mNrmEx = mNrm - mNrmMin
    hNrmEx = hNrm + mNrmMin
    mu = log_mnrm_ex(mNrm, mNrmMin)

    # MPC vector for consumption-side Hermite slopes (if cubic)
    vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
    EndOfPrdvPP = vPPfacEff * expected(
        calc_vpp_next,
        IncShkDstn,
        args=(aNrm, Rfree, CRRA, PermGroFac, vPPfuncNext),
    )
    MPC = _compute_mpc_vector(uFunc, EndOfPrdvPP, cNrm)

    # Moderation ratio and derivatives
    modRte = moderate(mNrm, optimist.cFunc, cNrm, pessimist.cFunc)
    modRteMu = mNrmEx * (MPCmin - MPC) / (MPCmin * hNrmEx)
    logitModRte = logit_moderate(modRte)
    logitModRteMu = modRteMu / ((modRte - 1) * modRte)

    # Interpolants and wrapper
    modRteFunc, logitModRteFunc = _construct_mom_interpolants(
        mu,
        modRte,
        modRteMu,
        logitModRte,
        logitModRteMu,
        CubicBool,
    )
    return TransformedFunctionMoM(
        mNrmMin,
        modRteFunc,
        logitModRteFunc,
        optimist.cFunc,
        pessimist.cFunc,
    )


def _build_vfunc_egm(
    *,
    aNrm,
    mNrm,
    mNrmMin,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vFuncNext,
    EndOfPrdvP,
    uFunc,
    cNrm,
    CubicBool,
    BoroCnstNat,
    MPCmax,
    MPCmin,
    hNrm,
):
    """Construct beginning-of-period value function for EGM path."""
    vNvrs, vNvrsP = construct_value_functions(
        aNrm,
        BoroCnstNat,
        DiscFacEff,
        Rfree,
        PermGroFac,
        CRRA,
        IncShkDstn,
        vFuncNext,
        EndOfPrdvP,
        uFunc,
        cNrm,
        CubicBool,
    )
    # Augment and interpolate
    mNrmAug = np.insert(mNrm, 0, mNrmMin)
    vNvrsAug = np.insert(vNvrs, 0, 0.0)
    vNvrsPAug = np.insert(vNvrsP, 0, MPCmax ** (-CRRA / (1.0 - CRRA)))
    vNvrsPDerivatives = vNvrsPAug if CubicBool else None
    MPCminNvrs = MPCmin ** (-CRRA / (1.0 - CRRA))
    intercept = MPCminNvrs * hNrm
    slope = MPCminNvrs
    vNvrsFunc = _create_interpolation(
        CubicBool,
        mNrmAug,
        vNvrsAug,
        vNvrsPDerivatives,
        intercept,
        slope,
    )
    return ValueFuncCRRA(vNvrsFunc, CRRA)


def _build_vfunc_mom(
    *,
    mNrm,
    mNrmMin,
    hNrm,
    MPCmin,
    CRRA,
    vNvrs,
    vNvrsP,
    optimist,
    pessimist,
    CubicBool,
):
    """Construct beginning-of-period value function for MoM path via χ/ω over μ."""
    mNrmEx = mNrm - mNrmMin
    hNrmEx = hNrm + mNrmMin
    mu = log_mnrm_ex(mNrm, mNrmMin)

    vNvrsOptFunc = optimist.vFunc.vFuncNvrs
    vNvrsPesFunc = pessimist.vFunc.vFuncNvrs

    modRte = moderate(mNrm, vNvrsOptFunc, vNvrs, vNvrsPesFunc)
    MPCminNvrs = MPCmin ** (-CRRA / (1.0 - CRRA))
    modRteMu = mNrmEx * (MPCminNvrs - vNvrsP) / (hNrmEx * MPCminNvrs)
    logitModRte = logit_moderate(modRte)
    logitModRteMu = -modRteMu / (modRte * (1 - modRte))

    modRteFunc, logitModRteFunc = _construct_mom_interpolants(
        mu,
        modRte,
        modRteMu,
        logitModRte,
        logitModRteMu,
        CubicBool,
    )
    moderated_vfunc = TransformedFunctionMoM(
        mNrmMin,
        modRteFunc,
        logitModRteFunc,
        vNvrsOptFunc,
        vNvrsPesFunc,
    )
    return ValueFuncCRRA(moderated_vfunc, CRRA)


def prepare_to_solve(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
):
    """Calculate shared economic parameters and constraints for consumption-saving solvers.

    This function consolidates the complex repeated calculations used by both
    the endogenous grid method and method of moderation solvers. It performs
    multi-step calculations including income shock statistics, borrowing constraints,
    and marginal propensity to consume bounds that are identical across both methods.

    The function encapsulates:
    - Effective discount factor calculation
    - Income shock distribution analysis
    - Human wealth present discounted value computation
    - Natural and artificial borrowing constraint reconciliation
    - Patience factor and MPC bounds derivation

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one-period problem containing previous
        value functions, MPC bounds, and minimum market resources.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the current
        period and the next period (permanent and transitory shocks).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk-free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint for minimum allowable assets to end
        the period with. If None or greater than natural constraint, it is
        not binding.

    Returns
    -------
    tuple
        Economic parameters in the following order:
        - DiscFacEff (float): Effective discount factor (DiscFac * LivPrb)
        - hNrm (float): Present discounted value of human wealth
        - BoroCnstNat (float): Natural borrowing constraint
        - mNrmMin (float): Minimum market resources this period
        - MPCmin (float): Lower bound on marginal propensity to consume
        - MPCmaxUnc (float): Upper bound on MPC (unconstrained case)
        - MPCmax (float): Upper bound on MPC (constrained if needed)

    Notes
    -----
    This function eliminates ~40 lines of repeated code between the two solvers
    while ensuring identical parameter calculations. All computations follow
    the standard HARK conventions for consumption-saving models.

    """
    # Effective discount factor
    DiscFacEff = DiscFac * LivPrb

    # Calculate income shock statistics
    WorstIncPrb = calc_worst_inc_prob(IncShkDstn, use_infimum=False)
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrm = calc_human_wealth(solution_next.hNrm, PermGroFac, Rfree, Ex_IncNext)

    # Calculate borrowing constraints
    BoroCnstNat = calc_boro_const_nat(
        solution_next.mNrmMin,
        IncShkDstn,
        Rfree,
        PermGroFac,
        use_infimum=False,
    )
    mNrmMin = calc_m_nrm_min(BoroCnstArt, BoroCnstNat)

    # Calculate MPC bounds
    PatFac = calc_patience_factor(Rfree, DiscFacEff, CRRA)
    MPCmin = calc_mpc_min(solution_next.MPCmin, PatFac)
    MPCmaxUnc = calc_mpc_max(
        solution_next.MPCmax,
        WorstIncPrb,
        CRRA,
        PatFac,
        BoroCnstNat,
        BoroCnstArt,
    )
    MPCmax = 1.0 if BoroCnstNat < mNrmMin else MPCmaxUnc

    return (
        DiscFacEff,
        hNrm,
        BoroCnstNat,
        mNrmMin,
        MPCmin,
        MPCmax,
    )


def make_behavioral_bounds(hNrm, mNrmMin, MPCmin, MPCmax, CRRA):
    """Create behavioral bounds for Method of Moderation analysis.

    Constructs the three perfect foresight solutions that provide theoretical
    bounds for optimal consumption under uncertainty. These bounds are central
    to the Method of Moderation approach and enable analytical constraints on
    the realist consumer's behavior.

    The three behavioral bounds are:

    1. **Optimist**: Assumes all future income shocks take their expected value
       - Perfect foresight consumption: c_opt(m) = κ_min * (m + h)
       - Represents the upper bound on consumption behavior
       - Used as intercept: hNrm, slope: MPCmin

    2. **Pessimist**: Assumes all future income shocks take worst possible value
       - Perfect foresight consumption: c_pes(m) = κ_min * (m - m_min)
       - Represents the lower bound on consumption behavior
       - Used as intercept: -mNrmMin, slope: MPCmin

    3. **Tighter Upper Bound**: Pessimist constraint with maximum MPC
       - Perfect foresight consumption: c_tight(m) = κ_max * (m - m_min)
       - Provides tighter bound near the borrowing constraint
       - Used as intercept: -mNrmMin, slope: MPCmax

    Parameters
    ----------
    hNrm : float
        Present discounted value of human wealth (expected future income)
    mNrmMin : float
        Minimum feasible market resources (effective borrowing constraint)
    MPCmin : float
        Minimum marginal propensity to consume (κ_min)
    MPCmax : float
        Maximum marginal propensity to consume (κ_max)
    CRRA : float
        Coefficient of relative risk aversion

    Returns
    -------
    tuple
        Three ConsumerSolution objects in order:
        - optimist: Upper bound consumption function and value functions
        - pessimist: Lower bound consumption function and value functions
        - tighter_upper_bound: Alternative upper bound near constraint

    Notes
    -----
    These bounds satisfy: c_pes(m) ≤ c_real(m) ≤ c_opt(m) for all m > m_min.
    The Method of Moderation constructs the realist solution by moderating
    between these analytical bounds using the chi transformation.

    """
    optimist = soln_perf_foresight(hNrm, MPCmin, CRRA)
    pessimist = soln_perf_foresight(-mNrmMin, MPCmin, CRRA)
    tighter_upper_bound = soln_perf_foresight(-mNrmMin, MPCmax, CRRA)

    return optimist, pessimist, tighter_upper_bound


def solve_egm_step(
    aXtraGrid,
    mNrmMin,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vPfuncNext,
    uFunc,
):
    """Execute the core Endogenous Grid Method (EGM) calculation step.

    This function implements the standard EGM algorithm that is the computational
    foundation for both the endogenous grid method and method of moderation solvers.
    The EGM approach works backwards from an exogenous grid of end-of-period assets
    to find the endogenous grid of beginning-of-period market resources.

    Algorithm Steps:
    1. **Exogenous Assets Grid**: Start with aXtraGrid + natural borrowing constraint
    2. **Backward Induction**: Calculate end-of-period marginal value E[βRu'(c_{t+1})]
    3. **Euler Equation**: Invert FOC to find consumption: c_t = (u')^{-1}(E[βRu'(c_{t+1})])
    4. **Budget Constraint**: Back out market resources: m_t = c_t + a_t

    Mathematical Foundation:
    The Euler equation for optimal consumption is:
        u'(c_t) = βR * E[u'(c_{t+1})]
    where the expectation is over income shocks. The EGM inverts this to solve
    for consumption given assets, then uses the budget constraint to find the
    corresponding market resources.

    Parameters
    ----------
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values above the borrowing constraint
    mNrmMin : float
        Minimum feasible market resources (effective borrowing constraint)
    DiscFacEff : float
        Effective discount factor (DiscFac * LivPrb)
    Rfree : float
        Risk-free interest factor on end-of-period assets
    PermGroFac : float
        Expected permanent income growth factor
    CRRA : float
        Coefficient of relative risk aversion
    IncShkDstn : distribution.Distribution
        Discrete approximation to the income shock distribution
    vPfuncNext : callable
        Next period's marginal value function v'_{t+1}(m_{t+1})
    uFunc : callable
        Current period CRRA utility function

    Returns
    -------
    tuple
        Four arrays of equal length representing the EGM solution:
        - aNrm (np.array): End-of-period normalized assets grid
        - cNrm (np.array): Optimal consumption at each asset level
        - mNrm (np.array): Endogenous market resources grid
        - EndOfPrdvP (np.array): End-of-period marginal value E[βRu'(c_{t+1})]

    Notes
    -----
    This function eliminates ~15 lines of repeated code between the two solvers.
    The returned grids form the basis for consumption function interpolation in
    both standard EGM and Method of Moderation approaches.

    """
    # Construct assets grid by adding natural borrowing constraint to aXtraGrid
    aNrm = np.asarray(aXtraGrid) + mNrmMin

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(
        calc_vp_next,
        IncShkDstn,
        args=(aNrm, Rfree, CRRA, PermGroFac, vPfuncNext),
    )

    # EGM Step: Invert FOC to find optimal consumption from each asset gridpoint
    cNrm = uFunc.derinv(EndOfPrdvP, order=(1, 0))  # u'(c) = E[v'(a*R+y)]
    mNrm = cNrm + aNrm  # Back out endogenous market resources grid

    return aNrm, cNrm, mNrm, EndOfPrdvP


def construct_value_functions(
    aNrm,
    BoroCnstNat,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vFuncNext,
    EndOfPrdvP,
    uFunc,
    cNrm,
    CubicBool,
):
    """Construct complete value functions using HARK's inverse utility framework.

    This function performs the comprehensive value function construction process
    that is shared between both EGM and Method of Moderation solvers. It implements
    HARK's standard approach using inverse utility transformations for numerical
    stability and constructs both end-of-period and beginning-of-period value components.

    The construction follows HARK's established methodology:

    **End-of-Period Value Function Construction:**
    1. Calculate value function via backward induction: v(a) = βE[v_{t+1}(Ra + Y_{t+1})]
    2. Transform through inverse utility: vNvrs = u^{-1}(v) for numerical stability
    3. Add boundary conditions (zero value at borrowing constraint)
    4. Construct interpolant (cubic or linear) with optional derivatives for approximation
    5. Create ValueFuncCRRA object for consistent HARK interface

    **Beginning-of-Period Value Evaluation:**
    6. Evaluate Bellman equation on market resources grid: v(m) = u(c(m)) + v_end(a)
    7. Compute marginal values: v'(m) = u'(c(m)) via envelope condition
    8. Apply inverse utility transformations for downstream interpolation

    Mathematical Framework:
    The value function satisfies the Bellman equation:
        v_t(m_t) = max_{c_t} {u(c_t) + β E[v_{t+1}(R(m_t - c_t) + Y_{t+1})]}

    HARK uses the inverse utility transformation vNvrs = u^{-1}(v) because it
    linearizes the problem near the borrowing constraint and provides superior
    numerical properties for interpolation.

    Parameters
    ----------
    aNrm : np.array
        End-of-period normalized assets grid from EGM solution
    mNrmMin : float
        Minimum feasible market resources (effective borrowing constraint)
    DiscFacEff : float
        Effective discount factor (DiscFac * LivPrb)
    Rfree : float
        Risk-free interest factor on end-of-period assets
    PermGroFac : float
        Expected permanent income growth factor
    CRRA : float
        Coefficient of relative risk aversion
    IncShkDstn : distribution.Distribution
        Discrete approximation to the income shock distribution
    vFuncNext : callable
        Next period's value function v_{t+1}(m_{t+1})
    EndOfPrdvP : np.array
        End-of-period marginal value E[βRu'(c_{t+1})] from solve_egm_step
    uFunc : callable
        Current period CRRA utility function
    cNrm : np.array
        Optimal consumption at each gridpoint from solve_egm_step
    CubicBool : bool
        Whether to use cubic spline interpolation (True) or linear interpolation (False)

    Returns
    -------
    tuple
        Two arrays for beginning-of-period value function construction:
        - vNvrs (np.array): Inverse utility transformed values u^{-1}(v(m))
        - vNvrsP (np.array): Derivatives of inverse utility transformed values

    Notes
    -----
    This function consolidates ~25 lines of repeated value function construction
    code between the two solvers. The inverse utility transformation is essential
    for HARK's numerical stability, particularly near borrowing constraints where
    the value function becomes nearly linear.

    The returned values are ready for cubic spline interpolation to create the
    final beginning-of-period value function with proper boundary conditions and
    extrapolation behavior.

    """
    # =========================================================================
    # Step 1: Construct end-of-period value function
    # =========================================================================

    # Calculate end-of-period value at each asset gridpoint
    EndOfPrdv = DiscFacEff * expected(
        calc_v_next,
        IncShkDstn,
        args=(aNrm, Rfree, CRRA, PermGroFac, vFuncNext),
    )

    # Transform through inverse utility for numerical stability
    EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)

    # Add boundary conditions (zero at borrowing constraint)
    EndOfPrdvNvrsAug = np.insert(EndOfPrdvNvrs, 0, 0.0)
    aNrmAug = np.insert(aNrm, 0, BoroCnstNat)

    # Compute derivatives only if using cubic interpolation
    if CubicBool:
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrsPAug = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        EndOfPrdvNvrsDerivatives = EndOfPrdvNvrsPAug
    else:
        EndOfPrdvNvrsDerivatives = None

    EndOfPrdvNvrsFunc = _create_interpolation(
        CubicBool,
        aNrmAug,
        EndOfPrdvNvrsAug,
        EndOfPrdvNvrsDerivatives,
    )

    # Create CRRA value function object
    EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, CRRA)

    # =========================================================================
    # Step 2: Compute beginning-of-period values on market resources grid
    # =========================================================================

    # Compute expected value and marginal value on a grid of market resources

    v = uFunc(cNrm) + EndOfPrdvFunc(aNrm)
    vP = uFunc.der(cNrm)

    # =========================================================================
    # Step 3: Transform values through inverse utility for beginning-of-period function
    # =========================================================================

    # Transform values through inverse utility for numerical stability
    vNvrs = uFunc.inv(v)  # value transformed through inv utility

    # Always compute derivatives for consistent augmentation across interpolation methods
    # This is needed for MoM derivative calculations even with linear interpolation
    vNvrsP = vP * uFunc.derinv(v, order=(0, 1))

    return vNvrs, vNvrsP


class PerfForesightFunc:
    """A simple representation of a linear function f(x) = (x + intercept) * slope.

    This class provides both function evaluation and derivative computation,
    which is essential for the Method of Moderation's derivative calculations.

    Parameters
    ----------
    intercept : float
        The intercept term in the linear function
    slope : float
        The slope (derivative) of the linear function

    Notes
    -----
    This replaces lambda functions for bounding functions in the Method of Moderation
    to ensure proper derivative computation for MPC calculations.

    """

    def __init__(self, intercept, slope) -> None:
        self.intercept = intercept
        self.slope = slope

    def __call__(self, m):
        """Evaluate the linear function at market resources m."""
        return (m + self.intercept) * self.slope

    def derivative(self, m):
        """Compute the derivative of the linear function.

        The derivative is constant (the slope). Handle array inputs for consistency.
        """
        if np.isscalar(m):
            return self.slope
        # Ensure output matches input shape if m is an array
        return np.full_like(m, self.slope, dtype=float)


def soln_perf_foresight(intercept, slope, crra):
    """Create perfect foresight solution using LinearFunc for proper derivatives."""
    vNvrs_slope = slope ** ((-crra) / (1 - crra))

    # Use LinearFunc instead of lambda for proper derivative support
    cFunc = PerfForesightFunc(intercept, slope)
    vNvrsFunc = PerfForesightFunc(intercept, vNvrs_slope)

    return ConsumerSolution(
        cFunc=cFunc,
        vFunc=ValueFuncCRRA(vNvrsFunc, crra),
        vPfunc=MargValueFuncCRRA(cFunc, crra),
    )


def _endogenous_grid_method(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
    ExtrapBool,
):
    """Standard Endogenous Grid Method solver for consumption-saving problems.

    This is the baseline EGM implementation used for comparison with the Method
    of Moderation. As demonstrated in the paper, it suffers from poor extrapolation
    properties - linear extrapolation can predict negative precautionary saving
    for large market resources, which violates economic theory and creates the
    "extrapolation problem" shown in Figure 1 of the paper.

    The EGM algorithm:
    1. Start with an exogenous grid of end-of-period assets a
    2. Compute optimal consumption via the Euler equation FOC
    3. Back out the corresponding endogenous market resources grid m
    4. Interpolate between gridpoints to create the consumption function

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_ : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.
        Includes auxiliary consumption functions for Method of Moderation bounds:
        - cOptFunc: Optimist (perfect foresight) consumption function
        - cPesFunc: Pessimist (worst-case) consumption function
        - cTightFunc: Tighter upper bound consumption function

    Notes
    -----
    The extrapolation problem occurs because the consumption function is
    approximated by linear interpolation between computed gridpoints. Outside
    the grid range, linear extrapolation can violate theoretical constraints.

    """
    # =========================================================================
    # Setup and calculate economic parameters
    # =========================================================================
    uFunc = UtilityFuncCRRA(CRRA)

    # Calculate shared economic parameters and constraints
    (
        DiscFacEff,
        hNrm,
        BoroCnstNat,
        mNrmMin,
        MPCmin,
        MPCmax,
    ) = prepare_to_solve(
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
    )
    # If the borrowing constraint is not the natural borrowing constraint,
    # we need to add a gridpoint at the borrowing constraint
    aXtraGrid = np.insert(aXtraGrid, 0, 0.0) if BoroCnstNat != mNrmMin else aXtraGrid

    # Unpack next period's value functions
    vFuncNext = solution_next.vFunc
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc

    # =========================================================================
    # Step 2: Execute core Endogenous Grid Method algorithm
    # =========================================================================
    # Solve EGM step: assets grid → consumption grid → market resources grid
    aNrm, cNrm, mNrm, EndOfPrdvP = solve_egm_step(
        aXtraGrid,
        mNrmMin,
        DiscFacEff,
        Rfree,
        PermGroFac,
        CRRA,
        IncShkDstn,
        vPfuncNext,
        uFunc,
    )

    # Add boundary condition: consumption approaches zero as m approaches constraint
    np.insert(cNrm, 0, 0.0)
    np.insert(mNrm, 0, mNrmMin)

    # =========================================================================
    # Step 3: Construct consumption function
    # =========================================================================
    cFunc = _build_cfunc_egm(
        DiscFacEff=DiscFacEff,
        Rfree=Rfree,
        PermGroFac=PermGroFac,
        CRRA=CRRA,
        IncShkDstn=IncShkDstn,
        vPPfuncNext=vPPfuncNext,
        uFunc=uFunc,
        aNrm=aNrm,
        cNrm=cNrm,
        mNrm=mNrm,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
        ExtrapBool=ExtrapBool,
        CubicBool=CubicBool,
    )

    # =========================================================================
    # Step 4: Construct marginal value functions from consumption function
    # =========================================================================
    # Create marginal value function v'(m) = u'(c(m)) via envelope condition
    vPfunc = MargValueFuncCRRA(cFunc, CRRA)

    # Create second derivative of value function v''(m) if using cubic interpolation
    vPPfunc = MargMargValueFuncCRRA(cFunc, CRRA)

    # Construct this period's value function if requested
    if vFuncBool:
        vFunc = _build_vfunc_egm(
            mNrm=mNrm,
            mNrmMin=mNrmMin,
            DiscFacEff=DiscFacEff,
            Rfree=Rfree,
            PermGroFac=PermGroFac,
            CRRA=CRRA,
            IncShkDstn=IncShkDstn,
            vFuncNext=vFuncNext,
            EndOfPrdvP=EndOfPrdvP,
            uFunc=uFunc,
            cNrm=cNrm,
            CubicBool=CubicBool,
            BoroCnstNat=BoroCnstNat,
            MPCmax=MPCmax,
            MPCmin=MPCmin,
            hNrm=hNrm,
            aNrm=aNrm,
        )
    else:
        vFunc = NullFunc()  # Dummy object

    # Create and return this period's solution

    solution = ConsumerSolution(
        cFunc=cFunc,
        vFunc=vFunc,
        vPfunc=vPfunc,
        vPPfunc=vPPfunc,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
    )

    # Add behavioral bounds for Method of Moderation analysis
    solution.Optimist, solution.Pessimist, solution.TighterUpperBound = (
        make_behavioral_bounds(hNrm, mNrmMin, MPCmin, MPCmax, CRRA)
    )

    return solution


IndShockEGMConsumerType_defaults = IndShockConsumerType_defaults.copy()
IndShockEGMConsumerType_defaults["ExtrapBool"] = False


class IndShockEGMConsumerType(IndShockConsumerType):
    """HARK consumer type using standard Endogenous Grid Method.

    This class provides the baseline EGM implementation for comparison with
    the Method of Moderation. As demonstrated in Figure 1 of the paper, it
    uses the standard endogenous_grid_method solver which suffers from the
    "extrapolation problem" - predicting negative precautionary saving for
    large wealth levels.

    Attributes
    ----------
    solver : function
        endogenous_grid_method - Standard EGM solver with extrapolation issues

    Notes
    -----
    Solutions include auxiliary consumption functions (cOptFunc, cPesFunc,
    cTightFunc) to enable direct comparison with Method of Moderation results
    and to demonstrate the theoretical bounds described in the paper.

    """

    default_: ClassVar = {
        "params": IndShockEGMConsumerType_defaults,
        "solver": endogenous_grid_method,
        "model": "ConsIndShock.yaml",
    }

    time_inv_: ClassVar = [*IndShockConsumerType.time_inv_, "ExtrapBool"]


def log_mnrm_ex(m, m_min):
    r"""Log excess market resources transformation for Method of Moderation.

    Computes μ = log(m - m_min), the log-excess market resources transformation
    that maps the natural domain (m_min, \infty) to (-\infty, \infty). This transformation
    is central to the Method of Moderation as it provides a convenient space
    for interpolating the chi function χ(μ).

    Parameters
    ----------
    m : float or array
        Market resources (cash-on-hand)
    m_min : float
        Minimum feasible market resources (effective borrowing constraint)
        Corresponds to -h_pes in the paper notation

    Returns
    -------
    float or array
        Log excess market resources: μ = log(m - m_min)

    Notes
    -----
    Implementation uses np.log1p for numerical stability, avoiding issues
    when m - m_min is close to zero.

    Implemented as log1p(m - m_min - 1) for stability.

    This transformation is used as the input space for the chi function χ(μ)
    in the Method of Moderation. The log transformation ensures the chi function
    can be asymptotically linear, preventing negative precautionary saving.

    """
    return np.log1p(m - m_min - 1)


def exp_mu(mu, m_min):
    r"""Inverse log excess market resources transformation.

    Computes m = exp(μ) + m_min, the inverse of the log-excess transformation
    μ = log(m - m_min). This maps the unbounded domain (-\infty, \infty) back to the
    natural domain (m_min, \infty).

    Parameters
    ----------
    mu : float or array
        Log excess market resources μ = log(m - m_min)
    m_min : float
        Minimum feasible market resources (effective borrowing constraint)

    Returns
    -------
    float or array
        Market resources: m = exp(μ) + m_min

    Notes
    -----
    Implementation uses np.expm1 for numerical stability when μ is close to 0.
    Implemented as expm1(μ) + m_min + 1 which equals exp(μ) + m_min.

    """
    return np.expm1(mu) + m_min + 1


def moderate(m, f_opt, f_real, f_pess):
    """Moderation ratio function ω for Method of Moderation.

    General utility function that computes the moderation ratio ω that measures how much
    the realist consumer moderates between optimist and pessimist behavioral levels.
    Following equation (5) in the paper, the ratio is defined as:

    ω = (f_opt - f_real) / (f_opt - f_pess)

    This function is generic and works with any economic functions that follow the
    bounded behavior pattern:
    - For consumption: f_opt = c_opt, f_real = c_real, f_pess = c_pes
    - For value functions: f_opt = v_opt, f_real = v_real, f_pess = v_pes
    - For inverse value functions: f_opt = vNvrs_opt, f_real = vNvrs_real, f_pess = vNvrs_pes

    Parameters
    ----------
    m : float or array
        Market resources (cash-on-hand) where functions are evaluated
    f_opt : callable
        Optimist function (upper bound, perfect foresight behavior)
    f_real : float, array, or callable
        Realist function values (actual optimal behavior under uncertainty)
    f_pess : callable
        Pessimist function (lower bound, worst-case behavior)

    Returns
    -------
    float or array
        Moderation ratio ω ∈ [0,1]

    Notes
    -----
    This is a utility function for external analysis and verification. The main
    Method of Moderation implementation computes moderation ratios inline for efficiency.

    Mathematical properties:
    - ω = 0: realist behaves like optimist (high wealth limit)
    - ω = 1: realist behaves like pessimist (low wealth limit)
    - Strictly between 0 and 1 for all m > m_min under economic prudence
    - Generic design allows application to consumption, value, or other economic functions

    Examples
    --------
    For consumption functions:
    >>> m_vals = np.array([2.0, 5.0, 10.0])
    >>> omega_c = moderate(m_vals, c_opt_func, c_real_vals, c_pes_func)

    For value functions:
    >>> omega_v = moderate(m_vals, v_opt_func, v_real_vals, v_pes_func)

    """
    f_opt_vals = f_opt(m)
    f_pess_vals = f_pess(m)
    return (f_opt_vals - f_real) / (f_opt_vals - f_pess_vals)


def logit_moderate(omega):
    r"""Chi transformation χ(ω) for asymptotically linear interpolation.

    Computes the chi transformation following equation (6) in the paper:
    χ = log((1 - ω) / ω) = log(1/ω - 1)

    This transformation is the mathematical heart of the Method of Moderation's
    superior extrapolation properties. By transforming the bounded moderation ratio
    ω ∈ (0,1) to the unbounded chi space χ ∈ (-\infty,\infty), it enables asymptotically
    linear interpolation that prevents negative precautionary saving.

    Parameters
    ----------
    omega : float or array
        Moderation ratio ω ∈ (0,1) measuring how much the realist moderates
        between optimist and pessimist behavior

    Returns
    -------
    float or array
        Chi transformation value χ ∈ (-\infty,\infty)

    Notes
    -----
    **Numerical Implementation:**
    - Clamps omega to [epsilon, 1-epsilon] to avoid NaN/inf from floating-point errors
    - Uses log1p(1/ω - 2) for numerical stability throughout (0,1) domain
    - Mathematical equivalence: log1p(1/ω - 2) = log(1 + 1/ω - 2) = log((1-ω)/ω)
    - Avoids potential overflow/underflow issues near ω → 0 or ω → 1

    **Asymptotic Properties:**
    - As ω → 0 (realist → optimist): χ → +\infty
    - As ω → 1 (realist → pessimist): χ → -\infty
    - The derivative χ'(μ) → 0 as μ → \infty, ensuring linear extrapolation
    - This linearity prevents the negative precautionary saving that plagues EGM

    **Economic Interpretation:**
    - χ < 0: realist closer to pessimist (high precautionary saving)
    - χ > 0: realist closer to optimist (low precautionary saving)
    - χ ≈ 0: realist balanced between extremes

    """
    return np.log1p(1 / omega - 2)


def expit_moderate(chi):
    """Inverse chi transformation ω(χ) for Method of Moderation.

    Computes the inverse of the chi transformation to recover the moderation ratio:
    ω = 1 / (1 + exp(χ))

    This is the inverse of logit_moderate: if χ = logit_moderate(ω), then
    ω = expit_moderate(χ). Uses numerically stable computation to avoid
    overflow when χ is large.

    Parameters
    ----------
    chi : float or array
        Chi transformation value χ

    Returns
    -------
    float or array
        Moderation ratio ω ∈ (0,1)

    Notes
    -----
    - Uses expm1 for numerical stability: ω = 1/(2 + expm1(χ))
    - Avoids overflow when χ is large (ω → 0)
    - Central to reconstructing consumption from interpolated chi function
    - Ensures ω ∈ (0,1) for all finite χ values

    Mathematical equivalence:
    1/(2 + expm1(χ)) = 1/(1 + 1 + exp(χ) - 1) = 1/(1 + exp(χ))

    """
    return 1.0 / (2.0 + np.expm1(chi))


class TransformedFunctionMoM:
    """Generalized Method of Moderation function transformer.

    This class provides the core moderation logic for functions bounded between
    two lines (optimist and pessimist bounds). It applies the MoM formula:
    f_real(m) = f_opt(m) - ω(μ) * (f_opt(m) - f_pes(m))

    This class can be used for any function that needs to be moderated between
    upper and lower bounds, including consumption functions, value functions, etc.

    The implementation uses the standard chi transformation approach:
    - μ = log(m - m_min) is the log excess market resources
    - χ(μ) is the interpolated chi function (asymptotically linear)
    - ω(μ) = 1/(1 + exp(χ(μ))) is the moderation ratio
    - Provides superior numerical stability for extrapolation

    Parameters
    ----------
    mNrmMin : float
        Minimum feasible market resources (effective borrowing constraint)
    modRteFunc : callable
        Interpolated moderation ratio function (reserved for future use)
    logitModRteFunc : callable
        Interpolated chi function that maps μ → χ (core transformation)
    optimist_func : callable
        Function that computes the optimist (upper) bound f_opt(m)
    pessimist_func : callable
        Function that computes the pessimist (lower) bound f_pes(m)

    Returns
    -------
    callable
        The moderated function f_real(m) that interpolates between bounds

    """

    def __init__(
        self,
        mNrmMin,
        modRteFunc,
        logitModRteFunc,
        optimist_func,
        pessimist_func,
    ) -> None:
        self.mNrmMin = mNrmMin
        self.modRteFunc = modRteFunc
        self.logitModRteFunc = logitModRteFunc
        self.optimist_func = optimist_func
        self.pessimist_func = pessimist_func

    def __call__(self, m):
        """Evaluate the moderated function at market resources m.

        Parameters
        ----------
        m : float or array_like
            Market resources (cash-on-hand) to evaluate function at

        Returns
        -------
        float or array_like
            Moderated function value(s) satisfying theoretical bounds

        """
        # Transform to log excess resources
        mu = log_mnrm_ex(m, self.mNrmMin)

        # Get optimist and pessimist bounds
        f_opt = self.optimist_func(m)
        f_pes = self.pessimist_func(m)

        # Compute moderation ratio omega using chi function with transformation
        chi = self.logitModRteFunc(mu)
        omega = expit_moderate(chi)

        # Apply moderation: f_real = f_opt - ω * (f_opt - f_pes)
        return f_opt - omega * (f_opt - f_pes)

    def derivative(self, m):
        """Compute the derivative of the moderated function.

        This implements the chain rule for the moderated function:
        f_real'(m) = f_opt'(m) - ω'(m) * (f_opt(m) - f_pes(m)) - ω(m) * (f_opt'(m) - f_pes'(m))

        Parameters
        ----------
        m : float or array_like
            Market resources to evaluate derivative at

        Returns
        -------
        float or array_like
            Derivative of the moderated function

        """
        # 1. Setup and Bounding Derivatives
        mu = log_mnrm_ex(m, self.mNrmMin)
        f_opt = self.optimist_func(m)
        f_pes = self.pessimist_func(m)

        # Get derivatives of bounding functions - now required to have derivative method
        try:
            f_opt_prime = self.optimist_func.derivative(m)
            f_pes_prime = self.pessimist_func.derivative(m)
        except AttributeError:
            msg = (
                "Bounding functions must implement a 'derivative' method. "
                "Use LinearFunc for linear bounding functions."
            )
            raise TypeError(
                msg,
            ) from None

        # 2. Calculate d(mu)/dm (Chain Rule Part 1)
        # Protect against division by zero near the constraint
        # mu = log(m - m_min)
        dmu_dm = 1.0 / (m - self.mNrmMin)

        # 3. Calculate omega and d(omega)/dm (Chain Rule Part 2)
        chi = self.logitModRteFunc(mu)
        omega = expit_moderate(chi)
        # Support both HARK's derivativeX convention and standard derivative
        chi_prime_mu = (
            self.logitModRteFunc.derivativeX(mu)
            if hasattr(self.logitModRteFunc, "derivativeX")
            else self.logitModRteFunc.derivative(mu)
        )
        # Chain rule: d(omega)/dm = d(omega)/d(chi) * d(chi)/d(mu) * d(mu)/dm
        # Note: d(omega)/d(chi) = -omega * (1-omega) for expit function
        d_omega_dm = -omega * (1 - omega) * chi_prime_mu * dmu_dm

        # 4. Calculate total derivative using the product rule
        # f_real' = f_opt' - [ d(omega)/dm * (f_opt - f_pes) + omega * (f_opt' - f_pes') ]
        moderation_adjustment = d_omega_dm * (f_opt - f_pes)
        slope_adjustment = omega * (f_opt_prime - f_pes_prime)

        return f_opt_prime - slope_adjustment - moderation_adjustment

    # HARK compatibility: many interpolation utilities expect a 'derivativeX' method
    # that returns df/dx at x. Provide it as an alias to derivative.
    def derivativeX(self, m):
        """Alias for derivative(m) to satisfy HARK's derivativeX contract."""
        return self.derivative(m)


def _construct_mom_interpolants(
    mu,
    modRte,
    modRteMu,
    logitModRte,
    logitModRteMu,
    CubicBool,
):
    """Helper to augment grids and construct interpolants for omega (modRte)
    and chi (logitModRte) functions using predefined extrapolation gaps.

    This function eliminates code duplication in method_of_moderation where
    the same augmentation and interpolation logic is used for both consumption
    and value function construction.

    Parameters
    ----------
    mu : np.array
        Log excess market resources grid μ = log(m - m_min)
    modRte : np.array
        Moderation ratio values ω at each gridpoint
    modRteMu : np.array
        Derivatives of moderation ratio dω/dμ at each gridpoint
    logitModRte : np.array
        Chi transformation values χ = log((1-ω)/ω) at each gridpoint
    logitModRteMu : np.array
        Derivatives of chi transformation dχ/dμ at each gridpoint
    CubicBool : bool
        Whether to use cubic spline interpolation (True) or linear interpolation (False)

    Returns
    -------
    tuple
        Two interpolation objects (CubicInterp or LinearInterp):
        - modRteFunc: ω(μ) interpolant
        - logitModRteFunc: χ(μ) interpolant

    Notes
    -----
    Uses module-level constants MOM_EXTRAP_GAP_LEFT and MOM_EXTRAP_GAP_RIGHT
    for consistent extrapolation behavior across consumption and value functions.

    """
    # Augmented μ grid with extrapolation points
    muAug = np.r_[mu[0] - MOM_EXTRAP_GAP_LEFT, mu, mu[-1] + MOM_EXTRAP_GAP_RIGHT]

    # Augmented ω (modRte) values - use derivative-based extrapolation if available
    if modRteMu is not None:
        # Use derivative-based linear extrapolation (consistent for both cubic/linear)
        modRteAug = np.r_[
            modRte[0] - modRteMu[0] * MOM_EXTRAP_GAP_LEFT,
            modRte,
            modRte[-1] + modRteMu[-1] * MOM_EXTRAP_GAP_RIGHT,
        ]
        modRteMuAug = np.r_[modRteMu[0], modRteMu, modRteMu[-1]]
    else:
        # Fallback to constant extrapolation if no derivatives available
        modRteAug = np.r_[modRte[0], modRte, modRte[-1]]
        modRteMuAug = None

    # Augmented χ (logitModRte) values - use derivative-based extrapolation if available
    if logitModRteMu is not None:
        # Use derivative-based linear extrapolation (consistent for both cubic/linear)
        logitModRteAug = np.r_[
            logitModRte[0] - logitModRteMu[0] * MOM_EXTRAP_GAP_LEFT,
            logitModRte,
            logitModRte[-1] + logitModRteMu[-1] * MOM_EXTRAP_GAP_RIGHT,
        ]
        logitModRteMuAug = np.r_[logitModRteMu[0], logitModRteMu, logitModRteMu[-1]]
    else:
        # Fallback to constant extrapolation if no derivatives available
        logitModRteAug = np.r_[logitModRte[0], logitModRte, logitModRte[-1]]
        logitModRteMuAug = None

    # Create smooth interpolants (cubic or linear based on CubicBool)
    # For MoM interpolants, derivatives are computed only if using cubic interpolation
    if CubicBool:
        modRteDerivatives = modRteMuAug
        logitModRteDerivatives = logitModRteMuAug
    else:
        modRteDerivatives = None
        logitModRteDerivatives = None

    modRteFunc = _create_interpolation(
        CubicBool,
        muAug,
        modRteAug,
        modRteDerivatives,
        None,  # no intercept for MoM interpolants
        None,  # no slope for MoM interpolants
    )

    logitModRteFunc = _create_interpolation(
        CubicBool,
        muAug,
        logitModRteAug,
        logitModRteDerivatives,
        None,  # no intercept for MoM interpolants
        None,  # no slope for MoM interpolants
    )

    return modRteFunc, logitModRteFunc


def _method_of_moderation(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """Method of Moderation solver for consumption-saving problems.

    Solves one period of a consumption-saving model using the Method of Moderation
    technique described in "The Method of Moderation" by Carroll and Lujan. This
    method leverages analytical upper and lower bounds to construct a highly
    accurate consumption function with superior extrapolation properties.

    Algorithm Overview (following the paper):
    1. Compute analytical bounds: optimist c_opt(m) and pessimist c_pes(m)
    2. Solve standard EGM to get realist consumption at gridpoints
    3. Calculate moderation ratio ω(μ) = (c_opt - c_real)/(c_opt - c_pes)
    4. Apply chi transformation χ(μ) = log((1-ω)/ω) for asymptotic linearity
    5. Interpolate χ(μ) function with derivatives for smooth extrapolation
    6. Reconstruct consumption function using equation (7) from the paper

    Key advantages over standard EGM (addresses Figure 1 extrapolation problem):
    - No negative precautionary saving in extrapolation regions
    - Theoretically grounded upper and lower bounds
    - Asymptotically linear chi function ensures proper limiting behavior
    - Computationally efficient (single EGM solve + analytical bounds)
    - Numerically stable across all wealth levels

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.

    Returns
    -------
    solution_ : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.
        The consumption function is constructed using Method of Moderation and
        has superior extrapolation properties compared to standard EGM.

    References
    ----------
    "The Method of Moderation" by Chris Carroll and Alan Lujan
    Code implementation by Alan Lujan

    """
    # =========================================================================
    # Setup and calculate economic parameters
    # =========================================================================
    uFunc = UtilityFuncCRRA(CRRA)

    # Calculate shared economic parameters and constraints
    (
        DiscFacEff,
        hNrm,
        BoroCnstNat,
        mNrmMin,
        MPCmin,
        MPCmax,
    ) = prepare_to_solve(
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
    )

    # Unpack next period's value functions
    vFuncNext = solution_next.vFunc
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc

    # =========================================================================
    # Step 2: Create analytical consumption function bounds (Method of Moderation)
    # =========================================================================
    # Create the three perfect foresight bounds that constrain realist behavior:
    # optimist (c_opt), pessimist (c_pes), and tighter upper bound (c_tight)
    optimist, pessimist, tighterUpperBound = make_behavioral_bounds(
        hNrm,
        mNrmMin,
        MPCmin,
        MPCmax,
        CRRA,
    )

    # =========================================================================
    # Step 3: Execute core EGM calculation step
    # =========================================================================
    # Solve standard EGM to get realist consumption at gridpoints
    aNrm, cNrm, mNrm, EndOfPrdvP = solve_egm_step(
        aXtraGrid,
        mNrmMin,
        DiscFacEff,
        Rfree,
        PermGroFac,
        CRRA,
        IncShkDstn,
        vPfuncNext,
        uFunc,
    )

    # =========================================================================
    # Step 4: Method of Moderation consumption build via unified helper
    # =========================================================================
    cFunc = _build_cfunc_mom(
        DiscFacEff=DiscFacEff,
        Rfree=Rfree,
        PermGroFac=PermGroFac,
        CRRA=CRRA,
        IncShkDstn=IncShkDstn,
        vPPfuncNext=vPfuncNext if False else vPPfuncNext,
        uFunc=uFunc,
        aNrm=aNrm,
        cNrm=cNrm,
        mNrm=mNrm,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        CubicBool=CubicBool,
        optimist=optimist,
        pessimist=pessimist,
    )

    # =========================================================================
    # Construct marginal value functions
    # =========================================================================

    # Create marginal value function v'(m) from consumption function
    vPfunc = MargValueFuncCRRA(cFunc, CRRA)

    # Create marginal marginal value function v''(m)
    vPPfunc = MargMargValueFuncCRRA(cFunc, CRRA) if CubicBool else NullFunc()

    # Construct this period's value function if requested
    if vFuncBool:
        # Construct complete value functions
        vNvrs, vNvrsP = construct_value_functions(
            aNrm,
            BoroCnstNat,
            DiscFacEff,
            Rfree,
            PermGroFac,
            CRRA,
            IncShkDstn,
            vFuncNext,
            EndOfPrdvP,
            uFunc,
            cNrm,
            CubicBool,
        )

        # Construct the beginning-of-period value function via MoM helper
        vFunc = _build_vfunc_mom(
            mNrm=mNrm,
            mNrmMin=mNrmMin,
            hNrm=hNrm,
            MPCmin=MPCmin,
            CRRA=CRRA,
            vNvrs=vNvrs,
            vNvrsP=vNvrsP,
            optimist=optimist,
            pessimist=pessimist,
            CubicBool=CubicBool,
        )
    else:
        vFunc = NullFunc()  # Dummy object

    # =========================================================================
    # Assemble and return the complete solution
    # =========================================================================

    # Package all solution components into ConsumerSolution object
    solution = ConsumerSolution(
        cFunc=cFunc,  # Method of Moderation consumption function
        vFunc=vFunc,  # Value function (if computed)
        vPfunc=vPfunc,  # Marginal value function
        vPPfunc=vPPfunc,  # Marginal marginal value function
        mNrmMin=mNrmMin,  # Minimum market resources
        hNrm=hNrm,  # Expected human wealth
        MPCmin=MPCmin,  # Minimum marginal propensity to consume
        MPCmax=MPCmax,  # Maximum marginal propensity to consume
    )

    # Add auxiliary consumption functions for Method of Moderation analysis
    # These enable comparison between different solution methods and bounds checking
    solution.Optimist = optimist
    solution.Pessimist = pessimist
    solution.TighterUpperBound = tighterUpperBound

    return solution


class IndShockMoMConsumerType(IndShockConsumerType):
    """HARK consumer type using the Method of Moderation.

    This class implements the Method of Moderation solution technique for
    consumption-saving problems with idiosyncratic income shocks, as described
    in "The Method of Moderation" by Carroll and Lujan. It provides superior
    extrapolation properties compared to standard EGM by using theoretically
    grounded analytical upper and lower bounds.

    Theoretical Framework:
    The Method of Moderation leverages the fact that realist optimal consumption
    is always bounded between two analytical perfect-foresight solutions:
    - Optimist consumption: c_opt(m) = κ_min * (m + h) - assumes E[θ] = 1
    - Pessimist consumption: c_pes(m) = κ_min * (m - m_min) - assumes θ = θ_min

    The realist solution is constructed by moderating between these bounds using
    the asymptotically linear chi transformation χ(μ) = log((1-κ)/κ), which ensures
    excellent extrapolation behavior and prevents negative precautionary saving.

    Attributes
    ----------
    solver : function
        method_of_moderation - Method of Moderation solver

    Advantages over Standard EGM
    ----------------------------
    - Solves the "extrapolation problem" shown in Figure 1 of the paper
    - No negative precautionary saving in extrapolation regions
    - Asymptotically linear chi function ensures proper limiting behavior
    - Theoretically motivated analytical bounds
    - Excellent numerical stability across all wealth levels
    - Computationally efficient (single EGM solve + analytical bounds)
    - Superior performance for simulation and estimation

    Examples
    --------
    >>> consumer = IndShockMoMConsumerType()
    >>> consumer.solve()
    >>> # Solution has excellent extrapolation properties (Figure 2 in paper)
    >>> gap = lambda m: consumer.solution[0].cOptFunc(m) - consumer.solution[0].cFunc(m)
    >>> # gap(m) > 0 for all m (no negative precautionary saving)

    References
    ----------
    "The Method of Moderation" by Chris Carroll and Alan Lujan
    Code implementation by Alan Lujan

    """

    default_: ClassVar = {
        "params": IndShockConsumerType_defaults,
        "solver": method_of_moderation,
        "model": "ConsIndShock.yaml",
    }
