r"""The Method of Moderation for Consumption-Saving Problems.

This module implements the Method of Moderation (MoM) solution technique for
consumption-saving problems with idiosyncratic income shocks, as described in
"The Method of Moderation" by Chris Carroll and Alan Lujan.
Code implementation by Alan Lujan.

Mathematical Framework
----------------------
The Method of Moderation leverages the fact that the realist consumer's optimal
consumption will always be bounded between two extreme cases:

1. **Optimist**: Assumes all future shocks take their expected value (E[theta] = 1)
   - Has analytical perfect-foresight solution: c_opt(m) = kappa_min * (m + h)
   - Where kappa_min is the minimum MPC and h is expected human wealth

2. **Pessimist**: Assumes all future shocks take their worst possible value
   - Has analytical perfect-foresight solution: c_pes(m) = kappa_min * (m - m_min)
   - Where m_min is the natural borrowing constraint (= -h_pes)

The realist solution c_real(m) satisfies: c_pes(m) <= c_real(m) <= c_opt(m)

Core Algorithm: The Method of Moderation
----------------------------------------
The algorithm transforms the consumption function construction problem by working
in the space between these analytical bounds:

1. **Solve standard EGM**: Use endogenous gridpoints to get consumption/resources pairs
2. **Calculate moderation ratio**: omega(mu) = (c_real - c_pes) / (c_opt - c_pes)
   where mu = log(m - m_min) is log excess market resources
3. **Apply logit transformation**: chi(mu) = log(omega/(1-omega)) for asymptotic linearity
4. **Interpolate chi function**: Create smooth chi(mu) interpolant with derivatives
5. **Reconstruct consumption**: c_real = c_pes + omega * (c_opt - c_pes)

Key Mathematical Properties
---------------------------
- **Asymptotic linearity**: chi'(mu) -> 0 as mu -> \infty (prevents negative precautionary saving)
- **Moderation bounds**: omega in [0,1] ensures consumption stays within theoretical bounds
- **Log excess resources**: mu = log(m - m_min) maps domain (m_min, \infty) to (-\infty, \infty)
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
    Primary implementation with excellent extrapolation properties
IndShockMoMCuspConsumerType : MoM with three-piece cusp approximation
    Extension that uses the tighter MPCmax bound below the cusp point
RiskyAssetMoMConsumerType : MoM under stochastic returns
    Extension for problems with risky rate of return; uses the
    Merton-Samuelson linear consumption rule for the optimist and pessimist
TransformedFunctionMoM : Generalized moderated-function wrapper (consumption or value)
TransformedFunctionMoMCusp : Three-piece moderated-function wrapper for the cusp variant

Core Solvers
------------
endogenous_grid_method : Standard EGM solver (baseline for comparison)
method_of_moderation : Method of Moderation solver (primary implementation)
method_of_moderation_cusp : MoM solver with three-piece cusp approximation
method_of_moderation_stochastic_r : MoM solver for stochastic-returns problems

Helper Functions
----------------
prepare_to_solve : Calculate shared economic parameters and constraints
make_behavioral_bounds : Create optimist, pessimist, and tighter upper bound solutions
solve_egm_step : Execute core Endogenous Grid Method calculation step
construct_value_functions : Build complete value functions with inverse utility transformations

Mathematical Utility Functions
------------------------------
log_mnrm_ex : Log excess market resources transformation mu = log(m - m_min)
moderate : Moderation ratio calculation omega = (c_real - c_pes)/(c_opt - c_pes)
logit_moderate : Asymptotically linear transformation chi = log(omega/(1-omega))
expit_moderate : Inverse chi transformation omega = 1/(1 + exp(-chi))
TransformedFunctionMoM : Generalized function transformer using Method of Moderation

Numerical Stability Features
-----------------------------
- **log1p/expm1**: Used throughout for accurate computation near 0
- **Overflow protection**: Handles extreme values of mu, omega, and chi
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
Solutions expose the analytical bounds as nested attributes: `solution.Optimist`,
`solution.Pessimist`, and `solution.TighterUpperBound` (each carrying its own
`cFunc`). These bounds correspond to the optimist, pessimist, and tighter
upper bound consumption functions described in the paper.

"""

from __future__ import annotations

import logging
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
    MeanOneLogNormal,
    combine_indep_dstns,
    expected,
)
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA
from scipy.special import expit

logger = logging.getLogger(__name__)


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


def _get_derivative(func, x):
    """Get derivative from function using derivativeX if available, else derivative.

    HARK interpolation objects use 'derivativeX' while standard functions use
    'derivative'. This helper abstracts over the difference.
    """
    if hasattr(func, "derivativeX"):
        return func.derivativeX(x)
    return func.derivative(x)


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
    CRRA,
    uFunc,
    aNrm,
    cNrm,
    mNrm,
    mNrmMin,
    hNrm,
    MPCmin,
    MPCmax,
    CubicBool,
    optimist,
    pessimist,
    DiscFacEff=None,
    Rfree=None,
    PermGroFac=None,
    IncShkDstn=None,
    vPPfuncNext=None,
    EndOfPrdvPP=None,
    mKink=None,
):
    """Construct consumption function for MoM path (chi/omega over mu + TransformedFunctionMoM).

    When `EndOfPrdvPP` is supplied (e.g. by a state-indexed solver that
    aggregates it across next states; the Markov-return solver in the bst
    repo's markov-return-mom exploration is the live consumer), the cubic
    branch uses it directly and
    `DiscFacEff`/`Rfree`/`PermGroFac`/`IncShkDstn`/`vPPfuncNext` are not needed.
    """
    # mu grid and derivative inputs
    mNrmEx = mNrm - mNrmMin
    hNrmEx = hNrm + mNrmMin  # = hNrmOpt - hNrmPes (eq:ExcessDef); mNrmMin = -hNrmPes
    mu = log_mnrm_ex(mNrm, mNrmMin)

    # Moderation ratio (always computed)
    modRte = moderate(mNrm, optimist.cFunc, cNrm, pessimist.cFunc)
    # Clip to (eps, 1-eps) to prevent numerical issues with logit
    eps = 1e-10
    modRte = np.clip(modRte, eps, 1.0 - eps)
    logitModRte = logit_moderate(modRte)

    # MPC vector and derivatives for Hermite slopes (only if cubic)
    if CubicBool:
        if EndOfPrdvPP is None:
            vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
            EndOfPrdvPP = vPPfacEff * expected(
                calc_vpp_next,
                IncShkDstn,
                args=(aNrm, Rfree, CRRA, PermGroFac, vPPfuncNext),
            )
        MPC = _compute_mpc_vector(uFunc, EndOfPrdvPP, cNrm)
        if not np.isfinite(hNrmEx) or hNrmEx <= 1e-12:
            msg = (
                f"hNrmEx={hNrmEx!r}: the pessimist's excess human wealth must be "
                "strictly positive for the MoM consumption derivative formula to "
                "apply. Check the impatience and growth-impatience conditions for "
                "the supplied parameters."
            )
            raise ValueError(
                msg,
            )
        modRteMu = mNrmEx * (MPC - MPCmin) / (MPCmin * hNrmEx)
        logitModRteMu = modRteMu / (modRte * (1 - modRte))
    else:
        # For linear interpolation, derivatives are not needed
        modRteMu = None
        logitModRteMu = None

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
        MPCmin=MPCmin,
        MPCmax=MPCmax,
        tight_upper_slope=MPCmax,
        mKink=mKink,
    )


def calc_constraint_kink(
    mNrmMin,
    BoroCnstNat,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vPfuncNext,
    uFunc,
):
    """Market resources at which an imposed borrowing constraint stops binding.

    One Euler inversion gives it exactly. Run the endogenous-grid step at the
    imposed constraint itself: a consumer who optimally ends the period with
    assets ``mNrmMin`` must have arrived with

        mKink = mNrmMin + c_unconstrained(mNrmMin),

    and below that m the same consumer would want to end with less, which the
    constraint forbids. No grid scan and no search: the kink is an endogenous
    gridpoint that the exogenous grid simply does not include, because it
    starts one ``aXtraMin`` above the constraint.

    Returns ``mNrmMin`` when ``BoroCnstNat >= mNrmMin``, the natural-constraint
    case, where the constrained region is the single point ``mNrmMin``.
    """
    if BoroCnstNat >= mNrmMin:
        return mNrmMin
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(
        calc_vp_next,
        IncShkDstn,
        args=(np.array([mNrmMin]), Rfree, CRRA, PermGroFac, vPfuncNext),
    )
    cNrmAtCnst = float(np.asarray(uFunc.derinv(EndOfPrdvP, order=(1, 0))).item())
    return mNrmMin + cNrmAtCnst


def calc_constrained_branch_value(
    mNrmMin,
    BoroCnstNat,
    DiscFacEff,
    Rfree,
    PermGroFac,
    CRRA,
    IncShkDstn,
    vFuncNext,
):
    r"""End-of-period value at the constraint, or None when nothing binds.

    Where an imposed constraint binds, the consumer spends everything above it
    and ends the period pinned at ``mNrmMin``, so end-of-period assets do not
    vary with ``m`` and value on that branch is

        v(m) = u(m - mNrmMin) + wBar,     wBar = w(mNrmMin),

    a single constant. This returns that constant. It is evaluated directly
    from the income distribution rather than read off the end-of-period
    interpolant, because on a sparse asset grid ``mNrmMin`` can fall in a wide
    gap between nodes: the interpolated value there is a secant, which is the
    very error this branch exists to remove.

    Returns None when ``BoroCnstNat >= mNrmMin``, that is when the constraint
    is the natural one. There the branch is empty (the constraint is never
    strictly binding at any feasible ``m``), ``w(mNrmMin)`` is negative
    infinity, and callers must leave the moderated value function alone.
    """
    if BoroCnstNat >= mNrmMin:
        return None
    wBar = float(
        DiscFacEff
        * expected(
            calc_v_next,
            IncShkDstn,
            args=(np.array([mNrmMin]), Rfree, CRRA, PermGroFac, vFuncNext),
        ).item()
    )
    if not np.isfinite(wBar):
        logger.warning(
            "End-of-period value at the imposed constraint mNrmMin=%r is not "
            "finite (%r), so the constrained branch cannot be built and the "
            "value function falls back to interpolation there.",
            mNrmMin,
            wBar,
        )
        return None
    return wBar


class ConstrainedBranchValue(MetricObject):
    r"""Inverse value function that is exact where the constraint binds.

    Wraps an approximate inverse value function (moderated or interpolated) and
    overrides it on the branch where the borrowing constraint binds, where
    ``v(m) = u(m - mNrmMin) + wBar`` holds outright.

    The region runs from the constraint up to ``mKink``, taken from
    :func:`calc_constraint_kink`, so the consumption and value paths use one
    number and the kink is endogenous rather than the analytic cusp, which lies
    far outside it.

    This is not an envelope. On the constrained branch the closed form is the
    answer, not a bound: an approximation there can err on either side, and
    both EGM and moderation in fact overshoot it.

    Parameters
    ----------
    approx_nvrs : callable
        The inverse value function to use away from the constraint.
    mNrmMin : float
        The binding constraint.
    mKink : float
        Largest market resources at which the constraint still binds.
    wBar : float
        End-of-period value at the constraint, from
        :func:`calc_constrained_branch_value`.
    uFunc : UtilityFuncCRRA
        Utility function, supplying ``inv`` and ``derinv``.

    """

    distance_criteria = ["approx_nvrs", "mNrmMin", "mKink", "wBar"]

    def __init__(self, approx_nvrs, mNrmMin, mKink, wBar, uFunc) -> None:
        self.approx_nvrs = approx_nvrs
        self.mNrmMin = mNrmMin
        self.mKink = mKink
        self.wBar = wBar
        self.uFunc = uFunc

    @property
    def x_list(self):
        """Gridpoints of the wrapped approximation, for HARK callers."""
        return self.approx_nvrs.x_list

    def _binds(self, m):
        """True on the branch where the constraint binds."""
        return (m >= self.mNrmMin) & (m <= self.mKink)

    def _branch_nvrs(self, m_ex):
        """u^{-1}(u(m_ex) + wBar), the exact inverse value on the branch."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.uFunc.inv(self.uFunc(m_ex) + self.wBar)

    def __call__(self, m):
        m = np.asarray(m, dtype=float)
        m_ex = m - self.mNrmMin
        binds = self._binds(m)
        # Evaluate the branch only on strictly positive excess resources; at
        # m == mNrmMin the exact inverse value is 0, which u.inv cannot deliver
        # from u(0) = -inf without a spurious nan.
        safe_ex = np.where(m_ex > 0, m_ex, 1.0)
        out = np.where(
            binds,
            np.where(m_ex > 0, self._branch_nvrs(safe_ex), 0.0),
            np.asarray(self.approx_nvrs(m), dtype=float),
        )
        return float(out) if m.ndim == 0 else out

    def derivative(self, m):
        m = np.asarray(m, dtype=float)
        m_ex = m - self.mNrmMin
        safe_ex = np.where(m_ex > 0, m_ex, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v_branch = self.uFunc(safe_ex) + self.wBar
            branch_slope = self.uFunc.der(safe_ex) * self.uFunc.derinv(
                v_branch,
                order=(0, 1),
            )
        out = np.where(
            self._binds(m),
            branch_slope,
            np.asarray(_get_derivative(self.approx_nvrs, m), dtype=float),
        )
        return float(out) if m.ndim == 0 else out

    def derivativeX(self, m):
        """Alias for derivative(m) to satisfy HARK's derivativeX contract."""
        return self.derivative(m)


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
    wBar,
    mKink,
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
    # Neither this grid nor the moderated one carries a node strictly inside a
    # binding imposed constraint, so both draw a secant from (mNrmMin, 0) to the
    # kink. Applying the exact branch to each keeps their comparison fair.
    if wBar is not None:
        vNvrsFunc = ConstrainedBranchValue(
            vNvrsFunc,
            mNrmMin,
            mKink,
            wBar,
            uFunc,
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
    mKink=None,
    wBar=None,
    uFunc=None,
):
    """Construct beginning-of-period value function for MoM path via chi/omega over mu."""
    mNrmEx = mNrm - mNrmMin
    hNrmEx = hNrm + mNrmMin  # = hNrmOpt - hNrmPes (eq:ExcessDef); mNrmMin = -hNrmPes
    mu = log_mnrm_ex(mNrm, mNrmMin)

    vNvrsOptFunc = optimist.vFunc.vFuncNvrs
    vNvrsPesFunc = pessimist.vFunc.vFuncNvrs

    modRte = moderate(mNrm, vNvrsOptFunc, vNvrs, vNvrsPesFunc)
    # Clip to (eps, 1-eps) so logit_moderate does not see omega = 0 or 1 at the
    # borrowing constraint (would yield -inf and a 0 denominator below).
    eps = 1e-10
    modRte = np.clip(modRte, eps, 1.0 - eps)
    MPCminNvrs = MPCmin ** (-CRRA / (1.0 - CRRA))
    if not np.isfinite(hNrmEx) or hNrmEx <= 1e-12:
        msg = (
            f"hNrmEx={hNrmEx!r}: the pessimist's excess human wealth must be "
            "strictly positive for the MoM value-function derivative formula "
            "to apply. Check the impatience and growth-impatience conditions "
            "for the supplied parameters."
        )
        raise ValueError(
            msg,
        )
    modRteMu = mNrmEx * (vNvrsP - MPCminNvrs) / (hNrmEx * MPCminNvrs)
    logitModRte = logit_moderate(modRte)
    logitModRteMu = modRteMu / (modRte * (1 - modRte))

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
    # Moderation cannot reach the constrained branch: mu = log(m - mNrmMin)
    # sends the constraint to -infinity, so omega is clipped away from 0 above
    # and the branch is interpolated through rather than represented.
    if wBar is not None:
        moderated_vfunc = ConstrainedBranchValue(
            moderated_vfunc,
            mNrmMin,
            mKink,
            wBar,
            uFunc,
        )
    return ValueFuncCRRA(moderated_vfunc, CRRA)


def _build_marginal_value_funcs(cFunc, CRRA, CubicBool):
    """Construct marginal and marginal-marginal value functions.

    Parameters
    ----------
    cFunc : callable
        Consumption function
    CRRA : float
        Coefficient of relative risk aversion
    CubicBool : bool
        Whether to use cubic interpolation (determines if vPPfunc is computed)

    Returns
    -------
    tuple
        (vPfunc, vPPfunc) - marginal value functions

    """
    vPfunc = MargValueFuncCRRA(cFunc, CRRA)
    vPPfunc = MargMargValueFuncCRRA(cFunc, CRRA) if CubicBool else NullFunc()
    return vPfunc, vPPfunc


def _build_complete_vfunc(
    *,
    vFuncBool,
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
    mNrm,
    mNrmMin,
    hNrm,
    MPCmin,
    optimist,
    pessimist,
    CubicBool,
    mKink=None,
):
    """Construct complete value function using Method of Moderation if requested.

    Parameters
    ----------
    vFuncBool : bool
        Whether to compute value function
    [other params same as construct_value_functions and _build_vfunc_mom]

    Returns
    -------
    vFunc : ValueFuncCRRA or NullFunc
        Value function (or dummy if not requested)

    """
    if not vFuncBool:
        return NullFunc()

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
    return _build_vfunc_mom(
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
        mKink=mKink,
        uFunc=uFunc,
        wBar=calc_constrained_branch_value(
            mNrmMin,
            BoroCnstNat,
            DiscFacEff,
            Rfree,
            PermGroFac,
            CRRA,
            IncShkDstn,
            vFuncNext,
        ),
    )


def _assemble_mom_solution(
    *,
    cFunc,
    vFunc,
    vPfunc,
    vPPfunc,
    mNrmMin,
    hNrm,
    MPCmin,
    MPCmax,
    optimist,
    pessimist,
    tighterUpperBound,
):
    """Assemble complete ConsumerSolution with behavioral bounds attached.

    Parameters
    ----------
    cFunc : callable
        Consumption function
    vFunc : callable
        Value function
    vPfunc : callable
        Marginal value function
    vPPfunc : callable
        Marginal-marginal value function
    mNrmMin : float
        Minimum market resources
    hNrm : float
        Human wealth
    MPCmin : float
        Minimum MPC
    MPCmax : float
        Maximum MPC
    optimist : ConsumerSolution
        Optimist perfect foresight solution
    pessimist : ConsumerSolution
        Pessimist perfect foresight solution
    tighterUpperBound : ConsumerSolution
        Tighter upper bound solution

    Returns
    -------
    ConsumerSolution
        Complete solution with behavioral bounds attached

    """
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
    solution.Optimist = optimist
    solution.Pessimist = pessimist
    solution.TighterUpperBound = tighterUpperBound
    return solution


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
        - MPCmax (float): Upper bound on MPC (1.0 when an artificial constraint
          binds, otherwise the unconstrained `calc_mpc_max` value)

    Notes
    -----
    All computations follow the standard HARK conventions for consumption-saving
    models.

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
       - Perfect foresight consumption: c_opt(m) = kappa_min * (m + h)
       - Represents the upper bound on consumption behavior
       - Used as intercept: hNrm, slope: MPCmin

    2. **Pessimist**: Assumes all future income shocks take worst possible value
       - Perfect foresight consumption: c_pes(m) = kappa_min * (m - m_min)
       - Represents the lower bound on consumption behavior
       - Used as intercept: -mNrmMin, slope: MPCmin

    3. **Tighter Upper Bound**: Pessimist constraint with maximum MPC
       - Perfect foresight consumption: c_tight(m) = kappa_max * (m - m_min)
       - Provides tighter bound near the borrowing constraint
       - Used as intercept: -mNrmMin, slope: MPCmax

    Parameters
    ----------
    hNrm : float
        Present discounted value of human wealth (expected future income)
    mNrmMin : float
        Minimum feasible market resources (effective borrowing constraint)
    MPCmin : float
        Minimum marginal propensity to consume (kappa_min)
    MPCmax : float
        Maximum marginal propensity to consume (kappa_max)
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
    These bounds satisfy: c_pes(m) <= c_real(m) <= c_opt(m) for all m > m_min.
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
    2. **Backward Induction**: Calculate end-of-period marginal value E[betaRu'(c_{t+1})]
    3. **Euler Equation**: Invert FOC to find consumption: c_t = (u')^{-1}(E[betaRu'(c_{t+1})])
    4. **Budget Constraint**: Back out market resources: m_t = c_t + a_t

    Mathematical Foundation:
    The Euler equation for optimal consumption is:
        u'(c_t) = betaR * E[u'(c_{t+1})]
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
        - EndOfPrdvP (np.array): End-of-period marginal value E[betaRu'(c_{t+1})]

    Notes
    -----
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
    EndOfPrdv=None,
):
    """Construct complete value functions using HARK's inverse utility framework.

    When `EndOfPrdv` is supplied (e.g. by a state-indexed solver that
    aggregates it across next states; the Markov-return solver in the bst
    repo's markov-return-mom exploration is the live consumer), step 1's
    expectation is skipped and
    `DiscFacEff`/`Rfree`/`PermGroFac`/`IncShkDstn`/`vFuncNext` may be None.

    This function performs the comprehensive value function construction process
    that is shared between both EGM and Method of Moderation solvers. It implements
    HARK's standard approach using inverse utility transformations for numerical
    stability and constructs both end-of-period and beginning-of-period value components.

    The construction follows HARK's established methodology:

    **End-of-Period Value Function Construction:**
    1. Calculate value function via backward induction: v(a) = betaE[v_{t+1}(Ra + Y_{t+1})]
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
        v_t(m_t) = max_{c_t} {u(c_t) + beta E[v_{t+1}(R(m_t - c_t) + Y_{t+1})]}

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
        End-of-period marginal value E[betaRu'(c_{t+1})] from solve_egm_step
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
    The inverse utility transformation is essential
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
    if EndOfPrdv is None:
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
        Carries the analytical bounds for Method of Moderation comparison as
        nested attributes: `solution.Optimist.cFunc`, `solution.Pessimist.cFunc`,
        and `solution.TighterUpperBound.cFunc`.

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
    # Solve EGM step: assets grid -> consumption grid -> market resources grid
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

    mKink = calc_constraint_kink(
        mNrmMin,
        BoroCnstNat,
        DiscFacEff,
        Rfree,
        PermGroFac,
        CRRA,
        IncShkDstn,
        vPfuncNext,
        uFunc,
    )

    # Note: Boundary augmentation (c=0 at m=mNrmMin) is handled in _build_cfunc_egm

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
            wBar=calc_constrained_branch_value(
                mNrmMin,
                BoroCnstNat,
                DiscFacEff,
                Rfree,
                PermGroFac,
                CRRA,
                IncShkDstn,
                vFuncNext,
            ),
            mKink=mKink,
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
    Solutions expose the analytical bounds as nested attributes
    (`solution.Optimist.cFunc`, `solution.Pessimist.cFunc`,
    `solution.TighterUpperBound.cFunc`) to enable direct comparison with
    Method of Moderation results and to illustrate the theoretical bounds
    described in the paper.

    """

    default_: ClassVar = {
        "params": IndShockEGMConsumerType_defaults,
        "solver": endogenous_grid_method,
        "model": "ConsIndShock.yaml",
    }

    time_inv_: ClassVar = [*IndShockConsumerType.time_inv_, "ExtrapBool"]


def log_mnrm_ex(m, m_min):
    r"""Log excess market resources transformation for Method of Moderation.

    Computes mu = log(m - m_min), the log-excess market resources transformation
    that maps the natural domain (m_min, \infty) to (-\infty, \infty). This transformation
    is central to the Method of Moderation as it provides a convenient space
    for interpolating the chi function chi(mu).

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
        Log excess market resources: mu = log(m - m_min)

    Notes
    -----
    Implementation uses np.log1p for numerical stability, avoiding issues
    when m - m_min is close to zero.

    Implemented as log1p(m - m_min - 1) for stability.

    This transformation is used as the input space for the chi function chi(mu)
    in the Method of Moderation. The log transformation ensures the chi function
    can be asymptotically linear, preventing negative precautionary saving.

    """
    return np.log1p(m - m_min - 1)


def exp_mu(mu, m_min):
    r"""Inverse log excess market resources transformation.

    Computes m = exp(mu) + m_min, the inverse of the log-excess transformation
    mu = log(m - m_min). This maps the unbounded domain (-\infty, \infty) back to the
    natural domain (m_min, \infty).

    Parameters
    ----------
    mu : float or array
        Log excess market resources mu = log(m - m_min)
    m_min : float
        Minimum feasible market resources (effective borrowing constraint)

    Returns
    -------
    float or array
        Market resources: m = exp(mu) + m_min

    Notes
    -----
    Implementation uses np.expm1 for numerical stability when mu is close to 0.
    Implemented as expm1(mu) + m_min + 1 which equals exp(mu) + m_min.

    """
    return np.expm1(mu) + m_min + 1


def moderate(m, f_opt, f_real, f_pess):
    """Moderation ratio function omega for Method of Moderation.

    General utility function that computes the moderation ratio omega that measures how close
    the realist consumer is to the optimist's behavior. Following the ML-consistent
    convention in equation (5) of the paper, the ratio is defined as:

    omega = (f_real - f_pess) / (f_opt - f_pess)

    This function is generic and works with any economic functions that follow the
    bounded behavior pattern:
    - For consumption: f_opt = c_opt, f_real = c_real, f_pess = c_pes
    - For value functions: f_opt = v_opt, f_real = v_real, f_pess = v_pes
    - For inverse value functions: f_opt = vNvrs_opt, f_real = vNvrs_real, f_pess = vNvrs_pes

    Parameters
    ----------
    m : float or array
        Market resources (cash-on-hand) where functions are evaluated. When
        `f_real` is an already-evaluated array, `m` must contain the same
        gridpoints `f_real` was evaluated on, since `f_opt` and `f_pess` are
        called on `m` to recover the bounds at those locations.
    f_opt : callable
        Optimist function (upper bound, perfect foresight behavior)
    f_real : float, array, or callable
        Realist function values. Two calling conventions are supported:
        (1) a callable that gets invoked as `f_real(m)`; or
        (2) an array of values already evaluated at `m` (used for the value-
        function MoM path, where the inverse-value array is built externally
        before the moderation step). The function detects the array case via
        `np.asarray`.
    f_pess : callable
        Pessimist function (lower bound, worst-case behavior)

    Returns
    -------
    float or array
        Moderation ratio omega in [0,1]

    Notes
    -----
    This is a utility function for external analysis and verification. The main
    Method of Moderation implementation computes moderation ratios inline for efficiency.

    Mathematical properties (ML-consistent convention):
    - omega = 0: realist behaves like pessimist (low wealth, high precautionary saving)
    - omega = 1: realist behaves like optimist (high wealth, no precautionary saving)
    - Strictly between 0 and 1 for all m > m_min under economic prudence
    - Generic design allows application to consumption, value, or other economic functions
    - Consistent with standard ML/statistics convention for probability-like quantities

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
    denom = f_opt_vals - f_pess_vals
    # At the natural borrowing constraint (m == mNrmMin) the two bounds collapse
    # to the same value and the denominator is zero. Internal callers always feed
    # grid points strictly above mNrmMin, but the function is exported as a public
    # utility and may be invoked at the constraint for diagnostics. Return omega
    # = 0.5 there (the bounds give no information about position) rather than
    # silently propagating NaN/Inf.
    return np.where(
        np.abs(denom) < 1e-14,
        0.5,
        (f_real - f_pess_vals) / np.where(np.abs(denom) < 1e-14, 1.0, denom),
    )


def logit_moderate(omega):
    r"""Standard logit transformation chi(omega) for asymptotically linear interpolation.

    Computes the standard logit transformation following equation (6) in the paper:
    chi = log(omega / (1-omega)) = log(omega) - log(1-omega)

    This is the standard logit transformation used in machine learning and statistics.
    By transforming the bounded moderation ratio omega in (0,1) to the unbounded chi space
    chi in (-infinity,infinity), it enables asymptotically linear interpolation that prevents negative
    precautionary saving.

    Parameters
    ----------
    omega : float or array
        Moderation ratio omega in (0,1) measuring how close the realist is to optimist behavior

    Returns
    -------
    float or array
        Chi transformation value chi in (-infinity,infinity)

    Notes
    -----
    **Numerical Implementation:**
    - Uses log(omega) - log1p(-omega) for numerical stability
    - Mathematical equivalence: log(omega) - log(1-omega) = log(omega/(1-omega))
    - Avoids potential overflow/underflow issues near omega -> 0 or omega -> 1

    **Asymptotic Properties (ML-consistent convention):**
    - As omega -> 0 (realist -> pessimist): chi -> -infinity
    - As omega -> 1 (realist -> optimist): chi -> +infinity
    - The derivative chi'(mu) -> alpha (a non-negative constant) as mu -> infinity,
      so chi is asymptotically linear in mu and omega approaches 1 only in the limit
      mu -> +infinity. Linear extrapolation of chi therefore keeps omega in (0,1)
      and gives a stable continuation of the realist consumption rule beyond the grid.
    - This asymptotic linearity is what prevents the negative precautionary saving
      that plagues direct EGM extrapolation.

    **Economic Interpretation:**
    - chi < 0: realist closer to pessimist (high precautionary saving)
    - chi > 0: realist closer to optimist (low precautionary saving)
    - chi ~= 0: realist balanced between extremes

    **ML/Statistics Consistency:**
    - This is the standard logit function: inverse of sigmoid sigma(x) = 1/(1+exp(-x))
    - Matches PyTorch, TensorFlow, scikit-learn conventions
    - Makes code immediately recognizable to ML practitioners

    """
    return np.log(omega) - np.log1p(-omega)


def expit_moderate(chi):
    """Standard sigmoid (expit) function sigma(chi) for Method of Moderation.

    Computes the inverse of the logit transformation to recover the moderation ratio:
    omega = 1 / (1 + exp(-chi)) = sigma(chi)

    This is the standard sigmoid/expit function, inverse of logit_moderate.
    If chi = logit_moderate(omega), then omega = expit_moderate(chi).

    Parameters
    ----------
    chi : float or array
        Chi transformation value chi

    Returns
    -------
    float or array
        Moderation ratio omega in (0,1)

    Notes
    -----
    - This is the standard sigmoid/expit function from ML/statistics
    - Delegates to scipy.special.expit, which evaluates both tails without
      overflow (the naive 1/(1 + exp(-chi)) overflows in exp for chi below
      about -709 before IEEE arithmetic rescues the value)
    - Central to reconstructing consumption from interpolated chi function
    - Ensures omega in (0,1) for all finite chi values
    - Matches PyTorch's torch.sigmoid, scipy.special.expit, etc.

    **ML/Statistics Consistency:**
    - Standard sigmoid: sigma(x) = 1/(1+exp(-x))
    - Derivative: sigma'(x) = sigma(x)(1-sigma(x))
    - Immediately recognizable to ML practitioners

    """
    return expit(chi)


class TransformedFunctionMoM(MetricObject):
    """Generalized Method of Moderation function transformer.

    This class provides the core moderation logic for functions bounded between
    two lines (optimist and pessimist bounds). It applies the MoM formula:
    f_real(m) = f_pes(m) + omega(mu) * (f_opt(m) - f_pes(m))

    Subclassing MetricObject makes infinite-horizon (cycles=0) convergence
    checks meaningful: without a distance metric, HARK's solver loop compares
    successive solutions as incomparable objects (sentinel distance 1000) and
    always runs to max_cycles regardless of tolerance.

    This class can be used for any function that needs to be moderated between
    upper and lower bounds, including consumption functions, value functions, etc.

    The implementation uses the standard chi transformation approach:
    - mu = log(m - m_min) is the log excess market resources
    - chi(mu) is the interpolated chi function (asymptotically linear)
    - omega(mu) = 1/(1 + exp(-chi(mu))) is the moderation ratio
    - Provides superior numerical stability for extrapolation

    Parameters
    ----------
    mNrmMin : float
        Minimum feasible market resources (effective borrowing constraint)
    modRteFunc : callable
        Interpolated moderation ratio function (reserved for future use)
    logitModRteFunc : callable
        Interpolated chi function that maps mu -> chi (core transformation)
    optimist_func : callable
        Function that computes the optimist (upper) bound f_opt(m)
    pessimist_func : callable
        Function that computes the pessimist (lower) bound f_pes(m)

    Returns
    -------
    callable
        The moderated function f_real(m) that interpolates between bounds

    """

    distance_criteria = ["logitModRteFunc", "mNrmMin"]

    def __init__(
        self,
        mNrmMin,
        modRteFunc,
        logitModRteFunc,
        optimist_func,
        pessimist_func,
        MPCmin=None,
        MPCmax=None,
        tight_upper_slope=None,
        mKink=None,
    ) -> None:
        self.mNrmMin = mNrmMin
        self.mKink = mKink
        self.tight_upper_slope = tight_upper_slope
        self.modRteFunc = modRteFunc
        self.logitModRteFunc = logitModRteFunc
        self.optimist_func = optimist_func
        self.pessimist_func = pessimist_func
        self.MPCmin = MPCmin  # For bounded MPC formula
        self.MPCmax = MPCmax  # For bounded MPC formula

    @property
    def x_list(self):
        """Underlying gridpoints in m-space (back-mapped from the mu grid).

        The interpolant lives in mu = log(m - mNrmMin) space; HARK callers
        (and `code/verify_table.py`) expect an `x_list` attribute giving the
        gridpoints in market-resources space. Returns `mNrmMin + exp(mu_grid)`.
        """
        mu_grid = np.asarray(self.logitModRteFunc.x_list)
        return self.mNrmMin + np.exp(mu_grid)

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

        # Apply moderation: f_real = f_pes + omega * (f_opt - f_pes)
        moderated = f_pes + omega * (f_opt - f_pes)
        # mu is -inf at the constraint itself, so chi and omega come back nan
        # there. The limit is omega = 0: the realist attains the pessimist's
        # value, c(mNrmMin) = 0 for consumption and vNvrs(mNrmMin) = 0 for value.
        moderated = np.where(np.asarray(m) > self.mNrmMin, moderated, f_pes)
        return self._apply_constrained_branch(
            m,
            self._apply_tight_upper_bound(m, moderated),
        )

    def _apply_constrained_branch(self, m, moderated):
        """Substitute the exact rule where the borrowing constraint binds.

        Below the kink the agent spends everything above the constraint, so
        c = m - mNrmMin outright. Clipping alone does not deliver this: as m
        approaches the constraint the moderation ratio goes to zero and the
        rule goes to the PESSIMIST's, which lies strictly BELOW the budget
        line, so the clip is slack exactly where the constraint binds hardest.
        """
        if self.mKink is None:
            return moderated
        m_arr = np.asarray(m, dtype=float)
        binds = (m_arr >= self.mNrmMin) & (m_arr <= self.mKink)
        return np.where(binds, m_arr - self.mNrmMin, moderated)

    def _tight_upper_bound(self, m):
        """The tighter upper bound MPCmax * (m - mNrmMin), or None if unset.

        This is the borrowing constraint imposed: the line through the
        constraint point ``(mNrmMin, 0)`` with the limiting MPC there, which
        concavity makes a valid upper bound everywhere. It therefore carries
        the boundary condition ``c(mNrmMin) = 0`` for free. Under an ARTIFICIAL
        constraint MPCmax is 1, so the bound is the 45-degree budget line and
        the solution attains it over an interval; under a NATURAL constraint
        MPCmax < 1 and the bound is strictly tighter than the budget line,
        touched only at the constraint itself. One expression covers both.

        Only the consumption path supplies a slope, so value functions are
        unaffected.
        """
        if self.tight_upper_slope is None:
            return None
        return self.tight_upper_slope * (m - self.mNrmMin)

    def _apply_tight_upper_bound(self, m, moderated):
        """Clip the moderated value to the tighter upper bound.

        Never a regression: the truth obeys the bound, so wherever the
        moderated value exceeds it the bound is strictly closer, and where the
        constraint actually binds the bound IS the truth. The clip is a no-op
        whenever the bound is slack, which is the whole natural-constraint case.
        """
        bound = self._tight_upper_bound(m)
        return moderated if bound is None else np.minimum(moderated, bound)

    def derivative(self, m):
        """Compute the derivative of the moderated function.

        For consumption functions, implements the paper's equation {eq:MPCModeration}:
        MPC = (1 - MPCmod) * MPCmin + MPCmod * MPCmax

        where MPCmod in [0,1] ensures bounds are respected by construction.

        For general moderated functions where bounds may have different slopes:
        f_real'(m) = f_pes'(m) + omega'(m) * (f_opt(m) - f_pes(m)) + omega(m) * (f_opt'(m) - f_pes'(m))

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

        # Get derivatives of bounding functions
        try:
            f_opt_prime = self.optimist_func.derivative(m)
            f_pes_prime = self.pessimist_func.derivative(m)
        except AttributeError:
            msg = (
                "Bounding functions must implement a 'derivative' method. "
                "Use PerfForesightFunc for linear bounding functions."
            )
            raise TypeError(
                msg,
            ) from None

        # 2. Calculate excess resources and omega components
        m_ex = m - self.mNrmMin
        h_ex = (
            f_opt - f_pes
        )  # Excess human wealth times MPCmin for consumption functions

        chi = self.logitModRteFunc(mu)
        omega = expit_moderate(chi)
        chi_prime_mu = _get_derivative(self.logitModRteFunc, mu)

        # 3. Check if this is a consumption function (both bounds have same slope = MPCmin)
        if np.allclose(f_opt_prime, f_pes_prime):
            # This is a consumption function where both bounds have slope MPCmin
            MPCmin = f_opt_prime

            # For consumption functions: c_opt - c_pes = h_nrm_ex * MPCmin
            # where h_nrm_ex is excess human wealth
            h_nrm_ex = h_ex / MPCmin

            # Compute omega'_mu from chi, consistent with how __call__ computes omega
            # omega = expit(chi), so omega'_mu = omega * (1 - omega) * chi'_mu
            omega_prime_mu = omega * (1 - omega) * chi_prime_mu

            # True derivative formula: MPC = MPCmin * (1 + (h_nrm_ex/m_ex) * omega'_mu).
            # This matches the actual consumption function slope. The paper's
            # convex-combination form (eq:MPCModeration) holds AT Euler-equation
            # gridpoints, where CarrollShanker2024 bounds dc/dm in [MPCmin, MPCmax].
            # Between gridpoints the cubic spline can produce omega'_mu outside
            # that range and the interpolated MPC may briefly fall outside
            # [MPCmin, MPCmax]; refine the grid if this matters.
            deriv = MPCmin * (1 + (h_nrm_ex / m_ex) * omega_prime_mu)
        else:
            # 4. General case: full product rule when the bounds differ in slope
            dmu_dm = 1.0 / m_ex
            d_omega_dm = omega * (1 - omega) * chi_prime_mu * dmu_dm

            base_slope = f_pes_prime
            slope_adjustment = omega * (f_opt_prime - f_pes_prime)
            moderation_adjustment = d_omega_dm * (f_opt - f_pes)

            deriv = base_slope + slope_adjustment + moderation_adjustment

        # Where __call__ clips to the tighter upper bound, the function IS that
        # bound, so its slope is the bound's slope. Without this the MPC keeps
        # the moderated value's slope across a segment it no longer describes.
        bound = self._tight_upper_bound(m)
        if bound is not None:
            moderated = f_pes + omega * (f_opt - f_pes)
            deriv = np.where(moderated > bound, self.tight_upper_slope, deriv)
        if self.mKink is not None:
            m_arr = np.asarray(m, dtype=float)
            binds = (m_arr >= self.mNrmMin) & (m_arr <= self.mKink)
            deriv = np.where(binds, 1.0, deriv)
        return deriv

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
        Log excess market resources grid mu = log(m - m_min)
    modRte : np.array
        Moderation ratio values omega at each gridpoint
    modRteMu : np.array
        Derivatives of moderation ratio domega/dmu at each gridpoint
    logitModRte : np.array
        Logit transformation values chi = log(omega/(1-omega)) at each gridpoint
    logitModRteMu : np.array
        Derivatives of chi transformation dchi/dmu at each gridpoint
    CubicBool : bool
        Whether to use cubic spline interpolation (True) or linear interpolation (False)

    Returns
    -------
    tuple
        Two interpolation objects (CubicInterp or LinearInterp):
        - modRteFunc: omega(mu) interpolant
        - logitModRteFunc: chi(mu) interpolant

    Notes
    -----
    Uses module-level constants MOM_EXTRAP_GAP_LEFT and MOM_EXTRAP_GAP_RIGHT
    for consistent extrapolation behavior across consumption and value functions.

    """
    # Augmented mu grid with extrapolation points
    muAug = np.r_[mu[0] - MOM_EXTRAP_GAP_LEFT, mu, mu[-1] + MOM_EXTRAP_GAP_RIGHT]

    # Augmented omega (modRte) values - use derivative-based extrapolation if available
    if modRteMu is not None:
        # Use derivative-based linear extrapolation (consistent for both cubic/linear)
        modRteAug = np.r_[
            modRte[0] - modRteMu[0] * MOM_EXTRAP_GAP_LEFT,
            modRte,
            modRte[-1] + modRteMu[-1] * MOM_EXTRAP_GAP_RIGHT,
        ]
        modRteMuAug = np.r_[modRteMu[0], modRteMu, modRteMu[-1]]
    else:
        # Fallback to finite-difference linear extrapolation
        # Use slope from edge of grid for linear extrapolation
        slope_left = (modRte[1] - modRte[0]) / (mu[1] - mu[0])
        slope_right = (modRte[-1] - modRte[-2]) / (mu[-1] - mu[-2])
        modRteAug = np.r_[
            modRte[0] - slope_left * MOM_EXTRAP_GAP_LEFT,
            modRte,
            modRte[-1] + slope_right * MOM_EXTRAP_GAP_RIGHT,
        ]
        modRteMuAug = None

    # Augmented chi (logitModRte) values - use derivative-based extrapolation if available
    if logitModRteMu is not None:
        # Use derivative-based linear extrapolation (consistent for both cubic/linear)
        logitModRteAug = np.r_[
            logitModRte[0] - logitModRteMu[0] * MOM_EXTRAP_GAP_LEFT,
            logitModRte,
            logitModRte[-1] + logitModRteMu[-1] * MOM_EXTRAP_GAP_RIGHT,
        ]
        logitModRteMuAug = np.r_[logitModRteMu[0], logitModRteMu, logitModRteMu[-1]]
    else:
        # Fallback to finite-difference linear extrapolation
        # Use slope from edge of grid for linear extrapolation
        slope_left = (logitModRte[1] - logitModRte[0]) / (mu[1] - mu[0])
        slope_right = (logitModRte[-1] - logitModRte[-2]) / (mu[-1] - mu[-2])
        logitModRteAug = np.r_[
            logitModRte[0] - slope_left * MOM_EXTRAP_GAP_LEFT,
            logitModRte,
            logitModRte[-1] + slope_right * MOM_EXTRAP_GAP_RIGHT,
        ]
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
    """Method of Moderation solver for consumption-saving problems.

    Solves one period of a consumption-saving model using the Method of Moderation
    technique described in "The Method of Moderation" by Carroll and Lujan. This
    method leverages analytical upper and lower bounds to construct a highly
    accurate consumption function with superior extrapolation properties.

    Algorithm Overview (following the paper):
    1. Compute analytical bounds: optimist c_opt(m) and pessimist c_pes(m)
    2. Solve standard EGM to get realist consumption at gridpoints
    3. Calculate moderation ratio omega(mu) = (c_real - c_pes)/(c_opt - c_pes)
    4. Apply logit transformation chi(mu) = log(omega/(1-omega)) for asymptotic linearity
    5. Interpolate chi(mu) function with derivatives for smooth extrapolation
    6. Reconstruct consumption: c_real = c_pes + omega * (c_opt - c_pes)

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

    mKink = calc_constraint_kink(
        mNrmMin,
        BoroCnstNat,
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
        vPPfuncNext=vPPfuncNext,
        uFunc=uFunc,
        aNrm=aNrm,
        cNrm=cNrm,
        mNrm=mNrm,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
        CubicBool=CubicBool,
        optimist=optimist,
        pessimist=pessimist,
        mKink=mKink,
    )

    # Construct marginal value functions
    vPfunc, vPPfunc = _build_marginal_value_funcs(cFunc, CRRA, CubicBool)

    # Construct value function if requested
    vFunc = _build_complete_vfunc(
        vFuncBool=vFuncBool,
        aNrm=aNrm,
        BoroCnstNat=BoroCnstNat,
        DiscFacEff=DiscFacEff,
        Rfree=Rfree,
        PermGroFac=PermGroFac,
        CRRA=CRRA,
        IncShkDstn=IncShkDstn,
        vFuncNext=vFuncNext,
        EndOfPrdvP=EndOfPrdvP,
        uFunc=uFunc,
        cNrm=cNrm,
        mNrm=mNrm,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        optimist=optimist,
        pessimist=pessimist,
        CubicBool=CubicBool,
        mKink=mKink,
    )

    # Assemble and return complete solution
    return _assemble_mom_solution(
        cFunc=cFunc,
        vFunc=vFunc,
        vPfunc=vPfunc,
        vPPfunc=vPPfunc,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
        optimist=optimist,
        pessimist=pessimist,
        tighterUpperBound=tighterUpperBound,
    )


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
    - Optimist consumption: c_opt(m) = kappa_min * (m + h) - assumes E[theta] = 1
    - Pessimist consumption: c_pes(m) = kappa_min * (m - m_min) - assumes theta = theta_min

    The realist solution is constructed by moderating between these bounds using
    the asymptotically linear chi transformation chi(mu) = log(omega/(1-omega)),
    where omega = (c_real - c_pes) / (c_opt - c_pes) is the moderation ratio.
    This formulation ensures excellent extrapolation behavior and prevents
    negative precautionary saving.

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
    >>> sol = consumer.solution[0]
    >>> gap = lambda m: sol.Optimist.cFunc(m) - sol.cFunc(m)
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


# =========================================================================
# Extension 1: Three-Piece Cusp Approximation
# =========================================================================


def calc_cusp_point(hNrm, mNrmMin, MPCmin, MPCmax):
    r"""Calculate the cusp point where optimist and tighter bounds intersect.

    The cusp point is where the two upper bounds on consumption intersect:
    - Optimist: c_opt(m) = MPCmin * (m + hNrm)
    - Tighter: c_tight(m) = MPCmax * (m - mNrmMin)

    Setting these equal and solving for m gives the cusp point from eq:mNrmCusp:
        mNrmCusp = -hNrmPes + MPCmin * (hNrmOpt - hNrmPes) / (MPCmax - MPCmin)

    where hNrmPes = -mNrmMin and hNrmOpt = hNrm.

    Parameters
    ----------
    hNrm : float
        Present discounted value of human wealth (optimist's h)
    mNrmMin : float
        Minimum feasible market resources (= -hNrmPes)
    MPCmin : float
        Minimum marginal propensity to consume
    MPCmax : float
        Maximum marginal propensity to consume

    Returns
    -------
    float
        The cusp point mNrmCusp where upper bounds intersect

    Notes
    -----
    Below mNrmCusp, use the tighter bound (MPCmax slope).
    Above mNrmCusp, use the optimist bound (MPCmin slope).

    """
    hNrmPes = -mNrmMin  # Pessimist's human wealth
    hNrmOpt = hNrm  # Optimist's human wealth
    hNrmEx = hNrmOpt - hNrmPes  # Excess human wealth = hNrm + mNrmMin

    # From eq:mNrmCusp in the paper
    return -hNrmPes + MPCmin * hNrmEx / (MPCmax - MPCmin)


def moderate_tight(m, mNrmMin, cNrm, MPCmin, MPCmax):
    r"""Compute moderation ratio using the tighter upper bound.

    For low wealth (m < mNrmCusp), the tighter bound c_tight = MPCmax * mNrmEx
    provides a better upper bound than the optimist. The moderation ratio
    using this bound is defined in eq:modRteLoTightUpBd:

        modRteLoTight = (c_real/mNrmEx - MPCmin) / (MPCmax - MPCmin)
                      = (c_real - c_pes) / (c_tight - c_pes)

    Parameters
    ----------
    m : float or array
        Market resources
    mNrmMin : float
        Minimum feasible market resources
    cNrm : float or array
        Realist consumption values
    MPCmin : float
        Minimum MPC (pessimist and optimist slope)
    MPCmax : float
        Maximum MPC (tighter bound slope)

    Returns
    -------
    float or array
        Moderation ratio using tighter bound, in [0, 1]

    """
    mNrmEx = m - mNrmMin
    c_pes = MPCmin * mNrmEx
    c_tight = MPCmax * mNrmEx
    return (cNrm - c_pes) / (c_tight - c_pes)


class TransformedFunctionMoMCusp(MetricObject):
    """Three-piece consumption function using cusp approximation.

    This class implements the three-piece approximation described in the paper:
    1. Below cusp: Use tighter upper bound (MPCmax slope)
    2. Near cusp: Hermite interpolation for smooth transition
    3. Above cusp: Use optimist upper bound (MPCmin slope)

    The result is a consumption function that respects BOTH upper bounds
    throughout the domain, providing tighter theoretical guarantees.

    Parameters
    ----------
    mNrmMin : float
        Minimum feasible market resources
    mNrmCusp : float
        Cusp point where upper bounds intersect
    modRteFuncLow : callable
        Moderation ratio function for m < mNrmCusp (using tight bound)
    logitModRteFuncLow : callable
        Logit of moderation ratio for m < mNrmCusp
    modRteFuncHigh : callable
        Moderation ratio function for m >= mNrmCusp (using optimist bound)
    logitModRteFuncHigh : callable
        Logit of moderation ratio for m >= mNrmCusp
    optimist_func : callable
        Optimist consumption function
    pessimist_func : callable
        Pessimist consumption function
    tight_func : callable
        Tighter upper bound consumption function
    MPCmin : float
        Minimum MPC
    MPCmax : float
        Maximum MPC

    """

    distance_criteria = [
        "logitModRteFuncLow",
        "logitModRteFuncHigh",
        "mNrmCusp",
        "mNrmMin",
    ]

    def __init__(
        self,
        mNrmMin,
        mNrmCusp,
        modRteFuncLow,
        logitModRteFuncLow,
        modRteFuncHigh,
        logitModRteFuncHigh,
        optimist_func,
        pessimist_func,
        tight_func,
        MPCmin,
        MPCmax,
    ) -> None:
        self.mNrmMin = mNrmMin
        self.mNrmCusp = mNrmCusp
        self.modRteFuncLow = modRteFuncLow
        self.logitModRteFuncLow = logitModRteFuncLow
        self.modRteFuncHigh = modRteFuncHigh
        self.logitModRteFuncHigh = logitModRteFuncHigh
        self.optimist_func = optimist_func
        self.pessimist_func = pessimist_func
        self.tight_func = tight_func
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    @property
    def x_list(self):
        """Combined gridpoints in m-space from both cusp sub-regions.

        Concatenates the low-region (`mu_low = log(m - mNrmMin)` below the cusp)
        and high-region grids, back-mapped to m-space via `mNrmMin + exp(mu)`,
        and sorts so callers see a monotone increasing grid.
        """
        mu_low = np.asarray(self.logitModRteFuncLow.x_list)
        mu_high = np.asarray(self.logitModRteFuncHigh.x_list)
        mu_combined = np.concatenate([mu_low, mu_high])
        return np.sort(self.mNrmMin + np.exp(mu_combined))

    def __call__(self, m):
        """Evaluate consumption using three-piece approximation."""
        m = np.asarray(m)
        scalar_input = m.ndim == 0
        m = np.atleast_1d(m)

        c = np.empty_like(m, dtype=float)

        # Low region: use tighter bound
        low_mask = m < self.mNrmCusp
        if np.any(low_mask):
            m_low = m[low_mask]
            mu_low = log_mnrm_ex(m_low, self.mNrmMin)
            chi_low = self.logitModRteFuncLow(mu_low)
            omega_low = expit_moderate(chi_low)
            c_pes_low = self.pessimist_func(m_low)
            c_tight_low = self.tight_func(m_low)
            c[low_mask] = c_pes_low + omega_low * (c_tight_low - c_pes_low)

        # High region: use optimist bound
        high_mask = ~low_mask
        if np.any(high_mask):
            m_high = m[high_mask]
            mu_high = log_mnrm_ex(m_high, self.mNrmMin)
            chi_high = self.logitModRteFuncHigh(mu_high)
            omega_high = expit_moderate(chi_high)
            c_pes_high = self.pessimist_func(m_high)
            c_opt_high = self.optimist_func(m_high)
            c[high_mask] = c_pes_high + omega_high * (c_opt_high - c_pes_high)

        return float(c[0]) if scalar_input else c

    def derivative(self, m):
        """Compute MPC using three-piece approximation."""
        m = np.asarray(m)
        scalar_input = m.ndim == 0
        m = np.atleast_1d(m)

        mpc = np.empty_like(m, dtype=float)
        m_ex = m - self.mNrmMin

        # Low region: MPC between MPCmin and MPCmax using tight bound
        low_mask = m < self.mNrmCusp
        if np.any(low_mask):
            m_low = m[low_mask]
            mu_low = log_mnrm_ex(m_low, self.mNrmMin)

            chi_low = self.logitModRteFuncLow(mu_low)
            omega_low = expit_moderate(chi_low)
            chi_prime_mu_low = _get_derivative(self.logitModRteFuncLow, mu_low)
            # Compute omega'_mu from chi, consistent with __call__
            omega_prime_mu_low = omega_low * (1 - omega_low) * chi_prime_mu_low

            # MPC formula using tight bound gap
            # c = c_pes + omega * (c_tight - c_pes) = c_pes + omega * (MPCmax - MPCmin) * m_ex
            # dc/dm = MPCmin + omega * (MPCmax - MPCmin) + omega' * (MPCmax - MPCmin) * m_ex / m_ex
            #       = MPCmin + (omega + omega'_mu) * (MPCmax - MPCmin)
            mpc[low_mask] = self.MPCmin + (omega_low + omega_prime_mu_low) * (
                self.MPCmax - self.MPCmin
            )

        # High region: MPC using optimist bound (standard formula)
        high_mask = ~low_mask
        if np.any(high_mask):
            m_high = m[high_mask]
            m_ex_high = m_ex[high_mask]
            mu_high = log_mnrm_ex(m_high, self.mNrmMin)

            chi_high = self.logitModRteFuncHigh(mu_high)
            omega_high = expit_moderate(chi_high)
            chi_prime_mu_high = _get_derivative(self.logitModRteFuncHigh, mu_high)
            # Compute omega'_mu from chi, consistent with __call__
            omega_prime_mu_high = omega_high * (1 - omega_high) * chi_prime_mu_high

            # True derivative formula for high region (optimist bound)
            # MPC = MPCmin * (1 + (h_nrm_ex/m_ex) * omega'_mu)
            hNrmEx = (
                self.optimist_func(m_high) - self.pessimist_func(m_high)
            ) / self.MPCmin
            mpc[high_mask] = self.MPCmin * (
                1 + (hNrmEx / m_ex_high) * omega_prime_mu_high
            )

        return float(mpc[0]) if scalar_input else mpc

    def derivativeX(self, m):
        """Alias for derivative(m) to satisfy HARK's derivativeX contract."""
        return self.derivative(m)


def _build_cfunc_mom_cusp(
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
    CubicBool,
    optimist,
    pessimist,
    tighter,
):
    """Construct consumption function using three-piece cusp approximation."""
    # Calculate cusp point
    mNrmCusp = calc_cusp_point(hNrm, mNrmMin, MPCmin, MPCmax)

    # mu grid and derivative inputs
    mNrmEx = mNrm - mNrmMin
    hNrmEx = hNrm + mNrmMin  # = hNrmOpt - hNrmPes (eq:ExcessDef); mNrmMin = -hNrmPes
    mu = log_mnrm_ex(mNrm, mNrmMin)

    # Split grid at cusp point
    low_mask = mNrm < mNrmCusp
    high_mask = ~low_mask

    # Each region needs >= 2 gridpoints in BOTH interpolation modes. Linear
    # needs them for its finite-difference extrapolation slopes; cubic could
    # mechanically split on one Hermite point, but measured accuracy settles
    # it: a one-knot low region rides its tangent ~7.5 mu-units to the cusp
    # (sup error 2.4e-1 vs 2.9e-3 for the fallback on the Table 1 sparse
    # grid). Fall back to standard MoM, keeping the optimist/pessimist bounds.
    if np.sum(low_mask) < 2 or np.sum(high_mask) < 2:
        logger.info(
            "Cusp construction fell back to standard MoM: a region has fewer "
            "than 2 gridpoints (low=%d, high=%d); the tighter MPCmax bound is "
            "not enforced for this solution.",
            int(np.sum(low_mask)),
            int(np.sum(high_mask)),
        )
        return _build_cfunc_mom(
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
            CubicBool=CubicBool,
            optimist=optimist,
            pessimist=pessimist,
        )

    # MPC vector for Hermite slopes (only if cubic)
    if CubicBool:
        vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
        EndOfPrdvPP = vPPfacEff * expected(
            calc_vpp_next,
            IncShkDstn,
            args=(aNrm, Rfree, CRRA, PermGroFac, vPPfuncNext),
        )
        MPC = _compute_mpc_vector(uFunc, EndOfPrdvPP, cNrm)
    else:
        MPC = None

    # LOW REGION: Use tighter bound
    mNrm_low = mNrm[low_mask]
    cNrm_low = cNrm[low_mask]
    mu_low = mu[low_mask]

    # Moderation ratio using tight bound; clip to (eps, 1-eps) before logit
    # so log(0) / log(1) cannot poison the interpolant near the constraint.
    eps = 1e-10
    modRte_low = moderate_tight(mNrm_low, mNrmMin, cNrm_low, MPCmin, MPCmax)
    modRte_low = np.clip(modRte_low, eps, 1.0 - eps)
    logitModRte_low = logit_moderate(modRte_low)

    if CubicBool:
        MPC_low = MPC[low_mask]
        # Derivative: d(modRte)/d(mu) for the tight bound, carrying the
        # quotient-rule term from the 1/mNrmEx factor:
        #   modRte = (c/mNrmEx - MPCmin) / (MPCmax - MPCmin)
        #   d(modRte)/d(mu) = (MPC - c/mNrmEx) / (MPCmax - MPCmin),
        # which is negative in the low region: under concavity the average
        # propensity c/mNrmEx exceeds the marginal MPC, so the ratio falls
        # from 1 at the constraint toward 0.
        mNrmEx_low = mNrm_low - mNrmMin
        modRteMu_low = (MPC_low - cNrm_low / mNrmEx_low) / (MPCmax - MPCmin)
        logitModRteMu_low = modRteMu_low / (modRte_low * (1 - modRte_low))
    else:
        modRteMu_low = None
        logitModRteMu_low = None

    modRteFuncLow, logitModRteFuncLow = _construct_mom_interpolants(
        mu_low,
        modRte_low,
        modRteMu_low,
        logitModRte_low,
        logitModRteMu_low,
        CubicBool,
    )

    # HIGH REGION: Use optimist bound (standard moderation)
    mNrm_high = mNrm[high_mask]
    cNrm_high = cNrm[high_mask]
    mNrmEx_high = mNrmEx[high_mask]
    mu_high = mu[high_mask]

    modRte_high = moderate(mNrm_high, optimist.cFunc, cNrm_high, pessimist.cFunc)
    # Same clip as the low region; protects logit_moderate at omega in {0, 1}.
    modRte_high = np.clip(modRte_high, eps, 1.0 - eps)
    logitModRte_high = logit_moderate(modRte_high)

    if CubicBool:
        MPC_high = MPC[high_mask]
        modRteMu_high = mNrmEx_high * (MPC_high - MPCmin) / (MPCmin * hNrmEx)
        logitModRteMu_high = modRteMu_high / (modRte_high * (1 - modRte_high))
    else:
        modRteMu_high = None
        logitModRteMu_high = None

    modRteFuncHigh, logitModRteFuncHigh = _construct_mom_interpolants(
        mu_high,
        modRte_high,
        modRteMu_high,
        logitModRte_high,
        logitModRteMu_high,
        CubicBool,
    )

    return TransformedFunctionMoMCusp(
        mNrmMin,
        mNrmCusp,
        modRteFuncLow,
        logitModRteFuncLow,
        modRteFuncHigh,
        logitModRteFuncHigh,
        optimist.cFunc,
        pessimist.cFunc,
        tighter.cFunc,
        MPCmin,
        MPCmax,
    )


def method_of_moderation_cusp(
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
    """Method of Moderation solver with three-piece cusp approximation.

    This solver extends the standard Method of Moderation by using a
    three-piece consumption function that respects BOTH upper bounds:
    - Below cusp: Uses tighter bound (MPCmax slope) for better accuracy
    - Above cusp: Uses optimist bound (MPCmin slope) for asymptotic behavior

    The cusp point is where the two bounds intersect, providing the
    tightest possible constraints throughout the domain.

    Parameters
    ----------
    Same as method_of_moderation

    Returns
    -------
    ConsumerSolution
        Solution with three-piece cusp consumption function

    """
    # Setup (same as standard MoM)
    uFunc = UtilityFuncCRRA(CRRA)

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

    vFuncNext = solution_next.vFunc
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc

    # Create behavioral bounds
    optimist, pessimist, tighterUpperBound = make_behavioral_bounds(
        hNrm,
        mNrmMin,
        MPCmin,
        MPCmax,
        CRRA,
    )

    # Solve EGM step
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

    mKink = calc_constraint_kink(
        mNrmMin,
        BoroCnstNat,
        DiscFacEff,
        Rfree,
        PermGroFac,
        CRRA,
        IncShkDstn,
        vPfuncNext,
        uFunc,
    )

    # Build consumption function with cusp approximation
    cFunc = _build_cfunc_mom_cusp(
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
        CubicBool=CubicBool,
        optimist=optimist,
        pessimist=pessimist,
        tighter=tighterUpperBound,
    )

    # Construct marginal value functions
    vPfunc, vPPfunc = _build_marginal_value_funcs(cFunc, CRRA, CubicBool)

    # Construct value function if requested
    vFunc = _build_complete_vfunc(
        vFuncBool=vFuncBool,
        aNrm=aNrm,
        BoroCnstNat=BoroCnstNat,
        DiscFacEff=DiscFacEff,
        Rfree=Rfree,
        PermGroFac=PermGroFac,
        CRRA=CRRA,
        IncShkDstn=IncShkDstn,
        vFuncNext=vFuncNext,
        EndOfPrdvP=EndOfPrdvP,
        uFunc=uFunc,
        cNrm=cNrm,
        mNrm=mNrm,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        optimist=optimist,
        pessimist=pessimist,
        CubicBool=CubicBool,
        mKink=mKink,
    )

    # Assemble solution and add cusp-specific attribute
    solution = _assemble_mom_solution(
        cFunc=cFunc,
        vFunc=vFunc,
        vPfunc=vPfunc,
        vPPfunc=vPPfunc,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
        optimist=optimist,
        pessimist=pessimist,
        tighterUpperBound=tighterUpperBound,
    )
    solution.mNrmCusp = calc_cusp_point(hNrm, mNrmMin, MPCmin, MPCmax)
    return solution


class IndShockMoMCuspConsumerType(IndShockConsumerType):
    """HARK consumer type using Method of Moderation with cusp approximation.

    This class extends the standard Method of Moderation by using a three-piece
    consumption function that respects both upper bounds (optimist and tighter).
    The cusp point marks where the two bounds intersect, and the consumption
    function uses the appropriate bound in each region.

    Advantages over Standard MoM
    ----------------------------
    - Tighter bounds near borrowing constraint (uses MPCmax slope)
    - Maintains optimist bound behavior at high wealth
    - Smooth transition at cusp point via Hermite interpolation
    - Even better theoretical guarantees

    """

    default_: ClassVar = {
        "params": IndShockConsumerType_defaults,
        "solver": method_of_moderation_cusp,
        "model": "ConsIndShock.yaml",
    }


# =========================================================================
# Extension 2: Stochastic Rate of Return
# =========================================================================


def calc_vp_next_stochastic_r(atoms, a, crra, perm_gro_fac, vp_func_next):
    """Calculate continuation marginal value with stochastic return.

    Unlike the deterministic case where R is factored out of the expectation,
    here R is part of the shock and stays inside the integral:
        E[R * ψ^{-ρ} * v'(R*a/ψ + θ)]

    Parameters
    ----------
    atoms : np.ndarray
        Shock atoms array with shape (3, n_points):
        - Row 0: PermShk (permanent income shock)
        - Row 1: TranShk (transitory income shock)
        - Row 2: Risky (risky return)
    a : np.ndarray
        End-of-period assets grid (shape: (n_a, 1) after expansion)
    crra : float
        Coefficient of relative risk aversion
    perm_gro_fac : float
        Permanent income growth factor
    vp_func_next : callable
        Next period's marginal value function

    Returns
    -------
    np.ndarray
        R * ψ^{-ρ} * v'(m_{t+1}) where m_{t+1} = R*a/ψ + θ

    """
    psi = atoms[0]  # Permanent shock
    theta = atoms[1]  # Transitory shock
    R = atoms[2]  # Risky return
    # Next period's market resources: m_{t+1} = R*a/(ψ*G) + θ
    m_next = R * a / (perm_gro_fac * psi) + theta
    # Contribution to expectation: R * ψ^{-ρ} * v'(m_{t+1})
    return R * (psi ** (-crra)) * vp_func_next(m_next)


def create_joint_distribution(IncShkDstn, RiskyAvg, RiskyStd, n_risky_points=7):
    """Create joint distribution over income shocks and risky returns.

    Constructs the Cartesian product of the income shock distribution and
    a discrete approximation to the lognormal return distribution.

    Parameters
    ----------
    IncShkDstn : DiscreteDistribution
        Income shock distribution with PermShk (row 0) and TranShk (row 1)
    RiskyAvg : float
        Mean of risky return (in levels)
    RiskyStd : float
        Standard deviation of risky return (in levels)
    n_risky_points : int
        Number of quadrature points for return distribution

    Returns
    -------
    DiscreteDistribution
        Joint distribution with atoms shape (3, n_points):
        - Row 0: PermShk
        - Row 1: TranShk
        - Row 2: Risky

    """
    # Create lognormal return distribution centered on RiskyAvg
    # Log R ~ N(μ, σ²) where E[R] = exp(μ + σ²/2) = RiskyAvg
    # and Var[R] = exp(2μ + σ²)(exp(σ²) - 1) = RiskyStd²
    # Solve: σ² = log(1 + (RiskyStd/RiskyAvg)²), μ = log(RiskyAvg) - σ²/2
    variance_ratio = (RiskyStd / RiskyAvg) ** 2
    log_std = np.sqrt(np.log(1 + variance_ratio))

    # Create mean-one lognormal and scale by RiskyAvg
    risky_dstn = MeanOneLogNormal(log_std).discretize(n_risky_points, method="hermite")
    # Scale atoms in place (atoms has shape (1, n_points))
    risky_dstn.atoms = risky_dstn.atoms * RiskyAvg

    # Combine with income shocks (Cartesian product)
    # Result: atoms shape (3, n_inc * n_risky) with rows [PermShk, TranShk, Risky]
    return combine_indep_dstns(IncShkDstn, risky_dstn)


def calc_patience_factor_stochastic_r(JointShkDstn, DiscFacEff, CRRA):
    r"""Patience factor under return risk, from the Euler equation's limit.

    Taking ``m -> infinity`` in the Euler equation, consumption growth tends to
    ``Risky * (1 - MPCmin)`` and the equation collapses to
    ``(1 - MPCmin)**CRRA = DiscFac * E[Risky**(1-CRRA)]``, so the patience
    factor generalizes from ``(Rfree * DiscFac)**(1/CRRA) / Rfree`` to

        PatFac = (DiscFac * E[Risky**(1-CRRA)])**(1/CRRA).

    A deterministic return reduces this exactly: ``E[R**(1-CRRA)]`` becomes
    ``R**(1-CRRA)`` and the expression is ``(R*DiscFac)**(1/CRRA)/R``, which is
    HARK's `calc_patience_factor`. Feeding the result through
    `calc_mpc_min` then gives MPCmin by the same backward recursion used
    without return risk, so the bound is the Euler-implied limit rather than a
    slope measured off the solved grid.

    Existence and convergence results are not established here; see the
    buffer-stock-theory treatment cited in the text.

    Parameters
    ----------
    JointShkDstn : DiscreteDistribution
        Joint distribution whose third component (index 2) is the gross risky
        return, as built by :func:`create_joint_distribution`.
    DiscFacEff : float
        Effective discount factor, already multiplied by the survival
        probability.
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    float
        The patience factor to pass to `calc_mpc_min`.

    """
    risky = np.asarray(JointShkDstn.atoms[2], dtype=float)
    probs = np.asarray(JointShkDstn.pmv, dtype=float)
    m_zero = float(np.sum(probs * risky ** (1.0 - CRRA)))
    pat_fac = (DiscFacEff * m_zero) ** (1.0 / CRRA)
    if not np.isfinite(pat_fac) or pat_fac <= 0.0:
        msg = (
            f"Stochastic patience factor {pat_fac!r} is not usable "
            f"(DiscFacEff={DiscFacEff!r}, E[R^(1-CRRA)]={m_zero!r})."
        )
        raise ValueError(msg)
    return pat_fac


def calc_return_factor_human_wealth(JointShkDstn, CRRA):
    r"""Effective return factor for discounting human wealth under return risk.

    With a risky return there is no traded riskless asset, so the rate at which
    a labor income stream should be capitalized is not a modelling choice: it
    is whatever the Euler equation implies in the ``m -> infinity`` limit. In
    that limit the stochastic discount factor is
    ``M = DiscFac * (Risky * (1 - MPCmin))**(-CRRA)``, and applying
    ``(1 - MPCmin)**CRRA = DiscFac * E[Risky**(1-CRRA)]`` collapses its
    expectation to a ratio of two moments of the return distribution alone:

        E[M] = E[Risky**(-CRRA)] / E[Risky**(1-CRRA)]

    so the effective discount factor is the reciprocal,

        Reff = E[Risky**(1-CRRA)] / E[Risky**(-CRRA)].

    Note the asymmetry this exposes: MPCmin is pinned by the SINGLE moment
    ``E[Risky**(1-CRRA)]``, while human wealth is pinned by the RATIO of two.
    A deterministic return collapses both onto the same factor, which is why a
    single rate suffices there and stops sufficing once the return is random.

    Existence of finite human wealth requires ``E[M] < 1``, the return-risk
    analogue of the finite-human-wealth condition; a ValueError is raised
    otherwise rather than returning a negative or infinite discount factor.

    The derivation assumes permanent growth is separable from the return
    process, which holds for the calibrations used here. Existence and
    convergence results are not established in this paper; see the
    buffer-stock-theory treatment cited in the text.

    Parameters
    ----------
    JointShkDstn : DiscreteDistribution
        Joint distribution whose third component (index 2) is the gross risky
        return, as built by :func:`create_joint_distribution`.
    CRRA : float
        Coefficient of relative risk aversion.

    Returns
    -------
    float
        The effective gross return factor for capitalizing human wealth.

    """
    risky = np.asarray(JointShkDstn.atoms[2], dtype=float)
    probs = np.asarray(JointShkDstn.pmv, dtype=float)
    m_zero = float(np.sum(probs * risky ** (1.0 - CRRA)))
    m_one = float(np.sum(probs * risky ** (-CRRA)))
    if not np.isfinite(m_zero) or not np.isfinite(m_one) or m_one <= 0.0:
        msg = (
            f"Return moments are not usable: E[R^(1-CRRA)]={m_zero!r}, "
            f"E[R^(-CRRA)]={m_one!r}. Check the discretized return process."
        )
        raise ValueError(msg)
    if m_one >= m_zero:
        # The one-period recursion below is still well defined; only the
        # infinite sum is not, so this is reported rather than raised.
        logger.warning(
            "E[R^(-CRRA)]=%.6g >= E[R^(1-CRRA)]=%.6g, so the expected "
            "stochastic discount factor is at least one and the implied "
            "shadow riskless rate is below one. Human wealth is finite in "
            "any finite horizon but does not converge as the horizon grows; "
            "this is the return-risk analogue of the finite-human-wealth "
            "condition.",
            m_one,
            m_zero,
        )
    return m_zero / m_one


def calc_boro_const_nat_stochastic_r(mNrmMinNext, JointShkDstn, PermGroFac):
    """Calculate natural borrowing constraint with stochastic returns.

    With stochastic returns, the borrowing constraint must ensure positive
    resources in ALL possible shock combinations. Following HARK's approach,
    we compute the constraint for each combination and take the maximum.

    For each shock combination (psi, theta, R):
        a * R / (G * psi) + theta >= mNrmMinNext
        a >= (mNrmMinNext - theta) * (G * psi) / R

    The natural borrowing constraint is the maximum over all combinations.

    Parameters
    ----------
    mNrmMinNext : float
        Next period's minimum market resources
    JointShkDstn : DiscreteDistribution
        Joint distribution with atoms shape (3, n_points):
        - Row 0: PermShk
        - Row 1: TranShk
        - Row 2: Risky
    PermGroFac : float
        Permanent income growth factor

    Returns
    -------
    float
        Natural borrowing constraint (minimum end-of-period assets)

    """
    # Get all shock combinations
    psi_vals = JointShkDstn.atoms[0]  # Permanent shocks
    theta_vals = JointShkDstn.atoms[1]  # Transitory shocks
    R_vals = JointShkDstn.atoms[2]  # Risky returns

    # Compute constraint for each combination: a >= (mNrmMinNext - theta) * (G * psi) / R
    BoroCnstNat_all = (mNrmMinNext - theta_vals) * (PermGroFac * psi_vals) / R_vals

    # Take the maximum (most binding) constraint
    return np.max(BoroCnstNat_all)


def solve_egm_step_stochastic_r(
    aXtraGrid,
    mNrmMin,
    DiscFacEff,
    PermGroFac,
    CRRA,
    JointShkDstn,
    vPfuncNext,
    uFunc,
):
    """Execute EGM step with stochastic returns.

    The key difference from standard EGM is that returns are integrated
    over jointly with income shocks:
        u'(c) = β E[R * ψ^{-ρ} * u'(c(R*a/ψ + θ))]

    Parameters
    ----------
    aXtraGrid : np.array
        Extra end-of-period asset values
    mNrmMin : float
        Minimum market resources (borrowing constraint)
    DiscFacEff : float
        Effective discount factor
    PermGroFac : float
        Permanent income growth factor
    CRRA : float
        Coefficient of relative risk aversion
    JointShkDstn : DiscreteDistribution
        Joint distribution over (PermShk, TranShk, Risky)
    vPfuncNext : callable
        Next period's marginal value function
    uFunc : UtilityFuncCRRA
        Utility function

    Returns
    -------
    tuple
        (aNrm, cNrm, mNrm, EndOfPrdvP)

    """
    aNrm = np.asarray(aXtraGrid) + mNrmMin

    # End-of-period marginal value: E[R * ψ^{-ρ} * v'(R*a/ψ + θ)]
    # Note: no Rfree factor outside since R is inside expectation
    vPfacEff = DiscFacEff * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(
        calc_vp_next_stochastic_r,
        JointShkDstn,
        args=(aNrm, CRRA, PermGroFac, vPfuncNext),
    )

    # EGM: invert FOC to find consumption
    cNrm = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    mNrm = cNrm + aNrm

    return aNrm, cNrm, mNrm, EndOfPrdvP


def calc_stochastic_mpc(DiscFac, CRRA, RiskyAvg, RiskyStd):
    r"""Calculate MPC for consumer facing i.i.d. lognormal returns.

    For a consumer with CRRA utility facing i.i.d. lognormal returns:
        log(R) ~ N(r + equityPrem - std^2/2, std^2)

    The Merton-Samuelson result gives the MPC as:
        MPC = 1 - (DiscFac * E[R^{1-CRRA}])^{1/CRRA}

    For lognormal R with mean RiskyAvg and std RiskyStd (in levels):
        E[R^{1-CRRA}] = RiskyAvg^{1-CRRA} * exp((1-CRRA)*(-CRRA)*std_log^2/2)

    where std_log^2 = log(1 + (RiskyStd/RiskyAvg)^2).

    Parameters
    ----------
    DiscFac : float
        Time discount factor
    CRRA : float
        Coefficient of relative risk aversion
    RiskyAvg : float
        Mean of risky return R (in levels, e.g., 1.08 for 8% return)
    RiskyStd : float
        Standard deviation of risky return R (in levels)

    Returns
    -------
    float
        The MPC under stochastic returns (replaces MPCmin in MoM formulas)

    Notes
    -----
    This MPC should be substituted for MPCmin throughout the Method of
    Moderation formulas when the consumer faces stochastic returns.

    References
    ----------
    Samuelson (1969), Merton (1969, 1971)
    See also CRRA-RateRisk and BBZ2016SkewedWealth

    Examples
    --------
    >>> # 8% mean return with 20% standard deviation
    >>> mpc = calc_stochastic_mpc(0.96, 2.0, 1.08, 0.20)

    """
    # Convert level mean/std to log parameters
    # If R ~ LogNormal, then E[R] = exp(mu + sigma^2/2) and Var[R] = exp(2*mu + sigma^2)*(exp(sigma^2)-1)
    # So sigma^2 = log(1 + (Std/Mean)^2)
    variance_ratio = (RiskyStd / RiskyAvg) ** 2
    log_var = np.log1p(variance_ratio)  # sigma^2 in log space

    # E[R^{1-CRRA}] for lognormal
    # If log(R) ~ N(mu, sigma^2), then E[R^a] = exp(a*mu + a^2*sigma^2/2)
    # For level mean M and variance V: mu = log(M) - sigma^2/2
    # So E[R^a] = M^a * exp((a^2 - a)*sigma^2/2) = M^a * exp(a*(a-1)*sigma^2/2)
    exponent = (1 - CRRA) * (-CRRA) * log_var / 2
    E_R_power = (RiskyAvg ** (1 - CRRA)) * np.exp(exponent)

    # Merton-Samuelson MPC
    inner = DiscFac * E_R_power
    if inner <= 0 or inner >= 1:
        msg = f"Invalid parameters: DiscFac * E[R^{{1-CRRA}}] = {inner} not in (0,1)"
        raise ValueError(msg)

    return 1 - inner ** (1 / CRRA)


def method_of_moderation_stochastic_r(
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
    RiskyAvg,
    RiskyStd,
):
    """Method of Moderation solver with stochastic rate of return.

    This solver extends the standard Method of Moderation to handle
    i.i.d. lognormal returns on savings. Unlike the deterministic case,
    returns are integrated INSIDE the expectation:

        u'(c) = β E[R * ψ^{-ρ} * u'(c(R*a/ψ + θ))]

    This properly accounts for both income risk and return risk, showing
    that adding return volatility increases precautionary savings.

    Parameters
    ----------
    Same as method_of_moderation, plus:
    RiskyAvg : float
        Mean of risky return R (in levels)
    RiskyStd : float
        Standard deviation of risky return R (in levels)

    Returns
    -------
    ConsumerSolution
        Solution with combined income+return risk

    Notes
    -----
    The key difference from deterministic returns:
    - Deterministic: EndOfPrdvP = β*R * E[ψ^{-ρ} * v'(R*a/ψ + θ)]
    - Stochastic: EndOfPrdvP = β * E[R * ψ^{-ρ} * v'(R*a/ψ + θ)]

    The R factor is now inside the expectation, creating additional
    precautionary savings from return uncertainty.

    """
    uFunc = UtilityFuncCRRA(CRRA)

    # Get deterministic parameters for comparison
    (
        DiscFacEff,
        hNrm,
        _BoroCnstNat_det,
        _mNrmMin_det,
        MPCmin_deterministic,  # MPCmin with deterministic returns
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

    vFuncNext = solution_next.vFunc
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc

    # Create joint distribution over (income shocks, return shocks)
    JointShkDstn = create_joint_distribution(IncShkDstn, RiskyAvg, RiskyStd)

    # Recapitalize human wealth at the Euler-implied return factor. prepare_to_solve
    # discounted it at Rfree, which prices a riskless asset that this model does not
    # offer; the limiting stochastic discount factor supplies the rate instead.
    RfreeHuman = calc_return_factor_human_wealth(JointShkDstn, CRRA)
    Ex_IncNext = expected(lambda x: x["PermShk"] * x["TranShk"], IncShkDstn)
    hNrm = calc_human_wealth(
        solution_next.hNrm,
        PermGroFac,
        RfreeHuman,
        Ex_IncNext,
    )

    # Calculate stochastic borrowing constraint (accounts for minimum return)
    # This follows HARK's approach: constraint for all combinations, take max
    BoroCnstNat = calc_boro_const_nat_stochastic_r(
        solution_next.mNrmMin,
        JointShkDstn,
        PermGroFac,
    )

    # Set mNrmMin to the stochastic constraint (or artificial if more binding)
    mNrmMin = BoroCnstNat if BoroCnstArt is None else max(BoroCnstNat, BoroCnstArt)

    # Solve EGM with stochastic returns - R is inside the expectation
    aNrm, cNrm, mNrm, EndOfPrdvP = solve_egm_step_stochastic_r(
        aXtraGrid,
        mNrmMin,
        DiscFacEff,
        PermGroFac,
        CRRA,
        JointShkDstn,
        vPfuncNext,
        uFunc,
    )

    mKink = mNrmMin
    if BoroCnstNat < mNrmMin:
        vPatCnst = (
            DiscFacEff
            * PermGroFac ** (-CRRA)
            * expected(
                calc_vp_next_stochastic_r,
                JointShkDstn,
                args=(np.array([mNrmMin]), CRRA, PermGroFac, vPfuncNext),
            )
        )
        mKink = mNrmMin + float(
            np.asarray(uFunc.derinv(vPatCnst, order=(1, 0))).item(),
        )

    # Compute realized MPCmin from the stochastic solution: a 2-point finite
    # difference of the consumption function at the top of the grid.
    # Validate before using: if it falls outside (0, 1) the bound is structurally
    # corrupt and would silently poison every subsequent moderation interpolant.
    dm = mNrm[-1] - mNrm[-2]
    if dm <= 0 or not np.isfinite(dm):
        msg = (
            "Stochastic-R MPCmin estimation failed: top two gridpoints are "
            f"not strictly increasing (mNrm[-1]={mNrm[-1]}, mNrm[-2]={mNrm[-2]}). "
            "Refine the asset grid before retrying."
        )
        raise ValueError(
            msg,
        )
    mpc_at_high_m = (cNrm[-1] - cNrm[-2]) / dm
    if not (0.0 < mpc_at_high_m < 1.0) or not np.isfinite(mpc_at_high_m):
        msg = (
            f"Stochastic-R MPCmin estimate {mpc_at_high_m!r} is outside (0, 1); "
            "the finite-difference at the top of the grid is unreliable. Refine "
            "the asset grid, or supply MPCmin analytically via the Merton-"
            "Samuelson formula 1 - (DiscFac * E[Risky^(1-CRRA)])^(1/CRRA)."
        )
        raise ValueError(
            msg,
        )
    # The bound is the Euler-implied limit, obtained by the same backward
    # recursion used without return risk; the slope measured above is retained
    # as a check on it, since the two must agree as the grid top rises.
    PatFacStoch = calc_patience_factor_stochastic_r(JointShkDstn, DiscFacEff, CRRA)
    MPCmin = calc_mpc_min(solution_next.MPCmin, PatFacStoch)
    if not 0.0 < MPCmin < 1.0:
        msg = (
            f"Euler-implied MPCmin {MPCmin!r} is outside (0, 1) "
            f"(PatFac={PatFacStoch!r}). Check the return process and the "
            "patience conditions."
        )
        raise ValueError(msg)
    rel_gap = abs(mpc_at_high_m - MPCmin) / MPCmin
    if rel_gap > 1e-2:
        logger.warning(
            "Euler-implied MPCmin %.9f and the slope measured at the top of "
            "the grid %.9f differ by %.2e in relative terms. The measured "
            "slope overstates the limit at finite wealth, so a gap this large "
            "usually means the grid top is too low to confirm the bound.",
            MPCmin,
            mpc_at_high_m,
            rel_gap,
        )

    # Create behavioral bounds for moderation
    optimist, pessimist, tighterUpperBound = make_behavioral_bounds(
        hNrm,
        mNrmMin,
        MPCmin,
        MPCmax,
        CRRA,
    )

    # Also create deterministic bounds (for comparison)
    optimist_det, pessimist_det, _ = make_behavioral_bounds(
        hNrm,
        mNrmMin,
        MPCmin_deterministic,
        MPCmax,
        CRRA,
    )

    # Build consumption function using MoM
    cFunc = _build_cfunc_mom(
        DiscFacEff=DiscFacEff,
        Rfree=RiskyAvg,  # Use mean return for extrapolation
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
        CubicBool=CubicBool,
        optimist=optimist,
        pessimist=pessimist,
        mKink=mKink,
    )

    # Construct marginal value functions
    vPfunc, vPPfunc = _build_marginal_value_funcs(cFunc, CRRA, CubicBool)

    # Construct value function if requested
    vFunc = _build_complete_vfunc(
        vFuncBool=vFuncBool,
        aNrm=aNrm,
        BoroCnstNat=BoroCnstNat,
        DiscFacEff=DiscFacEff,
        Rfree=RiskyAvg,
        PermGroFac=PermGroFac,
        CRRA=CRRA,
        IncShkDstn=IncShkDstn,
        vFuncNext=vFuncNext,
        EndOfPrdvP=EndOfPrdvP,
        uFunc=uFunc,
        cNrm=cNrm,
        mNrm=mNrm,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        optimist=optimist,
        pessimist=pessimist,
        CubicBool=CubicBool,
        mKink=mKink,
    )

    # Assemble solution
    solution = _assemble_mom_solution(
        cFunc=cFunc,
        vFunc=vFunc,
        vPfunc=vPfunc,
        vPPfunc=vPPfunc,
        mNrmMin=mNrmMin,
        hNrm=hNrm,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
        optimist=optimist,
        pessimist=pessimist,
        tighterUpperBound=tighterUpperBound,
    )

    # Add comparison attributes
    solution.MPCmin_stochastic = MPCmin  # Combined income + return risk
    solution.MPCmin_deterministic = MPCmin_deterministic  # Income risk only
    solution.OptimistStochastic = optimist  # Bounds with stochastic MPCmin
    solution.PessimistStochastic = pessimist
    solution.Optimist = optimist_det  # Deterministic bounds for comparison
    solution.Pessimist = pessimist_det
    return solution


# Default parameters for stochastic return consumer
IndShockMoMStochasticR_defaults = IndShockConsumerType_defaults.copy()
IndShockMoMStochasticR_defaults["RiskyAvg"] = (
    1.03  # Same mean as Rfree (mean-preserving spread)
)
IndShockMoMStochasticR_defaults["RiskyStd"] = (
    0.10  # 10% std deviation (modest volatility)
)


class RiskyAssetMoMConsumerType(IndShockConsumerType):
    """HARK consumer type using Method of Moderation with stochastic returns.

    This class extends the standard Method of Moderation to handle i.i.d.
    lognormal returns on savings. The Merton-Samuelson result provides
    an analytical MPC that replaces MPCmin in the moderation formulas.

    Parameters
    ----------
    RiskyAvg : float
        Mean of risky return R (in levels, default 1.03 = same as Rfree)
    RiskyStd : float
        Standard deviation of risky return R (in levels, default 0.15)

    Notes
    -----
    - Both optimist and pessimist face the same stochastic return
    - The MPC is computed using the Merton-Samuelson formula
    - Returns here are idiosyncratic and i.i.d.; serially correlated
      (state-dependent) returns are outside this module's scope

    References
    ----------
    Samuelson (1969), Merton (1969, 1971)

    """

    default_: ClassVar = {
        "params": IndShockMoMStochasticR_defaults,
        "solver": method_of_moderation_stochastic_r,
        "model": "ConsIndShock.yaml",
    }

    time_inv_: ClassVar = [*IndShockConsumerType.time_inv_, "RiskyAvg", "RiskyStd"]
