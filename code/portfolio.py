r"""Method of Moderation applied to the portfolio-choice risky share.

The consumption problem moderates between two perfect-foresight agents. Portfolio
choice has only ONE limiting agent, the myopic (Merton-Samuelson) investor whose
share ``sigmaStar`` is the :math:`m \\to \\infty` limit, together with ONE
constraint, the leverage cap ``share = 1``: you cannot borrow to invest. The
bracket is therefore ``[sigmaStar, 1]``, whose width ``1 - sigmaStar`` is
constant in the state, exactly as the consumption bracket's width is.

``sigmaStar`` is taken from HARK's ``ShareLimit``. In HARK the discretized
process IS the process, so that value is exact for the model being solved; it is
the limiting share of that model, which is not the same object as the
Merton-Samuelson share of a continuous-lognormal model.

Orientation
-----------
The link is NOT inherited from the consumption code. Orient the transform to the
bound the solution APPROACHES and take the log of the distance to that bound; a
power-law approach then becomes a straight line. Consumption approaches the
optimist (``omega -> 1``), so its linearizer is ``log(1 - omega)``, and the
symmetric logit is a reasonable stand-in. The share approaches the myopic bound
from above (``omega -> 0``), so the correct linearizer is ``log(omega)``.

Measured on the default calibration against a converged reference, with n=5
nodes over ``m`` in [25, 100] and scored against interpolating the share itself:
the log link is 107x better, cloglog 8.4x, logit 4.5x, probit 3.7x, loglog 2.5x.
The tail is a power law, ``share(m) - sigmaStar ~ m**-phi``, which is what makes
``log(omega)`` linear in ``log m``. The ASYMPTOTIC exponent is 0.91, fitted over
``m`` in [1000, 50000]; over the [25, 100] range quoted just above, the local
exponent is nearer 0.81, so the link is close to but not exactly straight there.
MoM reads whatever slope the solve exhibits instead of imposing one.

Three regions
-------------
No single link wins everywhere, because the capped region approaches the UPPER
bound while the tail approaches the LOWER one. The construction is therefore
three-piece, mirroring the cusp construction of the consumption code
(``_build_cfunc_mom_cusp``) with the addition that the LINK changes with the
region, not merely the bound:

1. below the release, the share is exactly 1, so it is represented exactly;
2. a Hermite segment bridges the release, matching level and slope at the two
   nodes bracketing it (the ``cFuncMidTightUpBd`` analogue);
3. above the bridge, ``log(omega)`` is interpolated against ``log m``.

Omitting the bridge is what makes every single-region scheme transition-bound,
and keeping the log link away from the release is what prevents it from
exceeding the cap, since ``log`` has no upper asymptote.

Scope
-----
HARK's recursion reads ``vPfuncAdj``, ``dvdmFuncFxd``, ``dvdsFuncFxd``,
``vFuncAdj``, ``vFuncFxd``, ``MPCmin`` and ``hNrm`` from ``solution_next``, but
never ``ShareFuncAdj``. The share is an output of each period's solve, obtained
by root-finding the first-order condition over ``ShareGrid``, so a moderated
share does not feed back into the recursion and this module does not claim that
it does. What it supports is the cheaper claim: the expensive per-node FOC solve
can be done on far fewer nodes when the share in between is moderated rather
than interpolated directly.
"""

from __future__ import annotations

import logging

import numpy as np
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    solve_one_period_ConsPortfolio,
)
from HARK.interpolation import LinearInterp
from HARK.metric import MetricObject

logger = logging.getLogger(__name__)

OMEGA_CLIP = 1e-12
"""Floor for the moderation ratio before taking its log."""

CAP_TOL = 1e-9
"""How close to 1 a share must be to count as sitting on the leverage cap."""

M_FLOOR = 1e-12
"""Smallest m handed to log(); guards the link when no cap region exists."""


def moderate_share(share, sigma_star):
    """Moderation ratio of a risky share within ``[sigma_star, 1]``.

    Parameters
    ----------
    share : np.ndarray
        Risky share values.
    sigma_star : float
        The myopic agent's share, the limiting lower bound.

    Returns
    -------
    np.ndarray
        ``(share - sigma_star) / (1 - sigma_star)``, in ``[0, 1]``. The value
        approaches 0 as ``m`` grows and equals 1 where the leverage cap binds.

    """
    return (share - sigma_star) / (1.0 - sigma_star)


def unmoderate_share(omega, sigma_star):
    """Invert :func:`moderate_share`."""
    return sigma_star + omega * (1.0 - sigma_star)


class ModeratedShareFunc(MetricObject):
    """Three-piece moderated risky-share function.

    Exact on the leverage cap, Hermite across the release, and
    ``log(omega)``-moderated above it. See the module docstring for why the link
    differs by region.

    Parameters
    ----------
    m_nodes : np.ndarray
        Market-resources nodes, strictly increasing, where the share is known.
    s_nodes : np.ndarray
        Solved risky share at ``m_nodes``.
    sigma_star : float
        The myopic agent's share (HARK's ``ShareLimit``). Must lie strictly
        inside ``(0, 1)`` or the bracket ``[sigma_star, 1]`` is degenerate.

    Raises
    ------
    ValueError
        If ``sigma_star`` is outside ``(0, 1)``, if the node arrays disagree in
        length, if fewer than two nodes survive the ``m > 0`` filter, or if
        ``m_nodes`` is not strictly increasing. Every one of these otherwise
        fails silently rather than loudly: ``sigma_star = 1`` yields an
        all-ones function indistinguishable from a fully capped solution, and
        unsorted nodes yield plausible-looking wrong numbers.

    Attributes
    ----------
    mRelease : float
        Largest node at which the cap still binds, or ``-inf`` if it never does.
    mBridge : float
        First node above the release; the moderated region starts here.

    """

    # chiFunc carries its own grid; bare chi_nodes would not. mRelease is
    # excluded deliberately: it is -inf with no capped region, and HARK makes
    # abs(-inf - -inf) nan, so a solve would never converge.
    distance_criteria = ["sigma_star", "chiFunc"]

    def __init__(self, m_nodes, s_nodes, sigma_star) -> None:
        sigma_star = float(sigma_star)
        if not 0.0 < sigma_star < 1.0:
            msg = f"sigma_star must lie strictly in (0, 1); got {sigma_star!r}"
            raise ValueError(msg)

        m_nodes = np.asarray(m_nodes, dtype=float)
        s_nodes = np.asarray(s_nodes, dtype=float)
        if m_nodes.shape != s_nodes.shape:
            msg = f"m_nodes {m_nodes.shape} and s_nodes {s_nodes.shape} disagree"
            raise ValueError(msg)

        keep = m_nodes > 0.0  # log m is undefined at the origin
        m_nodes, s_nodes = m_nodes[keep], s_nodes[keep]
        if len(m_nodes) < 2:
            msg = f"need at least 2 nodes with m > 0; got {len(m_nodes)}"
            raise ValueError(msg)
        if not np.all(np.diff(m_nodes) > 0.0):
            msg = "m_nodes must be strictly increasing"
            raise ValueError(msg)

        self.sigma_star = sigma_star
        self.m_nodes, self.s_nodes = m_nodes, s_nodes
        capped = s_nodes >= 1.0 - CAP_TOL
        # Only a leading run of capped nodes counts; the cap binds at low wealth.
        n_capped = int(np.argmin(capped)) if not capped.all() else len(capped)
        if capped[n_capped:].any():
            # A node back on the cap above the release means the solve is not
            # monotone there, so the release sits too low and the exact-cap
            # region is short. Cheap to detect, impossible to see downstream.
            logger.warning(
                "%d capped node(s) lie above the release at m = %.4g; the "
                "capped run is not contiguous and the release may be misplaced",
                int(capped[n_capped:].sum()),
                float(m_nodes[n_capped - 1]) if n_capped else float(m_nodes[0]),
            )

        if len(m_nodes) - n_capped < 2:
            # Too few uncapped nodes to split on. Capped nodes carry omega = 1,
            # so chi = 0 and the link reproduces them exactly. Warned, not
            # whispered: dropping the bridge restores the transition error.
            logger.warning(
                "share capped on %d of %d nodes; falling back to one moderated "
                "region with no bridge",
                n_capped,
                len(m_nodes),
            )
            n_capped = 0

        self.mRelease = float(m_nodes[n_capped - 1]) if n_capped > 0 else -np.inf
        self.mBridge = float(m_nodes[n_capped])

        m_up, s_up = m_nodes[n_capped:], s_nodes[n_capped:]
        raw_omega = moderate_share(s_up, self.sigma_star)
        n_below = int(np.sum(raw_omega < 0.0))
        if n_below:
            # A solved share under the myopic bound is a real bound violation.
            # Clipping it silently would turn it into an ordinary finite chi
            # node and hide it from everything downstream.
            logger.warning(
                "solved share fell below the myopic bound at %d node(s); "
                "min omega = %.3e",
                n_below,
                float(raw_omega.min()),
            )
        self.chi_nodes = np.log(np.clip(raw_omega, OMEGA_CLIP, 1.0))
        self.chiFunc = LinearInterp(np.log(m_up), self.chi_nodes, lower_extrap=True)

        # Hermite bridge: level and slope matched at both ends. The capped side
        # is flat by construction, so its slope is zero.
        if np.isfinite(self.mRelease):
            self._h = self.mBridge - self.mRelease
            self._y_hi = float(self._share_upper(np.array([self.mBridge]))[0])
            self._d_hi = float(self._dshare_upper(np.array([self.mBridge]))[0])
        else:
            self._h = 0.0

    @property
    def x_list(self):
        """Nodes the function was built from, so the factory is closed on it."""
        return self.m_nodes

    @property
    def y_list(self):
        """Solved shares at :attr:`x_list`."""
        return self.s_nodes

    def _floor(self):
        """Smallest m safe to hand the log link.

        With a release, values below it are overwritten by the cap answer, so
        flooring at ``mBridge`` costs nothing. With no release nothing
        overwrites them, so flooring there would flatten the function instead
        of letting ``lower_extrap`` do its job.
        """
        return self.mBridge if np.isfinite(self.mRelease) else M_FLOOR

    def _share_upper(self, m):
        """Moderated share above the bridge, clamped to respect the cap."""
        raw = np.exp(self.chiFunc(np.log(m)))
        if np.any(raw > 1.0 + 1e-12):
            logger.debug("log link exceeded the cap; clamped")
        return unmoderate_share(np.minimum(raw, 1.0), self.sigma_star)

    def _dshare_upper(self, m):
        """d(share)/dm above the bridge, via the chain rule through the link."""
        mu = np.log(m)
        chi = self.chiFunc(mu)
        dchi = self.chiFunc.derivative(mu)
        return (1.0 - self.sigma_star) * np.exp(chi) * dchi / m

    def _bridge(self, m):
        """Cubic Hermite across the release: flat on the left, matched on the right."""
        t = (m - self.mRelease) / self._h
        t2, t3 = t * t, t * t * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return h00 * 1.0 + h01 * self._y_hi + h11 * self._h * self._d_hi

    def _dbridge(self, m):
        """d/dm of :meth:`_bridge`."""
        t = (m - self.mRelease) / self._h
        t2 = t * t
        dh00 = 6.0 * t2 - 6.0 * t
        dh01 = -6.0 * t2 + 6.0 * t
        dh11 = 3.0 * t2 - 2.0 * t
        return (dh00 * 1.0 + dh01 * self._y_hi) / self._h + dh11 * self._d_hi

    def _by_region(self, m, upper, bridge, capped):
        """Evaluate every branch on safe inputs, then select by region."""
        m_in = np.asarray(m, dtype=float)
        out = upper(np.maximum(m_in, self._floor()))
        if np.isfinite(self.mRelease):
            mid = (m_in > self.mRelease) & (m_in < self.mBridge)
            safe_mid = np.clip(m_in, self.mRelease, self.mBridge)
            out = np.where(mid, bridge(safe_mid), out)
            out = np.where(m_in <= self.mRelease, capped, out)
        return m_in, out

    def __call__(self, m):
        r"""Moderated share at ``m``, guaranteed to lie in ``[sigma_star, 1]``.

        The final clip carries that guarantee. Below the release the answer is
        the literal cap and above the bridge ``exp`` keeps omega in ``(0, 1]``,
        so only the Hermite segment can leave the bracket, and it does: its
        tangent basis ``t**3 - t**2`` reaches ``-4/27`` on ``(0, 1)`` while
        ``_d_hi`` is negative, adding a positive amount to a left endpoint
        already sitting on the cap. The unclipped breach is
        ``max(0, (4/27) * _h * abs(_d_hi) - (1 - _y_hi))``, positive exactly
        when the first uncapped node is still near the cap while the last
        capped node is far below it, which is the sparse-grid regime this
        method exists to serve. Measured breaches of 3.8e-2 and 5.6e-2 on
        two such grids match that expression to three figures.
        """
        m_in, out = self._by_region(m, self._share_upper, self._bridge, 1.0)
        out = np.clip(out, self.sigma_star, 1.0)  # enforces the bracket
        return float(out) if m_in.ndim == 0 else out

    def _share_upper_raw(self, m):
        """Moderated share above the bridge with NO cap clamp, for clip detection."""
        return unmoderate_share(np.exp(self.chiFunc(np.log(m))), self.sigma_star)

    def derivative(self, m):
        """d(share)/dm, matching HARK's interpolator contract.

        Wherever :meth:`__call__` returns a clipped value (the bridge above the
        cap, or the log link above it), the function is flat there, so the
        derivative is zero; returning the branch's own slope would contradict
        the value and break finite-difference checks.
        """
        m_in, out = self._by_region(m, self._dshare_upper, self._dbridge, 0.0)
        _, raw = self._by_region(m, self._share_upper_raw, self._bridge, 1.0)
        out = np.where((raw > 1.0) | (raw < self.sigma_star), 0.0, out)
        return float(out) if m_in.ndim == 0 else out


def make_moderated_share_func(share_func, sigma_star):
    """Rebuild a HARK share function as a :class:`ModeratedShareFunc`.

    Reads the nodes back off the ``LinearInterp`` that
    ``solve_one_period_ConsPortfolio`` builds at its line 857, so the moderated
    function is constructed from exactly the same solved data.
    """
    return ModeratedShareFunc(share_func.x_list, share_func.y_list, sigma_star)


def solve_one_period_ConsPortfolioMoM(
    solution_next,
    ShockDstn,
    IncShkDstn,
    RiskyDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    ShareGrid,
    AdjustPrb,
    ShareLimit,
    vFuncBool,
    DiscreteShareBool,
    IndepDstnBool,
):
    """Solve one period, then represent the risky share by moderation.

    The parameter list mirrors :func:`solve_one_period_ConsPortfolio` exactly,
    and must: HARK builds its call dictionary by introspecting the solver's
    argument names, so a ``*args``-style wrapper receives nothing. Only the
    construction of ``ShareFuncAdj`` differs. ``ShareLimit`` is stashed on the
    solution so tests and figures need not reach back into the agent.
    """
    solution = solve_one_period_ConsPortfolio(
        solution_next=solution_next,
        ShockDstn=ShockDstn,
        IncShkDstn=IncShkDstn,
        RiskyDstn=RiskyDstn,
        LivPrb=LivPrb,
        DiscFac=DiscFac,
        CRRA=CRRA,
        Rfree=Rfree,
        PermGroFac=PermGroFac,
        BoroCnstArt=BoroCnstArt,
        aXtraGrid=aXtraGrid,
        ShareGrid=ShareGrid,
        AdjustPrb=AdjustPrb,
        ShareLimit=ShareLimit,
        vFuncBool=vFuncBool,
        DiscreteShareBool=DiscreteShareBool,
        IndepDstnBool=IndepDstnBool,
    )
    sigma_star = float(np.atleast_1d(ShareLimit)[0])
    solution.ShareFuncAdjDirect = solution.ShareFuncAdj
    solution.ShareFuncAdj = make_moderated_share_func(solution.ShareFuncAdj, sigma_star)
    solution.ShareLimit = sigma_star
    return solution


class PortfolioMoMConsumerType(PortfolioConsumerType):
    """Portfolio-choice consumer whose risky share is represented by moderation."""

    default_ = {
        **PortfolioConsumerType.default_,
        "solver": solve_one_period_ConsPortfolioMoM,
    }
