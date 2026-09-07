"""Tests for the moderated portfolio-choice risky share.

Conventions follow ``test_moderation.py``: plain test functions, module-scope
fixtures so the model is solved once, and a stated tolerance per test chosen for
what that test is actually checking.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.interpolation import LinearInterp
from HARK.metric import distance_metric
from portfolio import (
    CAP_TOL,
    ModeratedShareFunc,
    PortfolioMoMConsumerType,
    make_moderated_share_func,
    moderate_share,
    unmoderate_share,
)

GRID = {"aXtraMax": 100.0, "aXtraCount": 200, "tolerance": 1e-6}


@pytest.fixture(scope="module")
def solved():
    """Solve the moderated portfolio type once for the whole module."""
    agent = PortfolioMoMConsumerType(**{**init_portfolio, "cycles": 0, **GRID})
    agent.solve()
    return agent


@pytest.fixture(scope="module")
def sol(solved):
    return solved.solution[0]


@pytest.fixture(scope="module")
def reference():
    """An independent, converged, dense solve to score sparse schemes against.

    Deliberately not the solve the test nodes are drawn from: scoring a scheme
    against its own grid is self-consistent and overstates the advantage.

    Density matters as much as independence, and in the opposite direction: a
    merely dense reference measures its OWN error rather than the scheme's, and
    understates the advantage. At 400/600 the tail ratios read 4.3-4.8x; at the
    2000/1000 used here they read 74-150x, reproducing the 108x recorded in
    make_paper_figures.py. The plan specifies 2000/1000 for this reason.
    """
    agent = PortfolioConsumerType(
        **{
            **init_portfolio,
            "cycles": 0,
            "aXtraMax": 2000.0,
            "aXtraCount": 1000,
            "tolerance": 1e-8,
        },
    )
    agent.solve()
    return agent.solution[0]


def test_moderate_share_roundtrip() -> None:
    """moderate_share and unmoderate_share invert each other exactly."""
    sigma = 0.31
    shares = np.linspace(sigma, 1.0, 25)
    back = unmoderate_share(moderate_share(shares, sigma), sigma)
    assert np.allclose(back, shares, rtol=0, atol=1e-15)


def test_solved_shares_lie_in_the_bracket(sol) -> None:
    """The SOLVED nodes respect [sigmaStar, 1] before any moderation.

    Asserted on the raw solve, not on ``ShareFuncAdj``: ``__call__`` clips its
    output into the bracket, so asserting there would test the clip. This is
    the only place the solver's own bound behaviour is visible.
    """
    raw = moderate_share(np.asarray(sol.ShareFuncAdjDirect.y_list), sol.ShareLimit)
    assert np.all(np.isfinite(raw))
    assert np.all(raw >= -1e-12), "solved share below the myopic bound"
    assert np.all(raw <= 1.0 + 1e-12), "solved share above the leverage cap"


def test_bridge_does_not_need_the_output_clip(sol) -> None:
    """The Hermite bridge is the only region that can leave the bracket.

    Below the release ``__call__`` returns a literal 1.0, and above the bridge
    the link returns ``sigma_star + (0, 1] * (1 - sigma_star)`` by arithmetic,
    so a bracket assertion on either is vacuous. Only the cubic can overshoot.
    Reads ``_bridge`` directly to sit upstream of the clip.
    """
    f = sol.ShareFuncAdj
    m = np.linspace(f.mRelease, f.mBridge, 5000)
    pre_clip = f._bridge(m)  # noqa: SLF001 - the clip is what we must bypass
    assert pre_clip.max() <= 1.0 + 1e-12, "bridge breaches the cap unclipped"
    assert pre_clip.min() >= sol.ShareLimit - 1e-12, "bridge dips below the bound"


def test_output_clip_catches_a_bridge_overshoot() -> None:
    """A sparse grid CAN drive the cubic through the cap; the clip must catch it.

    These nodes put a still-near-cap point beside a distant one, which steepens
    the matched slope and sends the unclipped cubic to about 1.056. The first
    assertion fails if the fixture ever stops overshooting, so this test cannot
    silently degrade into one that passes because there is nothing to catch.
    """
    m = np.array([1.0, 2.0, 3.0, 20.0, 100.0])
    s = np.array([1.0, 1.0, 0.999, 0.34, 0.315])
    f = ModeratedShareFunc(m, s, 0.310565)
    probe = np.linspace(f.mRelease, f.mBridge, 20001)
    assert f._bridge(probe).max() > 1.0, "fixture no longer overshoots"  # noqa: SLF001
    assert f(probe).max() <= 1.0 + 1e-15, "clip failed to contain the overshoot"


def test_derivative_is_flat_where_the_clip_binds() -> None:
    """``derivative`` must agree with the clipped ``__call__``, not the raw branch.

    Same overshooting fixture as above. Where the unclipped bridge sits above
    the cap the function is flat at 1, so the derivative there is zero; away
    from the clip and the kinks it matches a central finite difference of the
    function itself. Tolerance 1e-6 on the finite difference, step 1e-6.
    """
    m = np.array([1.0, 2.0, 3.0, 20.0, 100.0])
    s = np.array([1.0, 1.0, 0.999, 0.34, 0.315])
    f = ModeratedShareFunc(m, s, 0.310565)
    probe = np.linspace(f.mRelease, f.mBridge, 2001)[1:-1]
    raw = f._bridge(probe)  # noqa: SLF001
    clipped = raw > 1.0
    assert clipped.any(), "fixture no longer overshoots"
    assert np.all(f.derivative(probe[clipped]) == 0.0), "slope nonzero on the cap"

    h = 1e-6
    interior = (~clipped) & (raw < 1.0 - 1e-3)
    fd = (f(probe[interior] + h) - f(probe[interior] - h)) / (2 * h)
    assert np.allclose(f.derivative(probe[interior]), fd, atol=1e-6), (
        "derivative disagrees with finite difference off the clip"
    )


def test_capped_region_is_exact(sol) -> None:
    """Below the release the share is exactly 1, not merely close to it."""
    f = sol.ShareFuncAdj
    assert np.isfinite(f.mRelease), "cap should bind on this calibration"
    m = np.linspace(0.05, f.mRelease, 500)
    assert np.allclose(f(m), 1.0, rtol=0, atol=1e-15)
    # Not vacuous only because the SOLVE put those nodes on the cap as well;
    # without this the assertion above is a literal 1.0 compared against 1.0.
    on_cap = f.s_nodes[f.m_nodes <= f.mRelease]
    assert on_cap.size > 0, "no solved nodes inside the capped region"
    assert np.all(on_cap >= 1.0 - CAP_TOL), "solve did not cap where MoM claims"


def test_share_is_continuous_across_the_bridge(sol) -> None:
    """No jump at either end of the Hermite segment."""
    f = sol.ShareFuncAdj
    eps = 1e-7
    for knot in (f.mRelease, f.mBridge):
        lo = float(f(np.array([knot - eps]))[0])
        hi = float(f(np.array([knot + eps]))[0])
        assert abs(hi - lo) < 1e-5, f"discontinuity at {knot}"


def test_share_is_non_increasing(sol) -> None:
    """Risky share falls with wealth as human capital becomes negligible."""
    m = np.geomspace(0.05, 100.0, 3000)
    assert np.all(np.diff(sol.ShareFuncAdj(m)) <= 1e-9)


def test_solved_nodes_are_reproduced_exactly(sol) -> None:
    """Above the bridge the moderated function passes through its own nodes.

    This is an identity in the tail (the link interpolates those nodes), so the
    tolerance is machine epsilon rather than something loose that would hide a
    real gap. What it can still catch is REGION MIS-ROUTING: if the release or
    bridge were misplaced, tail nodes would be answered by the cap branch or
    the cubic and would stop reproducing.
    """
    nodes = np.asarray(sol.ShareFuncAdjDirect.x_list)
    nodes = nodes[(nodes > sol.ShareFuncAdj.mBridge) & (nodes <= 100.0)]
    assert nodes.size > 10, "expected many nodes above the bridge"
    assert np.allclose(
        sol.ShareFuncAdj(nodes),
        sol.ShareFuncAdjDirect(nodes),
        rtol=0,
        atol=1e-12,
    )


def test_link_slope_matches_measured_decay_exponent(sol) -> None:
    """The LINK's fitted slope equals minus the gap's decay exponent.

    Pins the claim that MoM recovers the power-law tail from a MEASURED slope
    rather than an imposed constant: a chi linear with slope -phi maps through
    the link to a gap decaying like m**-phi.

    Both quantities must come from DIFFERENT objects or the test is empty.
    Comparing polyfit of log(gap) against polyfit of log(moderate_share(...))
    is an identity for any function whatsoever, since the two differ by the
    constant log(1 - sigmaStar); a bogus m**-2.7 share satisfies it to ten
    digits. So the link side is read off chiFunc's own nodes instead.
    """
    f = sol.ShareFuncAdj
    mu = np.log(f.m_nodes[f.m_nodes >= f.mBridge])
    link_slope = np.polyfit(mu, f.chiFunc(mu), 1)[0]

    m = np.geomspace(30.0, 100.0, 40)
    decay = np.polyfit(np.log(m), np.log(f(m) - sol.ShareLimit), 1)[0]

    assert np.isclose(link_slope, decay, rtol=5e-2), (
        f"link slope {link_slope:.4f} does not track decay {decay:.4f}"
    )
    assert -1.0 < decay < -0.7, f"decay exponent {decay:.3f} outside the measured band"


def test_moderated_beats_linear_in_the_tail(reference, sol) -> None:
    """Accuracy per gridpoint in the TAIL, clear of the release.

    Both schemes are handed the SAME sparse nodes and the same solved values, so
    the comparison isolates the representation. The baseline is PLAIN LINEAR
    interpolation, matching the consumption sections; HARK's decay-to-ShareLimit
    extrapolator is a different object and is identical to linear here anyway,
    since the nodes span the probe and nothing is extrapolated.

    Note what this test does NOT cover. Every node here sits above the release,
    so ``n_capped`` is 0 and the object under test is a single-region log link
    with no capped piece and no bridge. The three-piece construction is scored
    by test_accuracy_by_region_on_a_grid_spanning_the_cap, where the margins
    are far narrower and one case loses outright.
    """
    probe = np.geomspace(25.0, 100.0, 400)
    truth = reference.ShareFuncAdj(probe)
    for n in (4, 5, 7):
        nodes = np.geomspace(25.0, 100.0, n)
        s_nodes = reference.ShareFuncAdj(nodes)
        f = ModeratedShareFunc(nodes, s_nodes, sol.ShareLimit)
        assert not np.isfinite(f.mRelease), "expected a single-region link here"
        linear = np.abs(LinearInterp(nodes, s_nodes, lower_extrap=True)(probe) - truth)
        mod = np.abs(f(probe) - truth)
        ratio = linear.max() / mod.max()
        assert mod.max() < linear.max(), f"moderation lost at n={n}"
        # Measured 149.6x / 108.1x / 74.1x at n = 4 / 5 / 7 against the
        # converged reference; 40x leaves headroom without being vacuous.
        assert ratio > 40.0, f"n={n}: only {ratio:.1f}x, expected >40x"


def test_all_capped_nodes_fall_back_without_splitting(sol, caplog) -> None:
    """With too few uncapped nodes the constructor must not try to split.

    The fallback drops the bridge, restoring the transition error the
    three-piece design exists to remove, so it must log at warning level.
    """
    nodes = np.array([1.0, 2.0, 3.0, 4.0])
    with caplog.at_level(logging.WARNING, logger="portfolio"):
        f = ModeratedShareFunc(nodes, np.ones_like(nodes), sol.ShareLimit)
    assert not np.isfinite(f.mRelease)
    assert np.allclose(f(np.array([0.5, 2.5, 10.0])), 1.0, atol=1e-12)
    assert "no bridge" in caplog.text, "silent fallback"


def test_origin_node_is_dropped_not_fatal(sol) -> None:
    """HARK's share grid starts at m=0, where log m is undefined."""
    nodes = np.array([0.0, 1.0, 10.0, 100.0])
    shares = np.array([1.0, 1.0, 0.85, 0.39])
    f = ModeratedShareFunc(nodes, shares, sol.ShareLimit)
    assert np.all(np.isfinite(f(np.array([0.0, 0.5, 50.0]))))


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1])
def test_degenerate_bracket_raises(bad) -> None:
    """sigma_star outside (0, 1) used to yield a silent all-ones function.

    The clip chain absorbed the division blow-up, so the constructor succeeded
    and returned something indistinguishable from a fully capped solution.
    """
    with pytest.raises(ValueError, match="sigma_star"):
        ModeratedShareFunc(np.array([1.0, 10.0]), np.array([1.0, 0.4]), bad)


def test_unsorted_nodes_raise() -> None:
    """Unsorted nodes used to return plausible-looking wrong numbers, not nan."""
    with pytest.raises(ValueError, match="strictly increasing"):
        ModeratedShareFunc(
            np.array([1.0, 20.0, 3.0, 100.0]),
            np.array([1.0, 0.34, 0.999, 0.315]),
            0.310565,
        )


def test_too_few_surviving_nodes_raise() -> None:
    """One node cannot support an interpolant; this used to be an IndexError."""
    with pytest.raises(ValueError, match="at least 2"):
        ModeratedShareFunc(np.array([0.0, 5.0]), np.array([1.0, 0.4]), 0.31)


def test_mismatched_node_lengths_raise() -> None:
    """Ragged inputs used to raise IndexError from a boolean mask."""
    with pytest.raises(ValueError, match="disagree"):
        ModeratedShareFunc(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.4]), 0.31)


def test_distance_is_finite_when_the_cap_never_binds() -> None:
    """Convergence must not depend on whether the leverage cap happens to bind.

    mRelease is -inf with no capped region, and HARK turns abs(-inf - -inf)
    into nan, which never compares below tolerance, so listing it in
    distance_criteria made an infinite-horizon solve run forever.
    """
    m = np.array([1.0, 2.0, 3.0, 20.0, 100.0])
    s = np.array([0.9, 0.7, 0.55, 0.34, 0.315])
    a = ModeratedShareFunc(m, s, 0.310565)
    b = ModeratedShareFunc(m, s * 1.0001, 0.310565)
    assert not np.isfinite(a.mRelease), "fixture should have no capped region"
    d = distance_metric(a, b)
    assert np.isfinite(d), "distance is nan; solve would never converge"
    assert d > 0.0, "distinct solutions compared equal"


def test_no_release_extrapolates_instead_of_flattening() -> None:
    """With no capped region nothing overrides the low-m branch.

    Flooring evaluation at mBridge there would silently flatten the function
    below the grid instead of letting the log link extrapolate.
    """
    m = np.array([1.0, 2.0, 3.0, 20.0, 100.0])
    s = np.array([0.9, 0.7, 0.55, 0.34, 0.315])
    f = ModeratedShareFunc(m, s, 0.310565)
    assert np.isclose(f(1.0), 0.9, atol=1e-12), "first node not reproduced"
    assert f(0.5) > f(1.0), "flattened below the grid instead of extrapolating"


def test_interpolator_contract_for_hark_substitution(sol) -> None:
    """HARK swaps this in for a LinearInterp, so it must answer the same queries.

    x_list/y_list also close make_moderated_share_func on its own output, which
    reads those attributes off whatever it is handed.
    """
    f = sol.ShareFuncAdj
    assert len(f.x_list) == len(f.y_list)
    again = make_moderated_share_func(f, sol.ShareLimit)
    assert np.allclose(again(np.array([50.0])), f(np.array([50.0])))

    m = np.array([20.0, 50.0, 90.0])
    eps = 1e-5
    finite_diff = (f(m + eps) - f(m - eps)) / (2.0 * eps)
    assert np.allclose(f.derivative(m), finite_diff, rtol=1e-4, atol=1e-9)


def test_scalar_input_returns_a_scalar(sol) -> None:
    """Matches TransformedFunctionMoMCusp, which returns float for scalar m."""
    assert isinstance(sol.ShareFuncAdj(50.0), float)
    assert isinstance(sol.ShareFuncAdj.derivative(50.0), float)


# n, tail-ratio floor, transition-ratio floor. Measured against the converged
# reference: tail 162.9 / 12.9 / 12.5 and transition 0.54 / 2.14 / 2.03.
# n=8 LOSES on the transition and the floor records that rather than hiding it;
# all three are lower bounds, so an improvement still passes.
SPANNING_ACCURACY = ((8, 100.0, 0.4), (12, 10.0, 1.5), (20, 10.0, 1.5))


@pytest.mark.parametrize(("n", "tail_floor", "transition_floor"), SPANNING_ACCURACY)
def test_accuracy_by_region_on_a_grid_spanning_the_cap(
    reference,
    sol,
    n,
    tail_floor,
    transition_floor,
) -> None:
    """Per-region accuracy for the actual three-piece object.

    The tail-only test above never builds a capped region or a bridge. Here the
    nodes span m in [1, 100], so all three pieces exist and each is scored
    separately, as the plan requires.

    The honest result: moderation wins the tail by one to two orders of
    magnitude and is roughly 2x on the transition once the grid resolves the
    release, but at n=8 the bridge is worse than plain linear across the
    transition (0.54x). That is a documented limit of the Hermite segment on
    very coarse grids, in the same spirit as the near-constraint head losing to
    EGM in the Markov module.
    """
    probe = np.geomspace(1.0, 100.0, 2000)
    truth = reference.ShareFuncAdj(probe)
    nodes = np.geomspace(1.0, 100.0, n)
    s_nodes = reference.ShareFuncAdj(nodes)
    f = ModeratedShareFunc(nodes, s_nodes, sol.ShareLimit)
    assert np.isfinite(f.mRelease), "expected all three regions on a spanning grid"

    linear = np.abs(LinearInterp(nodes, s_nodes, lower_extrap=True)(probe) - truth)
    mod = np.abs(f(probe) - truth)
    tail = probe >= f.mBridge
    mid = (probe > f.mRelease) & (probe < f.mBridge)

    tail_ratio = linear[tail].max() / mod[tail].max()
    assert tail_ratio > tail_floor, f"n={n}: tail only {tail_ratio:.1f}x"

    assert mid.any(), "probe does not enter the transition region"
    transition_ratio = linear[mid].max() / max(mod[mid].max(), 1e-300)
    assert transition_ratio > transition_floor, (
        f"n={n}: transition {transition_ratio:.2f}x fell below {transition_floor}"
    )


def test_bridge_overshoot_bound_holds_at_construction(sol) -> None:
    """The cap breach has a closed form, so assert it without probing anywhere.

    ``_bridge`` adds ``h11 * _h * _d_hi`` to a left endpoint already at the cap,
    and ``h11 = t**3 - t**2`` bottoms out at -4/27 on (0, 1) while ``_d_hi`` is
    negative. The unclipped peak therefore exceeds 1 by
    ``(4/27) * _h * abs(_d_hi) - (1 - _y_hi)``. A probe grid can miss the
    maximum; this expression cannot.
    """
    f = sol.ShareFuncAdj
    breach = (4.0 / 27.0) * f._h * abs(f._d_hi) - (1.0 - f._y_hi)  # noqa: SLF001
    assert breach <= 0.0, f"bridge overshoots the cap by {breach:.3e} before clipping"


def test_log_link_stays_under_the_cap_without_the_clamp(sol) -> None:
    """Pins the module docstring's claim about keeping the link off the release.

    The docstring says holding the log link away from the release is what stops
    it exceeding the cap. Today ``np.minimum`` in ``_share_upper`` would also
    stop it, and nothing distinguishes the two. This asserts the raw link is
    already inside the bracket, so the clamp is inactive on this calibration.
    """
    f = sol.ShareFuncAdj
    mu = np.log(f.m_nodes[f.m_nodes >= f.mBridge])
    raw_omega = np.exp(f.chiFunc(mu))
    assert raw_omega.max() <= 1.0, f"raw omega reached {raw_omega.max():.6f}"


def test_slope_is_continuous_at_both_seams(sol) -> None:
    """Slope matching is the whole reason for a Hermite, so test the slope.

    test_share_is_continuous_across_the_bridge only bounds the JUMP, and at
    eps=1e-7 with atol=1e-5 it tolerates an effective slope of 50 against a
    true slope near 7e-2. This reads the analytic derivative on both sides.
    """
    f = sol.ShareFuncAdj
    eps = 1e-6
    for knot in (f.mRelease, f.mBridge):
        lo = float(f.derivative(np.array([knot - eps]))[0])
        hi = float(f.derivative(np.array([knot + eps]))[0])
        assert abs(hi - lo) < 1e-3, f"slope jump {abs(hi - lo):.3e} at {knot}"


def test_non_contiguous_capped_run_is_reported(sol, caplog) -> None:
    """argmin takes the FIRST uncapped node and ignores capped ones after it.

    A node back on the cap above the release places the release too low and
    shortens the exact region, with nothing visible downstream.
    """
    m = np.array([1.0, 2.0, 3.0, 4.0, 20.0, 100.0])
    s = np.array([1.0, 1.0, 0.98, 1.0, 0.34, 0.315])
    with caplog.at_level(logging.WARNING, logger="portfolio"):
        f = ModeratedShareFunc(m, s, sol.ShareLimit)
    assert f.mRelease == 2.0, "release should sit at the last contiguous cap node"
    assert "not contiguous" in caplog.text, "misplaced release went unreported"


def test_degenerate_all_myopic_solution(sol) -> None:
    """Anchor: a solve pinned at the myopic share must be reproduced exactly.

    This is the closed form the portfolio problem admits - with the cap slack
    everywhere and no wealth effect, the share IS sigmaStar at every m. It also
    exercises OMEGA_CLIP, which no other test reaches: omega is identically 0
    here, so the floor is what keeps log finite.
    """
    sigma = sol.ShareLimit
    m = np.geomspace(1.0, 100.0, 12)
    f = ModeratedShareFunc(m, np.full_like(m, sigma), sigma)
    assert not np.isfinite(f.mRelease), "cap must not bind in the myopic case"
    probe = np.geomspace(0.5, 500.0, 400)
    assert np.allclose(f(probe), sigma, rtol=0, atol=1e-11)


def test_moderating_the_share_leaves_the_recursion_untouched(sol) -> None:
    """The module claims the moderated share does not feed back into the solve.

    HARK reads vPfuncAdj, dvdmFuncFxd, dvdsFuncFxd, vFuncAdj, vFuncFxd, MPCmin
    and hNrm from solution_next, never ShareFuncAdj. If that ever changes, the
    consumption function will move and this test fires.
    """
    vanilla = PortfolioConsumerType(**{**init_portfolio, "cycles": 0, **GRID})
    vanilla.solve()
    m = np.linspace(0.5, 50.0, 300)
    zero = np.zeros_like(m)
    assert np.allclose(
        sol.cFuncAdj(m),
        vanilla.solution[0].cFuncAdj(m),
        rtol=0,
        atol=1e-12,
    )
    assert np.allclose(sol.cFuncFxd(m, zero), vanilla.solution[0].cFuncFxd(m, zero))
