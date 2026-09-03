"""Comprehensive test suite for the Method of Moderation module.

Run with:
    uv run python code/test_moderation.py     # Script mode
    uv run pytest code/test_moderation.py     # Pytest mode
"""

from __future__ import annotations

import moderation
import numpy as np
import plotting
import pytest
import verify_euler
import verify_table
from moderation import (
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
    IndShockMoMCuspConsumerType,
    RiskyAssetMoMConsumerType,
)

# =============================================================================
# Pytest fixtures for running tests with pytest
# =============================================================================


@pytest.fixture(scope="module")
def solved_consumers():
    """Solve all consumer types once per test module."""
    egm = IndShockEGMConsumerType()
    egm.solve()

    mom = IndShockMoMConsumerType()
    mom.solve()

    cusp = IndShockMoMCuspConsumerType()
    cusp.solve()

    stoch = RiskyAssetMoMConsumerType()
    stoch.solve()

    return egm, mom, cusp, stoch


@pytest.fixture(scope="module")
def sol_egm(solved_consumers):
    """EGM solution fixture."""
    return solved_consumers[0].solution[0]


@pytest.fixture(scope="module")
def sol_mom(solved_consumers):
    """MoM solution fixture."""
    return solved_consumers[1].solution[0]


@pytest.fixture(scope="module")
def sol_cusp(solved_consumers):
    """Cusp solution fixture."""
    return solved_consumers[2].solution[0]


@pytest.fixture(scope="module")
def sol_stoch(solved_consumers):
    """Stochastic returns solution fixture."""
    return solved_consumers[3].solution[0]


def _solve_all_consumer_types():
    """Helper: solve all consumer types and return them (for script mode)."""
    egm = IndShockEGMConsumerType()
    egm.solve()

    mom = IndShockMoMConsumerType()
    mom.solve()

    cusp = IndShockMoMCuspConsumerType()
    cusp.solve()

    stoch = RiskyAssetMoMConsumerType()
    stoch.solve()

    return egm, mom, cusp, stoch


def test_all_consumer_types_solve() -> None:
    """Test that all consumer types solve without error."""
    # Just verify solving works; fixtures handle actual solving for other tests
    _solve_all_consumer_types()


def test_consumption_values(sol_egm, sol_mom, sol_cusp, sol_stoch) -> None:
    """Test consumption function values across wealth levels.

    Only MoM carries pinned baselines. The other three solutions are checked
    for the properties any consumption function must have; a loop that merely
    evaluated them and discarded the results used to sit here, left behind
    when the printed comparison table was removed.

    These fixtures use HARK defaults, where `BoroCnstArt = 0.0` binds, so the
    constraint is active at low m and the correct answer there is c = m. The
    first two baseline entries are exactly 0.1 and 0.5 for that reason. They
    previously read 0.172 and 0.606 - infeasible, because the moderated
    function extrapolated below its lowest node (m = 0.798) with no upper
    bound. The tighter-upper-bound clip in TransformedFunctionMoM now pins
    them, so this baseline doubles as the regression test for that clip.
    """
    m_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])

    for name, sol in [("EGM", sol_egm), ("Cusp", sol_cusp), ("StochR", sol_stoch)]:
        c = sol.cFunc(m_test)
        assert np.all(np.isfinite(c)), f"{name}: non-finite consumption"
        assert np.all(c > 0.0), f"{name}: non-positive consumption"
        assert np.all(np.diff(c) > 0.0), f"{name}: consumption not increasing in m"

        on_grid = m_test >= np.asarray(sol.cFunc.x_list)[0]
        assert on_grid.any(), f"{name}: probe lies entirely below the grid"
        assert np.all(c[on_grid] <= m_test[on_grid] - sol.mNrmMin + 1e-9), (
            f"{name}: consumption exceeds the budget on the solved grid"
        )

    # Verify expected baseline values for MoM
    # Note: These values are with linear interpolation (CubicBool=False)
    expected_mom_c = np.array(
        [
            0.10000000,
            0.50000000,
            0.93568397,
            1.48844974,
            3.04444330,
            5.60785359,
            26.06608667,
        ],
    )
    c_mom = sol_mom.cFunc(m_test)
    assert np.allclose(c_mom, expected_mom_c, rtol=1e-6), "MoM consumption mismatch!"


def test_mpc_accuracy(sol_mom, sol_cusp, sol_stoch) -> None:
    """Test that analytical MPC matches numerical derivative.

    Asserts per (solution, m) rather than accumulating into one flag: this
    covers 15 combinations, and a single bare "failed!" cannot say which of
    MoM, Cusp or StochR broke, nor at which wealth level.
    """
    eps = 1e-7
    m_check = np.array([0.1, 0.5, 1.0, 5.0, 10.0])

    for name, sol in [("MoM", sol_mom), ("Cusp", sol_cusp), ("StochR", sol_stoch)]:
        for m in m_check:
            mpc_analytical = sol.cFunc.derivative(m)
            mpc_numerical = (sol.cFunc(m + eps) - sol.cFunc(m - eps)) / (2 * eps)
            rel_err = abs(mpc_analytical - mpc_numerical) / max(
                abs(mpc_numerical),
                1e-10,
            )
            assert rel_err < 1e-5, (
                f"{name} at m={m}: analytical MPC {mpc_analytical:.9f} vs "
                f"numerical {mpc_numerical:.9f}, rel err {rel_err:.3e}"
            )


def test_solution_attributes(sol_mom, sol_cusp, sol_stoch) -> None:
    """Test that all expected solution attributes are present."""
    attrs_common = [
        "cFunc",
        "vFunc",
        "vPfunc",
        "vPPfunc",
        "mNrmMin",
        "hNrm",
        "MPCmin",
        "MPCmax",
    ]
    attrs_bounds = ["Optimist", "Pessimist", "TighterUpperBound"]

    for _name, sol in [("MoM", sol_mom), ("Cusp", sol_cusp), ("StochR", sol_stoch)]:
        missing = [a for a in attrs_common + attrs_bounds if not hasattr(sol, a)]
        assert not missing, f"Missing attributes: {missing}"

    # Check cusp-specific
    assert hasattr(sol_cusp, "mNrmCusp"), "Missing mNrmCusp"

    # Check stochastic-specific
    stoch_attrs = [
        "MPCmin_stochastic",
        "MPCmin_deterministic",
        "OptimistStochastic",
        "PessimistStochastic",
    ]
    for attr in stoch_attrs:
        assert hasattr(sol_stoch, attr), f"Missing {attr}"


def test_consumption_bounds(sol_mom) -> None:
    """Test that consumption stays within theoretical bounds."""
    m_dense = np.linspace(0.01, 20.0, 100)
    c_pes = sol_mom.Pessimist.cFunc(m_dense)
    c_opt = sol_mom.Optimist.cFunc(m_dense)
    c_real = sol_mom.cFunc(m_dense)

    lower_ok = np.all(c_real >= c_pes - 1e-10)
    upper_ok = np.all(c_real <= c_opt + 1e-10)

    assert lower_ok and upper_ok, "Bounds violated!"


@pytest.mark.parametrize("cubic", [False, True], ids=["linear", "cubic"])
def test_value_function(cubic) -> None:
    """Value function is computed when requested, in both interpolation modes.

    The cubic leg covers the Hermite value-function reconstruction path
    (_build_marginal_value_funcs' MargMargValueFuncCRRA branch), which was
    previously untested.
    """
    # Solve with vFuncBool=True
    mom_with_vfunc = IndShockMoMConsumerType(vFuncBool=True, CubicBool=cubic)
    mom_with_vfunc.solve()
    sol = mom_with_vfunc.solution[0]

    m_test = np.array([1.0, 5.0, 10.0])

    # Check that vFunc returns reasonable values
    v_vals = sol.vFunc(m_test)

    # Value should be negative for CRRA utility with rho > 1
    assert np.all(v_vals < 0), "Value should be negative for CRRA > 1"

    # Value should be monotonically increasing in m
    assert np.all(np.diff(v_vals) > 0), "Value should increase with m"


@pytest.fixture(scope="module")
def constrained_vfunc_solutions():
    """MoM and EGM solved with a binding imposed constraint and vFuncBool.

    HARK's defaults set ``BoroCnstArt = 0.0``, which is tighter than the
    natural constraint for this calibration, so the constraint binds over an
    interval. `test_constrained_branch_fixture_really_binds` checks that
    rather than assuming it.
    """
    mom = IndShockMoMConsumerType(vFuncBool=True)
    mom.solve()
    egm = IndShockEGMConsumerType(vFuncBool=True)
    egm.solve()
    return mom.solution[0], egm.solution[0]


def _binding_interval(sol, hi=3.0, n=200000):
    """Excess resources over which the consumption rule is on the budget line."""
    m = sol.mNrmMin + np.linspace(1e-9, hi, n)
    m_ex = m - sol.mNrmMin
    # Relative, not absolute: at m_ex = 1e-9 an absolute 1e-9 tolerance calls
    # every rule "on the line" regardless of what it does.
    gap = m_ex - np.asarray(sol.cFunc(m), dtype=float)
    on_line = gap <= 1e-12 * np.maximum(1.0, m_ex)
    if on_line.all():
        return hi
    if not on_line[0]:
        return 0.0
    return float(m[np.argmax(~on_line)] - sol.mNrmMin)


def test_constrained_branch_fixture_really_binds(constrained_vfunc_solutions) -> None:
    """The fixture must have a constrained region for the next tests to mean anything.

    Without this the exactness assertions below would pass vacuously on a
    calibration where the imposed constraint never binds.
    """
    mom, _ = constrained_vfunc_solutions
    assert mom.mNrmMin == 0.0, "expected the imposed constraint to set mNrmMin"
    assert _binding_interval(mom) > 0.1, "the imposed constraint never binds"


def test_value_is_exact_where_the_constraint_binds(
    constrained_vfunc_solutions,
) -> None:
    """v(m) - u(m - mNrmMin) is one constant across the constrained branch.

    Pinned at the constraint, end-of-period assets do not vary with m, so
    value there is current utility plus w(mNrmMin) and nothing else. The
    constant recovered at any point of the branch must therefore be the same
    constant. Interpolating through the branch instead makes it drift: before
    the exact branch was applied the recovered value scaled like
    1/(m - mNrmMin), which is what this test catches.
    """
    uFunc = moderation.UtilityFuncCRRA(2.0)
    recovered = {}
    for name, sol in zip(("MoM", "EGM"), constrained_vfunc_solutions, strict=True):
        m_ex = np.geomspace(1e-6, _binding_interval(sol) * 0.9, 40)
        v = np.asarray(sol.vFunc(sol.mNrmMin + m_ex), dtype=float)
        wbar_hat = v - uFunc(m_ex)
        assert np.allclose(wbar_hat, wbar_hat[-1], rtol=1e-9), (
            f"{name}: v - u(mEx) drifts across the constrained branch, so the "
            f"branch is being interpolated rather than evaluated"
        )
        recovered[name] = wbar_hat[-1]

    assert recovered["MoM"] == pytest.approx(recovered["EGM"], rel=1e-9), (
        "the two solvers disagree about end-of-period value at the constraint"
    )


def test_inverse_value_at_the_constraint_is_zero(constrained_vfunc_solutions) -> None:
    """vNvrs(mNrmMin) = 0, the value-side analogue of c(mNrmMin) = 0.

    Consumption is zero at the constraint, so value is minus infinity and its
    inverse-utility transform is exactly zero. Moderation cannot reach the
    point by itself: mu = log(m - mNrmMin) sends it to minus infinity.
    """
    for sol in constrained_vfunc_solutions:
        got = float(np.asarray(sol.vFunc.vFuncNvrs(sol.mNrmMin)).ravel()[0])
        assert got == pytest.approx(0.0, abs=1e-12)


def test_inverse_value_slope_at_the_constraint(constrained_vfunc_solutions) -> None:
    """vNvrs'(mNrmMin) = MPCmax^(-rho/(1-rho)), which is 1 when the constraint binds.

    This is the slope HARK inserts at the constraint node and that linear
    interpolation would otherwise discard in favour of a secant to the kink.
    """
    mom, egm = constrained_vfunc_solutions
    for sol in (mom, egm):
        expected = sol.MPCmax ** (-2.0 / (1.0 - 2.0))
        got = (
            float(np.asarray(sol.vFunc.vFuncNvrs(sol.mNrmMin + 1e-9)).ravel()[0]) / 1e-9
        )
        assert got == pytest.approx(expected, rel=1e-6)


def test_constrained_branch_is_inert_under_the_natural_constraint() -> None:
    """No imposed constraint means no constrained branch and no change in value.

    The natural constraint is never strictly binding at feasible m, so
    w(mNrmMin) is minus infinity and the branch must be declined rather than
    built from an infinite constant.
    """
    plain = IndShockMoMConsumerType(vFuncBool=True, BoroCnstArt=None)
    plain.solve()
    sol = plain.solution[0]
    assert sol.mNrmMin < 0.0, "the natural constraint should be negative here"
    assert _binding_interval(sol) < 1e-6, "nothing should bind under BoroCnstArt=None"

    # The guard returns before touching the recursion, so the remaining
    # arguments are never read and are passed as None deliberately.
    declined = moderation.calc_constrained_branch_value(
        sol.mNrmMin, sol.mNrmMin, None, None, None, None, None, None
    )
    assert declined is None, "the branch must be declined when nothing binds"

    m = sol.mNrmMin + np.geomspace(1e-6, 10.0, 50)
    assert np.all(np.isfinite(sol.vFunc.vFuncNvrs(m)))


def test_moderation_ratio_formula(sol_mom) -> None:
    """Test that moderation ratio satisfies ω = (c_real - c_pes) / (c_opt - c_pes) ∈ (0,1).

    This tests equation (7) from the Method of Moderation paper.
    """
    m_test = np.linspace(0.1, 20.0, 50)

    c_real = sol_mom.cFunc(m_test)
    c_pes = sol_mom.Pessimist.cFunc(m_test)
    c_opt = sol_mom.Optimist.cFunc(m_test)

    # Compute moderation ratio: ω = (c_real - c_pes) / (c_opt - c_pes)
    omega = (c_real - c_pes) / (c_opt - c_pes)

    # ω must be strictly in (0, 1)
    assert np.all(omega > 0), "Moderation ratio must be > 0"
    assert np.all(omega < 1), "Moderation ratio must be < 1"

    # ω should increase with m (approaching optimist at high wealth)
    assert np.all(np.diff(omega) > 0), "Moderation ratio should increase with m"


def test_cusp_point_formula(sol_cusp) -> None:
    """Test that cusp point satisfies theoretical formula.

    From appendix eq. (5): mNrmCusp = -hNrmPes + MPCmin*(hNrmOpt - hNrmPes)/(MPCmax - MPCmin)
    """
    # Extract parameters
    mNrmMin = sol_cusp.mNrmMin
    hNrm = sol_cusp.hNrm  # This is hNrmOpt
    MPCmin = sol_cusp.MPCmin
    MPCmax = sol_cusp.MPCmax

    # For standard params, hNrmPes = 0 (unemployment possible)
    hNrmPes = -mNrmMin  # Natural borrowing constraint
    hNrmOpt = hNrm
    hNrmEx = hNrmOpt - hNrmPes

    # Theoretical cusp point formula
    mNrmCusp_theoretical = mNrmMin + (MPCmin * hNrmEx) / (MPCmax - MPCmin)

    assert np.isclose(sol_cusp.mNrmCusp, mNrmCusp_theoretical, rtol=1e-10), (
        "Cusp point mismatch!"
    )


def test_cusp_sparse_grid_degenerate_region_falls_back() -> None:
    """A grid with fewer than two points below the cusp still solves.

    The paper's sparse demonstration grid (5 linearly spaced points up to
    aXtraMax = 4) leaves a single gridpoint below the cusp. The finite-
    difference extrapolation slopes need at least two points per region, so
    the cusp construction must fall back to the standard MoM build (as it
    already does when a region is empty) instead of crashing.
    """
    sparse = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 5, "aXtraNestFac": -1}
    agent = IndShockMoMCuspConsumerType(**sparse)
    agent.solve()
    sol = agent.solution[0]

    assert isinstance(sol.cFunc, moderation.TransformedFunctionMoM), (
        "Linear sparse-grid cusp consumer should fall back to the plain MoM build"
    )
    m = np.linspace(sol.mNrmMin + 0.01, 20.0, 200)
    c_pes = sol.Pessimist.cFunc(m)
    c_opt = sol.Optimist.cFunc(m)
    c = sol.cFunc(m)
    assert np.all(c >= c_pes - 1e-10), "Fallback violated the pessimist bound"
    assert np.all(c <= c_opt + 1e-10), "Fallback violated the optimist bound"


def test_expit_moderate_extreme_chi_no_overflow() -> None:
    """The sigmoid inversion evaluates extreme chi without overflow warnings.

    Extrapolated chi can reach large magnitudes (a one-point cusp region
    extrapolates far past the grid), where the naive 1/(1+exp(-chi)) overflows
    in exp before IEEE arithmetic rescues the value. The implementation must
    compute the correct limits {0, 1} without tripping floating-point flags.
    """
    with np.errstate(over="raise"):
        vals = moderation.expit_moderate(np.array([-800.0, 0.0, 800.0]))
    assert np.allclose(vals, [0.0, 0.5, 1.0]), "Sigmoid limits wrong"


def test_cusp_sparse_grid_cubic_falls_back() -> None:
    """A one-point region falls back to plain MoM in cubic mode too.

    Hermite slopes make a one-point cusp split mechanically constructible,
    but measurement settled the design (2026-07-16): on the Table 1 sparse
    grid the one-knot low region anchors the chi interpolant at mu = -6.9 and
    rides its tangent ~7.5 log-units out to the cusp, giving a sup consumption
    error of 2.4e-1 at the cusp versus 2.9e-3 for the plain-MoM fallback, an
    ~80x accuracy cost for a tighter-bound guarantee wrapped around a badly
    wrong interpolant. Both interpolation modes therefore require two points
    per region and otherwise fall back (losing the tighter MPCmax bound but
    keeping the optimist/pessimist bounds structurally).
    """
    sparse = {"aXtraMin": 0.001, "aXtraMax": 4, "aXtraCount": 5, "aXtraNestFac": -1}
    agent = IndShockMoMCuspConsumerType(**(sparse | {"CubicBool": True}))
    agent.solve()
    sol = agent.solution[0]

    assert isinstance(sol.cFunc, moderation.TransformedFunctionMoM), (
        "Cubic sparse-grid cusp consumer should fall back to the plain MoM "
        "build; the one-point cusp split costs ~80x accuracy at the cusp"
    )
    m = np.linspace(sol.mNrmMin + 0.01, 20.0, 200)
    assert np.all(sol.cFunc(m) >= sol.Pessimist.cFunc(m) - 1e-10), (
        "Fallback violated the pessimist bound"
    )
    assert np.all(sol.cFunc(m) <= sol.Optimist.cFunc(m) + 1e-10), (
        "Fallback violated the optimist bound"
    )


def test_mpc_bounds_everywhere(sol_mom) -> None:
    """Test that MPC stays within [MPCmin, MPCmax] bounds in the grid region.

    This tests the fundamental MPC bounds from buffer-stock theory.
    Note: In the extrapolation region (m < first gridpoint), MPC can exceed
    MPCmax due to the moderated transformation. We test within the grid.
    """
    # Test within the grid region (m >= 1.0 to avoid extrapolation edge)
    m_test = np.linspace(1.0, 50.0, 200)
    mpc = sol_mom.cFunc.derivative(m_test)

    MPCmin = sol_mom.MPCmin
    MPCmax = sol_mom.MPCmax

    # Allow small numerical tolerance
    tol = 1e-6
    lower_ok = np.all(mpc >= MPCmin - tol)
    upper_ok = np.all(mpc <= MPCmax + tol)

    assert lower_ok, f"MPC below MPCmin! Min MPC = {mpc.min()}"
    assert upper_ok, f"MPC above MPCmax! Max MPC = {mpc.max()}"

    # MPC should decrease with wealth (concave consumption function)
    assert np.all(np.diff(mpc) <= tol), "MPC should decrease with m"


def test_hermite_slope_formulas(sol_mom) -> None:
    """Test Hermite interpolation slope formulas from appendix.

    Tests three related formulas:
    - Moderation ratio derivative: ∂ω/∂μ = m_ex*(MPC - κ_min)/(κ_min * h_ex)
    - Logit slope: ∂χ/∂μ = (∂ω/∂μ)/[ω(1-ω)]
    - MPC weight: λ = (κ_min/(κ_max-κ_min)) * (h_ex/m_ex) * ∂ω/∂μ
    """
    # Test at several wealth levels within grid
    m_test = np.array([1.0, 2.0, 5.0, 10.0])

    mNrmMin = sol_mom.mNrmMin
    hNrm = sol_mom.hNrm
    MPCmin = sol_mom.MPCmin
    MPCmax = sol_mom.MPCmax

    hNrmPes = -mNrmMin
    hNrmEx = hNrm - hNrmPes

    for m in m_test:
        mNrmEx = m - mNrmMin

        # Get consumption values
        c_real = sol_mom.cFunc(m)
        c_pes = sol_mom.Pessimist.cFunc(m)
        c_opt = sol_mom.Optimist.cFunc(m)
        MPC = sol_mom.cFunc.derivative(m)

        # Compute moderation ratio
        omega = (c_real - c_pes) / (c_opt - c_pes)

        # Formula 1: Moderation ratio derivative
        # ∂ω/∂μ = m_ex * (MPC - κ_min) / (κ_min * h_ex)
        omega_prime_mu = mNrmEx * (MPC - MPCmin) / (MPCmin * hNrmEx)

        # Verify ω derivative is positive (ω increases with wealth)
        assert omega_prime_mu > 0, (
            f"Formula 1 at m={m}: d(omega)/d(mu) = {omega_prime_mu:.6e}, not positive"
        )

        # Formula 2: Logit slope
        # ∂χ/∂μ = (∂ω/∂μ) / [ω(1-ω)]
        chi_prime_mu = omega_prime_mu / (omega * (1 - omega))

        # Verify χ derivative is positive (χ increases with wealth)
        assert chi_prime_mu > 0, (
            f"Formula 2 at m={m}: d(chi)/d(mu) = {chi_prime_mu:.6e}, not positive"
        )

        # Formula 3: MPC weight (verify MPC = (1-λ)*MPCmin + λ*MPCmax)
        # λ = (κ_min/(κ_max-κ_min)) * (h_ex/m_ex) * ∂ω/∂μ
        mpc_weight = (MPCmin / (MPCmax - MPCmin)) * (hNrmEx / mNrmEx) * omega_prime_mu

        # Reconstruct MPC from weight formula
        MPC_reconstructed = (1 - mpc_weight) * MPCmin + mpc_weight * MPCmax

        # Verify MPC matches reconstruction
        rel_err = abs(MPC - MPC_reconstructed) / MPC
        assert rel_err < 1e-6, (
            f"Formula 3 at m={m}: MPC {MPC:.9f} vs reconstruction "
            f"{MPC_reconstructed:.9f}, rel err {rel_err:.3e}"
        )


@pytest.mark.parametrize("cubic", [False, True], ids=["linear", "cubic"])
def test_no_negative_precautionary_saving_in_extrapolation(cubic) -> None:
    """Paper's central claim: sparse-grid MoM has no negative precautionary saving.

    The reproduction package's headline figure shows that with a 5-point sparse
    grid (the same calibration as Table 1), standard EGM produces negative
    precautionary saving when extrapolating past the top gridpoint, while MoM
    does not. This test asserts the MoM half of that claim quantitatively,
    on the paper's linear baseline AND the cubic Hermite refinement.
    Failure here means the paper's central correctness claim has regressed.
    """
    # Match the Table 1 / verify_table.py calibration (cubic = the refinement).
    params = verify_table.PARAMS | {"CubicBool": cubic}
    sparse_grid = {
        "aXtraMin": 0.001,
        "aXtraMax": 4,
        "aXtraCount": 5,
        "aXtraNestFac": -1,
    }

    mom = IndShockMoMConsumerType(**(params | sparse_grid))
    mom.solve()
    sol_mom = mom.solution[0]

    # The top of the MoM grid is around m = 4; evaluate well past it to test
    # extrapolation specifically.
    m_extrapolate = np.linspace(5.0, 30.0, 100)
    c_opt = sol_mom.Optimist.cFunc(m_extrapolate)
    c_mom = sol_mom.cFunc(m_extrapolate)
    c_pes = sol_mom.Pessimist.cFunc(m_extrapolate)

    # Precautionary saving = c_opt - c_real should remain strictly positive.
    precautionary_gap = c_opt - c_mom
    min_gap = float(precautionary_gap.min())
    assert min_gap > 0.0, (
        f"Sparse-grid MoM produced non-positive precautionary saving "
        f"(min gap = {min_gap}) in the extrapolation region. "
        "This contradicts the paper's central correctness claim."
    )

    # And c_pes < c_mom < c_opt should hold throughout.
    assert np.all(c_mom > c_pes - 1e-12), "MoM consumption fell below pessimist bound"
    assert np.all(c_mom < c_opt + 1e-12), "MoM consumption exceeded optimist bound"


def test_stochastic_mpc_formula() -> None:
    """Test that stochastic returns reduce MPC (more precautionary saving).

    Note: The Merton-Samuelson MPC formula applies to consumers with NO labor income.
    Our stochastic model combines income risk with return risk, so we verify that:
    1. Adding return volatility reduces MPCmin (more precautionary saving)
    2. The stochastic MPCmin is stored correctly alongside deterministic MPCmin
    """
    stoch = RiskyAssetMoMConsumerType()
    stoch.solve()
    sol = stoch.solution[0]

    # Get parameters

    # Verify that return volatility reduces MPC (increases precautionary saving)
    assert sol.MPCmin < sol.MPCmin_deterministic, (
        "Stochastic returns should reduce MPCmin (more precautionary saving)"
    )

    # Verify both MPCmin values are reasonable (between 0 and 1)
    assert 0 < sol.MPCmin < 1, "MPCmin must be in (0, 1)"
    assert 0 < sol.MPCmin_deterministic < 1, "MPCmin_deterministic must be in (0, 1)"

    # Verify MPCmin is close to deterministic (small effect of return volatility)
    # With income risk already present, return risk adds a smaller effect
    mpc_reduction = sol.MPCmin_deterministic - sol.MPCmin
    assert mpc_reduction > 0 and mpc_reduction < 0.1, (
        "Return volatility should have a modest effect on MPC"
    )


def test_stochastic_bracket_contains_realist() -> None:
    """Under return risk the realist stays strictly inside the two bounds.

    This is the numerical check the return-risk section reports in place of a
    proof: the bracket's validity under a stochastic return factor is verified
    rather than proved, so the paper quotes the worst-case slack this test
    measures. Calibration matches the section's text, mean return equal to
    `Rfree` and level standard deviation `0.08`, on the Table 1 parameters.

    The volatility is not free. The section states the return-risk analogue of
    the finite-human-wealth condition as `E[R^-rho] < E[R^(1-rho)]`, and on
    this calibration it fails above roughly `RiskyStd = 0.10`. An earlier draft
    of this test used `0.20`, where the shadow riskless return is `0.946` and
    human wealth does not converge as the horizon grows; quoting a bracket
    check from a parameterization that violates the paper's own existence
    condition is the defect the moment assertion below now prevents.

    The two bounds are the stochastic optimist and pessimist that the solver
    attaches to the solution, not their risk-free counterparts, because return
    risk lowers the limiting MPC and therefore moves both rules.

    Tolerance: containment must be strict at every probe point, so the assertion
    is on positive slack rather than on a numerical tolerance. The probe is
    dense (2000 points) and starts one cent above the borrowing constraint,
    where the bracket is narrowest and the test is hardest to pass.
    """
    params = verify_table.PARAMS | {
        "RiskyAvg": verify_table.PARAMS["Rfree"][0],
        "RiskyStd": 0.08,
    }

    # Gate the calibration before solving on it; see the docstring.
    base = moderation.IndShockConsumerType(**verify_table.PARAMS)
    base.update_income_process()
    joint = moderation.create_joint_distribution(
        base.IncShkDstn[0],
        params["RiskyAvg"],
        params["RiskyStd"],
    )
    risky = np.asarray(joint.atoms[2], dtype=float)
    probs = np.asarray(joint.pmv, dtype=float)
    crra = verify_table.PARAMS["CRRA"]
    m_num = float(np.sum(probs * risky ** (1.0 - crra)))
    m_den = float(np.sum(probs * risky ** (-crra)))
    assert m_den < m_num, (
        f"Calibration violates the return-risk finite-human-wealth condition: "
        f"E[R^-rho] = {m_den:.5f} is not below E[R^(1-rho)] = {m_num:.5f}, so "
        f"the shadow riskless return {m_num / m_den:.4f} is below one. Lower "
        f"RiskyStd; the threshold is near 0.10 on these parameters."
    )

    stoch = RiskyAssetMoMConsumerType(**params)
    stoch.solve()
    sol = stoch.solution[0]

    assert hasattr(sol, "OptimistStochastic"), (
        "Return-risk solution must carry the stochastic bounding rules; without "
        "them this test would silently check the risk-free bracket instead."
    )

    m_grid = np.linspace(sol.mNrmMin + 0.01, 10.0, 2000)
    c_real = sol.cFunc(m_grid)
    c_opt = sol.OptimistStochastic.cFunc(m_grid)
    c_pes = sol.PessimistStochastic.cFunc(m_grid)

    slack_below = np.min(c_real - c_pes)
    slack_above = np.min(c_opt - c_real)

    assert slack_below > 0, (
        f"Realist falls to or below the pessimist under return risk; worst "
        f"margin {slack_below:.3e} at m = {m_grid[np.argmin(c_real - c_pes)]:.4f}"
    )
    assert slack_above > 0, (
        f"Realist reaches or exceeds the optimist under return risk; worst "
        f"margin {slack_above:.3e} at m = {m_grid[np.argmin(c_opt - c_real)]:.4f}"
    )

    # The moderation ratio is what the method actually interpolates, so pin it
    # in its own units too: strict containment in consumption is equivalent to
    # omega in (0, 1), and a violation there is what would break the transform.
    omega = (c_real - c_pes) / (c_opt - c_pes)
    assert np.all((omega > 0) & (omega < 1)), (
        f"Moderation ratio left (0, 1) under return risk: "
        f"min {omega.min():.6f}, max {omega.max():.6f}"
    )


def test_cusp_cubic_low_region_mpc_within_bounds() -> None:
    """Below the cusp, the analytical MPC at solved knots stays in [MPCmin, MPCmax].

    The tight-bound moderation derivative must carry the quotient-rule term:
    for modRte = (c/mNrmEx - MPCmin)/(MPCmax - MPCmin) the mu-derivative is
    (MPC - c/mNrmEx)/(MPCmax - MPCmin), which is negative in the low region
    (the average propensity exceeds the marginal one under concavity).
    Dropping the c/mNrmEx term inflates the reconstructed dc/dm by
    (c/mNrmEx - MPCmin) and pushes the analytical MPC above MPCmax near the
    constraint, violating the very bound the cusp construction enforces.
    """
    agent = IndShockMoMCuspConsumerType(CubicBool=True)
    agent.solve()
    sol = agent.solution[0]
    assert isinstance(sol.cFunc, moderation.TransformedFunctionMoMCusp), (
        "Default grid should build the genuine cusp function"
    )

    mu_knots = np.asarray(sol.cFunc.logitModRteFuncLow.x_list)[1:-1]
    m_knots = sol.mNrmMin + np.exp(mu_knots)
    mpc = np.array([float(sol.cFunc.derivative(m)) for m in m_knots])
    assert np.all(mpc <= sol.MPCmax + 1e-6), (
        f"Analytical MPC exceeds MPCmax below the cusp: "
        f"max {mpc.max():.6f} vs MPCmax {sol.MPCmax:.6f}"
    )
    assert np.all(mpc >= sol.MPCmin - 1e-6), (
        f"Analytical MPC below MPCmin below the cusp: "
        f"min {mpc.min():.6f} vs MPCmin {sol.MPCmin:.6f}"
    )


def test_euler_residual_exact_at_reference_gridpoints() -> None:
    """The c-equivalent Euler residual is ~0 at the reference's own knots.

    EGM computes gridpoint consumption by analytically inverting the Euler
    equation against next period's rule, so at its own knots the residual of
    the interpolant is zero to floating point. This pins the estimator's
    factor conventions (effective discounting, growth normalization, the
    solver's own shock discretization) against the solver's: any mismatch
    shows up as a systematic nonzero residual at every knot.
    """
    truth = verify_euler.solve_consumer(IndShockEGMConsumerType)
    sol = truth.solution[0]
    m_knots = np.asarray(sol.cFunc.x_list)[1:]  # skip the borrowing constraint
    r = verify_euler.euler_residual(
        sol.cFunc,
        m_knots,
        **verify_euler.model_kwargs(truth),
    )
    worst = float(np.max(np.abs(r)))
    assert worst < 1e-9, "Estimator disagrees with the solver at its own knots"


def test_euler_residual_rejects_optimist() -> None:
    """The estimator fires on the optimist's rule with the correct sign.

    The optimist overconsumes everywhere (no precautionary motive and lower
    implied assets), so the Euler-implied consumption is below the rule and
    the residual is strictly negative and economically large. An estimator
    that scores the optimist near zero has no power and must not be used.
    """
    mom = verify_euler.solve_consumer(
        IndShockMoMConsumerType,
        grid=verify_euler.SPARSE_GRID,
    )
    m = np.linspace(1.0, 30.0, 50)
    r = verify_euler.euler_residual(
        mom.solution[0].Optimist.cFunc,
        m,
        **verify_euler.model_kwargs(mom),
    )
    assert np.all(r < 0), "Optimist residual should be negative (overconsumption)"
    assert np.max(np.abs(r)) > 1e-3, "Optimist residual should be economically large"


def _clean_euler_stats():
    """Synthetic protocol stats mirroring the measured verify_euler run.

    Refreshed 2026-08-14 from the linear-candidate / cubic-reference run
    (the paper's baseline configuration); the integration test below keeps
    this fixture from drifting away from the live script again.
    """
    return {
        "sups": {
            "floor (truth-500)": [1.6e-7, 8.8e-12, 8.7e-12, 1.0e-11, 1.6e-11],
            "EGM-5": [7.7e-1, 4.6e-3, 1.1e-3, 4.2e-4, 1.8e-2],
            "MoM-5": [1.7e-1, 2.7e-4, 9.8e-5, 3.6e-5, 4.1e-4],
            "optimist": [1.0, 0.28, 0.12, 0.072, 0.047],
            "pessimist": [1.65, 0.37, 0.24, 0.18, 0.148],
        },
        "gap": {
            "EGM-5": {"rel_err": 3.19, "min": -9.6e-2},
            "MoM-5": {"rel_err": 6.9e-2, "min": 4.7e-2},
        },
        "dist": {
            "truth-500": [1.8e-10, 9.2e-12, 1.5e-11, 2.3e-11, 1.2e-10],
            "MoM-500": [3.9e-10, 2.6e-13, 3.5e-13, 3.5e-13, 3.2e-13],
            "EGM-5": [5.4e-2, 4.2e-3, 1.6e-3, 8.6e-4, 1.4e-1],
            "MoM-5": [1.5e-2, 2.5e-4, 1.4e-4, 7.4e-5, 3.0e-3],
        },
        "ratio0": 0.22,
        "pess_min_signed": 1.0e-3,
    }


def test_euler_protocol_failures_branches() -> None:
    """Every rejection branch of the protocol validator fires; clean passes.

    The protocol is itself an instrument, so it gets the same treatment as
    the residual: known-good data must pass, and each tampered input must be
    rejected. The scale case is the important one: power checks built only
    from ratios pass unchanged if a units bug shrinks every residual by
    orders of magnitude, so the validator must anchor to absolute magnitudes.
    Comparisons must also fail CLOSED on NaN (NaN > x is False, so a check
    written as `if metric > bound: fail` silently passes NaN).
    """
    assert verify_euler.protocol_failures(_clean_euler_stats()) == [], (
        "Clean measured stats must pass the protocol"
    )

    def tampered(mutate):
        stats = _clean_euler_stats()
        mutate(stats)
        return verify_euler.protocol_failures(stats)

    cases = {
        "global 1e-100 rescale (units bug)": lambda s: s["sups"].update(
            {k: [v * 1e-100 for v in vals] for k, vals in s["sups"].items()},
        ),
        "NaN in MoM score": lambda s: s["sups"]["MoM-5"].__setitem__(2, float("nan")),
        "NaN in gap metric": lambda s: s["gap"]["MoM-5"].__setitem__(
            "rel_err",
            float("nan"),
        ),
        "EGM gap loses sign flip": lambda s: s["gap"]["EGM-5"].__setitem__(
            "min",
            5.0e-2,
        ),
        "MoM gap goes negative": lambda s: s["gap"]["MoM-5"].__setitem__(
            "min",
            -1.0e-4,
        ),
        "pessimist sign flips": lambda s: s.__setitem__("pess_min_signed", -1e-6),
        "interval-0 ratio blows past cap": lambda s: s.__setitem__("ratio0", 5.0),
        "MoM worse than EGM in interval 2": lambda s: s["sups"]["MoM-5"].__setitem__(
            2,
            2.0e-3,
        ),
        "optimist rejection dies": lambda s: s["sups"].__setitem__(
            "optimist",
            [1e-12] * 5,
        ),
        "floor above ceiling": lambda s: s["sups"].__setitem__(
            "floor (truth-500)",
            [1e-3] * 5,
        ),
        "MoM dist worse than EGM in interval 0": lambda s: s["dist"][
            "MoM-5"
        ].__setitem__(0, 9.0e-2),
        "reference drifts from exact truth": lambda s: s["dist"].__setitem__(
            "truth-500",
            [1e-4] * 5,
        ),
        "NaN in distance": lambda s: s["dist"]["EGM-5"].__setitem__(3, float("nan")),
        "EGM extrapolation distance collapses": lambda s: s["dist"][
            "EGM-5"
        ].__setitem__(4, 1.0e-3),
        "dense MoM drifts from exact truth": lambda s: s["dist"].__setitem__(
            "MoM-500",
            [1e-4] * 5,
        ),
    }
    for name, mutate in cases.items():
        failures = tampered(mutate)
        assert failures, f"Protocol failed to reject: {name}"


def test_euler_residual_guards() -> None:
    """The estimator rejects nonpositive current consumption and clamps
    infeasible continuation plans to a finite residual of exactly -1.
    """
    mom = verify_euler.solve_consumer(
        IndShockMoMConsumerType,
        grid=verify_euler.SPARSE_GRID,
    )
    kw = verify_euler.model_kwargs(mom)

    def zero_policy(m):
        return np.zeros_like(np.asarray(m, dtype=float))

    with pytest.raises(ValueError, match="nonpositive"):
        verify_euler.euler_residual(zero_policy, np.array([1.0]), **kw)

    def half_policy(m):
        return 0.5 * np.asarray(m, dtype=float)

    def infeasible_next(m):
        return np.asarray(m, dtype=float) - 1.0e9

    r = verify_euler.euler_residual(
        half_policy,
        np.array([1.0]),
        **(kw | {"cfunc_next": infeasible_next}),
    )
    assert np.all(np.isfinite(r)) and np.allclose(r, -1.0), (
        "Infeasible continuation must clamp the residual to exactly -1"
    )


def test_exact_truth_evaluator() -> None:
    """The root-find truth matches EGM knots exactly and collapses the floor.

    In the Table 1 calibration the exact policy at any m is the root of
    c = C*(m - c), where C* involves only the analytic terminal rule and the
    shock discretization, so no reference interpolation enters. Two checks:
    at the 500-point reference's own knots the evaluator must agree with the
    (exact-at-knots) EGM values to float precision, and OFF the knots the
    evaluator's own Euler residual must sit at root-finder tolerance, far
    below the ~1.6e-7 interpolation floor of the gridded reference near the
    constraint.
    """
    truth = verify_euler.solve_consumer(IndShockEGMConsumerType)
    sol = truth.solution[0]
    kw = verify_euler.model_kwargs(truth)
    mNrmMin = sol.mNrmMin

    m_knots = np.asarray(sol.cFunc.x_list)[1:-1:25]  # subsample interior knots
    c_exact = verify_euler.c_true_exact(m_knots, mNrmMin=mNrmMin, **kw)
    knot_gap = float(np.max(np.abs(c_exact - sol.cFunc(m_knots))))
    assert knot_gap < 1e-9, "Exact truth disagrees with EGM at its own knots"

    # Off-grid, including deep in the near-constraint region where the
    # gridded reference's interpolation floor lives.
    m_off = np.concatenate(
        [np.linspace(mNrmMin + 0.005, mNrmMin + 0.5, 15), np.linspace(1.0, 25.0, 15)],
    )

    def c_exact_policy(m):
        return verify_euler.c_true_exact(m, mNrmMin=mNrmMin, **kw)

    r = verify_euler.euler_residual(c_exact_policy, m_off, **kw)
    worst = float(np.max(np.abs(r)))
    assert worst < 1e-9, "Exact truth should reduce the floor to solver tolerance"


def test_verify_table_reproduces_table() -> None:
    """Integration guard: the Table 1 reproduction script passes on live output.

    The suite previously validated only synthetic snapshots, so the paper's
    central claim (MoM <= EGM in every interval) had no CI protection; this
    runs the actual script end to end.
    """
    assert verify_table.main() == 0, (
        "verify_table.py reports a regression against Table 1"
    )


def test_verify_euler_protocol_passes_live() -> None:
    """Integration guard: the Euler-residual protocol passes on live output.

    Keeps the gate constants, the reference configuration, and the synthetic
    fixture above from drifting apart again (in 2026-08 the script was red
    for two commits while the suite reported no failures).
    """
    assert verify_euler.main() == 0, (
        "verify_euler.py protocol failures on the live configuration"
    )


def test_mom_grid_markers_exclude_synthetic_knots() -> None:
    """Figure markers show only true gridpoints, not the synthetic end knots.

    extract_mom_grid_points trims the one synthetic extrapolation knot that
    _construct_mom_interpolants appends at each end of the mu grid; a
    regression here re-marks synthetic knots as gridpoints in the paper's
    five-gridpoint exhibits.
    """
    mom = IndShockMoMConsumerType(**(verify_table.PARAMS | verify_table.SPARSE_GRID))
    mom.solve()
    sol = mom.solution[0]

    raw_mu = np.asarray(sol.cFunc.logitModRteFunc.x_list)
    marker_m, marker_c = plotting.extract_mom_grid_points(sol)
    assert marker_m is not None and marker_c is not None, (
        "marker extraction returned None"
    )
    assert len(marker_m) == len(raw_mu) - 2, (
        f"expected {len(raw_mu) - 2} markers (raw knots minus 2 synthetic), "
        f"got {len(marker_m)}"
    )
    real_m = sol.mNrmMin + np.exp(raw_mu[1:-1])
    assert np.allclose(marker_m, real_m, rtol=0, atol=1e-12), (
        "marker positions do not match the real (trimmed) gridpoints"
    )
    assert np.all(np.isfinite(marker_c)), "non-finite marker values"


def run_all_tests() -> None:
    """Run the complete test suite."""
    # Solve all consumer types
    egm, mom, cusp, stoch = _solve_all_consumer_types()

    # Get solutions
    sol_egm = egm.solution[0]
    sol_mom = mom.solution[0]
    sol_cusp = cusp.solution[0]
    sol_stoch = stoch.solution[0]

    # Run all tests
    test_consumption_values(sol_egm, sol_mom, sol_cusp, sol_stoch)
    test_mpc_accuracy(sol_mom, sol_cusp, sol_stoch)
    test_solution_attributes(sol_mom, sol_cusp, sol_stoch)
    test_consumption_bounds(sol_mom)
    test_value_function(cubic=False)
    test_value_function(cubic=True)

    # Paper formula tests
    test_moderation_ratio_formula(sol_mom)
    test_cusp_point_formula(sol_cusp)
    test_cusp_sparse_grid_degenerate_region_falls_back()
    test_expit_moderate_extreme_chi_no_overflow()
    test_cusp_sparse_grid_cubic_falls_back()
    test_cusp_cubic_low_region_mpc_within_bounds()
    test_mpc_bounds_everywhere(sol_mom)
    test_hermite_slope_formulas(sol_mom)
    test_no_negative_precautionary_saving_in_extrapolation(cubic=False)
    test_no_negative_precautionary_saving_in_extrapolation(cubic=True)
    test_stochastic_mpc_formula()
    test_stochastic_bracket_contains_realist()

    test_euler_residual_exact_at_reference_gridpoints()
    test_euler_residual_rejects_optimist()
    test_euler_protocol_failures_branches()
    test_euler_residual_guards()
    test_exact_truth_evaluator()

    test_verify_table_reproduces_table()
    test_verify_euler_protocol_passes_live()
    test_mom_grid_markers_exclude_synthetic_knots()


if __name__ == "__main__":
    run_all_tests()
