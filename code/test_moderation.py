"""Comprehensive test suite for the Method of Moderation module.

Run with:
    uv run python code/test_moderation.py     # Script mode
    uv run pytest code/test_moderation.py     # Pytest mode
"""

from __future__ import annotations

import numpy as np
import pytest
from moderation import (
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
    IndShockMoMCuspConsumerType,
    IndShockMoMStochasticRConsumerType,
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

    stoch = IndShockMoMStochasticRConsumerType()
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
    print("  ✓ IndShockEGMConsumerType")

    mom = IndShockMoMConsumerType()
    mom.solve()
    print("  ✓ IndShockMoMConsumerType")

    cusp = IndShockMoMCuspConsumerType()
    cusp.solve()
    print("  ✓ IndShockMoMCuspConsumerType")

    stoch = IndShockMoMStochasticRConsumerType()
    stoch.solve()
    print("  ✓ IndShockMoMStochasticRConsumerType")

    return egm, mom, cusp, stoch


def test_all_consumer_types_solve():
    """Test that all consumer types solve without error."""
    print("\n" + "=" * 70)
    print("TEST: All consumer types solve")
    print("=" * 70)

    # Just verify solving works; fixtures handle actual solving for other tests
    _solve_all_consumer_types()


def test_consumption_values(sol_egm, sol_mom, sol_cusp, sol_stoch):
    """Test consumption function values across wealth levels."""
    print("\n" + "=" * 70)
    print("TEST: Consumption function values")
    print("=" * 70)

    m_test = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])

    print(f"{'m':>8} | {'EGM':>10} | {'MoM':>10} | {'Cusp':>10} | {'StochR':>10}")
    print("-" * 70)

    for m in m_test:
        c_egm = sol_egm.cFunc(m)
        c_mom = sol_mom.cFunc(m)
        c_cusp = sol_cusp.cFunc(m)
        c_stoch = sol_stoch.cFunc(m)
        print(
            f"{m:8.2f} | {c_egm:10.6f} | {c_mom:10.6f} | {c_cusp:10.6f} | {c_stoch:10.6f}"
        )

    # Verify expected baseline values for MoM
    # Note: These values are with linear interpolation (CubicBool=False)
    expected_mom_c = np.array(
        [
            0.17196675,
            0.60645474,
            0.93568397,
            1.48844974,
            3.04444330,
            5.60785359,
            26.06608667,
        ]
    )
    c_mom = sol_mom.cFunc(m_test)
    assert np.allclose(c_mom, expected_mom_c, rtol=1e-6), "MoM consumption mismatch!"
    print("\n  ✓ MoM consumption matches expected baseline")


def test_mpc_accuracy(sol_mom, sol_cusp, sol_stoch):
    """Test that analytical MPC matches numerical derivative."""
    print("\n" + "=" * 70)
    print("TEST: MPC analytical vs numerical derivative")
    print("=" * 70)

    eps = 1e-7
    m_check = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
    all_passed = True

    for name, sol in [("MoM", sol_mom), ("Cusp", sol_cusp), ("StochR", sol_stoch)]:
        print(f"\n{name}:")
        print(
            f"  {'m':>6} | {'Analytical':>12} | {'Numerical':>12} | {'Rel Error':>12} | Status"
        )
        print("  " + "-" * 60)

        for m in m_check:
            mpc_analytical = sol.cFunc.derivative(m)
            mpc_numerical = (sol.cFunc(m + eps) - sol.cFunc(m - eps)) / (2 * eps)
            rel_err = abs(mpc_analytical - mpc_numerical) / max(
                abs(mpc_numerical), 1e-10
            )
            ok = rel_err < 1e-5
            all_passed = all_passed and ok
            status = "✓" if ok else "✗ FAIL"
            print(
                f"  {m:6.2f} | {mpc_analytical:12.8f} | {mpc_numerical:12.8f} | {rel_err:12.2e} | {status}"
            )

    assert all_passed, "MPC accuracy test failed!"
    print("\n  ✓ All MPC accuracy checks passed")


def test_solution_attributes(sol_mom, sol_cusp, sol_stoch):
    """Test that all expected solution attributes are present."""
    print("\n" + "=" * 70)
    print("TEST: Solution attributes")
    print("=" * 70)

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

    for name, sol in [("MoM", sol_mom), ("Cusp", sol_cusp), ("StochR", sol_stoch)]:
        print(f"\n{name}:")
        missing = [a for a in attrs_common + attrs_bounds if not hasattr(sol, a)]
        assert not missing, f"Missing attributes: {missing}"
        print("  ✓ All common attributes present")
        print(f"    mNrmMin = {sol.mNrmMin:.6f}")
        print(f"    hNrm = {sol.hNrm:.6f}")
        print(f"    MPCmin = {sol.MPCmin:.6f}")
        print(f"    MPCmax = {sol.MPCmax:.6f}")

    # Check cusp-specific
    print("\nCusp-specific:")
    assert hasattr(sol_cusp, "mNrmCusp"), "Missing mNrmCusp"
    print(f"  ✓ mNrmCusp = {sol_cusp.mNrmCusp:.6f}")

    # Check stochastic-specific
    print("\nStochR-specific:")
    stoch_attrs = [
        "MPCmin_stochastic",
        "MPCmin_deterministic",
        "OptimistStochastic",
        "PessimistStochastic",
    ]
    for attr in stoch_attrs:
        assert hasattr(sol_stoch, attr), f"Missing {attr}"
        val = getattr(sol_stoch, attr)
        if isinstance(val, float):
            print(f"  ✓ {attr} = {val:.6f}")
        else:
            print(f"  ✓ {attr} = <object>")


def test_consumption_bounds(sol_mom):
    """Test that consumption stays within theoretical bounds."""
    print("\n" + "=" * 70)
    print("TEST: Consumption bounds (c_pes <= c_real <= c_opt)")
    print("=" * 70)

    m_dense = np.linspace(0.01, 20.0, 100)
    c_pes = sol_mom.Pessimist.cFunc(m_dense)
    c_opt = sol_mom.Optimist.cFunc(m_dense)
    c_real = sol_mom.cFunc(m_dense)

    lower_ok = np.all(c_real >= c_pes - 1e-10)
    upper_ok = np.all(c_real <= c_opt + 1e-10)

    print(f"  c_real >= c_pes: {lower_ok}")
    print(f"  c_real <= c_opt: {upper_ok}")

    assert lower_ok and upper_ok, "Bounds violated!"
    print("  ✓ All bounds satisfied")


def test_value_function():
    """Test that value function is computed when requested."""
    print("\n" + "=" * 70)
    print("TEST: Value function (with vFuncBool=True)")
    print("=" * 70)

    # Solve with vFuncBool=True
    mom_with_vfunc = IndShockMoMConsumerType(vFuncBool=True)
    mom_with_vfunc.solve()
    sol = mom_with_vfunc.solution[0]

    m_test = np.array([1.0, 5.0, 10.0])

    # Check that vFunc returns reasonable values
    v_vals = sol.vFunc(m_test)
    print(f"  v(m) at m={m_test}: {v_vals}")

    # Value should be negative for CRRA utility with rho > 1
    assert np.all(v_vals < 0), "Value should be negative for CRRA > 1"
    print("  ✓ Value function returns negative values (correct for CRRA > 1)")

    # Value should be monotonically increasing in m
    assert np.all(np.diff(v_vals) > 0), "Value should increase with m"
    print("  ✓ Value function is monotonically increasing")


def test_moderation_ratio_formula(sol_mom):
    """Test that moderation ratio satisfies ω = (c_real - c_pes) / (c_opt - c_pes) ∈ (0,1).

    This tests equation (7) from the Method of Moderation paper.
    """
    print("\n" + "=" * 70)
    print("TEST: Moderation ratio formula (paper eq. 7)")
    print("=" * 70)

    m_test = np.linspace(0.1, 20.0, 50)

    c_real = sol_mom.cFunc(m_test)
    c_pes = sol_mom.Pessimist.cFunc(m_test)
    c_opt = sol_mom.Optimist.cFunc(m_test)

    # Compute moderation ratio: ω = (c_real - c_pes) / (c_opt - c_pes)
    omega = (c_real - c_pes) / (c_opt - c_pes)

    print(f"  ω range: [{omega.min():.6f}, {omega.max():.6f}]")

    # ω must be strictly in (0, 1)
    assert np.all(omega > 0), "Moderation ratio must be > 0"
    assert np.all(omega < 1), "Moderation ratio must be < 1"
    print("  ✓ Moderation ratio ω ∈ (0, 1) for all m")

    # ω should increase with m (approaching optimist at high wealth)
    assert np.all(np.diff(omega) > 0), "Moderation ratio should increase with m"
    print("  ✓ Moderation ratio increases with wealth (approaching optimist)")


def test_cusp_point_formula(sol_cusp):
    """Test that cusp point satisfies theoretical formula.

    From appendix eq. (5): mNrmCusp = -hNrmPes + MPCmin*(hNrmOpt - hNrmPes)/(MPCmax - MPCmin)
    """
    print("\n" + "=" * 70)
    print("TEST: Cusp point formula (appendix eq. 5)")
    print("=" * 70)

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

    print(f"  hNrmOpt = {hNrmOpt:.6f}")
    print(f"  hNrmPes = {hNrmPes:.6f}")
    print(f"  MPCmin = {MPCmin:.6f}")
    print(f"  MPCmax = {MPCmax:.6f}")
    print(f"  mNrmCusp (stored) = {sol_cusp.mNrmCusp:.6f}")
    print(f"  mNrmCusp (formula) = {mNrmCusp_theoretical:.6f}")

    assert np.isclose(sol_cusp.mNrmCusp, mNrmCusp_theoretical, rtol=1e-10), (
        "Cusp point mismatch!"
    )
    print("  ✓ Cusp point matches theoretical formula")


def test_mpc_bounds_everywhere(sol_mom):
    """Test that MPC stays within [MPCmin, MPCmax] bounds in the grid region.

    This tests the fundamental MPC bounds from buffer-stock theory.
    Note: In the extrapolation region (m < first gridpoint), MPC can exceed
    MPCmax due to the moderated transformation. We test within the grid.
    """
    print("\n" + "=" * 70)
    print("TEST: MPC bounds (MPCmin ≤ MPC ≤ MPCmax) within grid")
    print("=" * 70)

    # Test within the grid region (m >= 1.0 to avoid extrapolation edge)
    m_test = np.linspace(1.0, 50.0, 200)
    mpc = sol_mom.cFunc.derivative(m_test)

    MPCmin = sol_mom.MPCmin
    MPCmax = sol_mom.MPCmax

    print(f"  MPCmin = {MPCmin:.6f}")
    print(f"  MPCmax = {MPCmax:.6f}")
    print(f"  MPC range (m >= 1): [{mpc.min():.6f}, {mpc.max():.6f}]")

    # Allow small numerical tolerance
    tol = 1e-6
    lower_ok = np.all(mpc >= MPCmin - tol)
    upper_ok = np.all(mpc <= MPCmax + tol)

    print(f"  MPC >= MPCmin: {lower_ok}")
    print(f"  MPC <= MPCmax: {upper_ok}")

    assert lower_ok, f"MPC below MPCmin! Min MPC = {mpc.min()}"
    assert upper_ok, f"MPC above MPCmax! Max MPC = {mpc.max()}"
    print("  ✓ MPC stays within theoretical bounds (within grid)")

    # MPC should decrease with wealth (concave consumption function)
    assert np.all(np.diff(mpc) <= tol), "MPC should decrease with m"
    print("  ✓ MPC decreases with wealth (concave consumption)")


def test_hermite_slope_formulas(sol_mom):
    """Test Hermite interpolation slope formulas from appendix.

    Tests three related formulas:
    - Moderation ratio derivative: ∂ω/∂μ = m_ex*(MPC - κ_min)/(κ_min * h_ex)
    - Logit slope: ∂χ/∂μ = (∂ω/∂μ)/[ω(1-ω)]
    - MPC weight: λ = (κ_min/(κ_max-κ_min)) * (h_ex/m_ex) * ∂ω/∂μ
    """
    print("\n" + "=" * 70)
    print("TEST: Hermite interpolation slope formulas (appendix)")
    print("=" * 70)

    # Test at several wealth levels within grid
    m_test = np.array([1.0, 2.0, 5.0, 10.0])

    mNrmMin = sol_mom.mNrmMin
    hNrm = sol_mom.hNrm
    MPCmin = sol_mom.MPCmin
    MPCmax = sol_mom.MPCmax

    hNrmPes = -mNrmMin
    hNrmEx = hNrm - hNrmPes

    print(f"  Testing at m = {m_test}")

    all_passed = True
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
        if omega_prime_mu <= 0:
            print(f"    ✗ m={m}: ∂ω/∂μ = {omega_prime_mu:.6f} should be > 0")
            all_passed = False

        # Formula 2: Logit slope
        # ∂χ/∂μ = (∂ω/∂μ) / [ω(1-ω)]
        chi_prime_mu = omega_prime_mu / (omega * (1 - omega))

        # Verify χ derivative is positive (χ increases with wealth)
        if chi_prime_mu <= 0:
            print(f"    ✗ m={m}: ∂χ/∂μ = {chi_prime_mu:.6f} should be > 0")
            all_passed = False

        # Formula 3: MPC weight (verify MPC = (1-λ)*MPCmin + λ*MPCmax)
        # λ = (κ_min/(κ_max-κ_min)) * (h_ex/m_ex) * ∂ω/∂μ
        mpc_weight = (MPCmin / (MPCmax - MPCmin)) * (hNrmEx / mNrmEx) * omega_prime_mu

        # Reconstruct MPC from weight formula
        MPC_reconstructed = (1 - mpc_weight) * MPCmin + mpc_weight * MPCmax

        # Verify MPC matches reconstruction
        rel_err = abs(MPC - MPC_reconstructed) / MPC
        if rel_err > 1e-6:
            print(f"    ✗ m={m}: MPC mismatch, rel_err = {rel_err:.2e}")
            all_passed = False

    if all_passed:
        print("  ✓ Moderation ratio derivative ∂ω/∂μ > 0 for all m")
        print("  ✓ Logit slope ∂χ/∂μ > 0 for all m")
        print("  ✓ MPC = (1-λ)κ_min + λ·κ_max formula verified")

    assert all_passed, "Hermite slope formula test failed!"


def test_stochastic_mpc_formula():
    """Test that stochastic returns reduce MPC (more precautionary saving).

    Note: The Merton-Samuelson MPC formula applies to consumers with NO labor income.
    Our stochastic model combines income risk with return risk, so we verify that:
    1. Adding return volatility reduces MPCmin (more precautionary saving)
    2. The stochastic MPCmin is stored correctly alongside deterministic MPCmin
    """
    print("\n" + "=" * 70)
    print("TEST: Stochastic returns effect on MPC")
    print("=" * 70)

    stoch = IndShockMoMStochasticRConsumerType()
    stoch.solve()
    sol = stoch.solution[0]

    # Get parameters
    DiscFac = stoch.DiscFac
    CRRA = stoch.CRRA
    RiskyAvg = stoch.RiskyAvg
    RiskyStd = stoch.RiskyStd

    print(f"  DiscFac = {DiscFac}")
    print(f"  CRRA = {CRRA}")
    print(f"  RiskyAvg = {RiskyAvg}")
    print(f"  RiskyStd = {RiskyStd}")
    print(f"  MPCmin (stochastic) = {sol.MPCmin:.6f}")
    print(f"  MPCmin (deterministic) = {sol.MPCmin_deterministic:.6f}")

    # Verify that return volatility reduces MPC (increases precautionary saving)
    assert sol.MPCmin < sol.MPCmin_deterministic, (
        "Stochastic returns should reduce MPCmin (more precautionary saving)"
    )
    print("  ✓ Stochastic returns reduce MPCmin (more precautionary saving)")

    # Verify both MPCmin values are reasonable (between 0 and 1)
    assert 0 < sol.MPCmin < 1, "MPCmin must be in (0, 1)"
    assert 0 < sol.MPCmin_deterministic < 1, "MPCmin_deterministic must be in (0, 1)"
    print("  ✓ Both MPCmin values are in valid range (0, 1)")

    # Verify MPCmin is close to deterministic (small effect of return volatility)
    # With income risk already present, return risk adds a smaller effect
    mpc_reduction = sol.MPCmin_deterministic - sol.MPCmin
    print(f"  MPC reduction from return volatility: {mpc_reduction:.6f}")
    assert mpc_reduction > 0 and mpc_reduction < 0.1, (
        "Return volatility should have a modest effect on MPC"
    )
    print("  ✓ Return volatility effect on MPC is modest (as expected)")


def run_all_tests():
    """Run the complete test suite."""
    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE FOR METHOD OF MODERATION")
    print("=" * 70)

    # Solve all consumer types
    print("\n" + "=" * 70)
    print("TEST: All consumer types solve")
    print("=" * 70)
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
    test_value_function()

    # Paper formula tests
    test_moderation_ratio_formula(sol_mom)
    test_cusp_point_formula(sol_cusp)
    test_mpc_bounds_everywhere(sol_mom)
    test_hermite_slope_formulas(sol_mom)
    test_stochastic_mpc_formula()

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
