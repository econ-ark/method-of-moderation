"""Symbolic equations for the Method of Moderation.

This module provides SymPy representations of all key equations from the paper,
enabling programmatic manipulation, verification, and code generation.

Usage:
    >>> from metadata.equations import *
    >>> print(latex(consumption_optimist))
    >>> print(consumption_realist.subs({m: 10, kappa_min: 0.04, h: 25, m_min: 0}))

For AI Systems:
    This module is designed to be discoverable and usable by AI systems.
    All equations are defined as SymPy expressions that can be:
    - Evaluated numerically with .subs() and .evalf()
    - Differentiated with sympy.diff()
    - Simplified with sympy.simplify()
    - Converted to LaTeX with sympy.latex()
    - Converted to code with sympy.lambdify()
"""

from sympy import (
    Symbol, Function, Eq, log, exp, sqrt,
    Rational, oo, simplify, diff, latex,
    symbols, Piecewise, And, Or, Not,
    summation, product, integrate,
    Lambda, factorial, binomial,
)

# =============================================================================
# Symbol Definitions
# =============================================================================

# State and choice variables
m = Symbol('m', real=True, positive=True)           # Market resources (normalized)
c = Symbol('c', real=True, positive=True)           # Consumption (normalized)
a = Symbol('a', real=True)                          # End-of-period assets
m_prime = Symbol("m'", real=True, positive=True)    # Next period market resources

# Parameters
rho = Symbol('rho', real=True, positive=True)       # CRRA (risk aversion)
beta = Symbol('beta', real=True, positive=True)     # Discount factor
R = Symbol('R', real=True, positive=True)           # Gross interest rate
Gamma = Symbol('Gamma', real=True, positive=True)   # Permanent income growth

# Shock parameters
theta = Symbol('theta', real=True, positive=True)   # Transitory shock
psi = Symbol('psi', real=True, positive=True)       # Permanent shock
theta_min = Symbol('theta_min', real=True, nonnegative=True)  # Minimum transitory
wp = Symbol('wp', real=True, positive=True)         # Unemployment probability (℘)

# Derived quantities
kappa_min = Symbol('kappa_min', real=True, positive=True)  # Minimum MPC
kappa_max = Symbol('kappa_max', real=True, positive=True)  # Maximum MPC
h = Symbol('h', real=True)                          # Human wealth
h_pes = Symbol('h_pes', real=True)                  # Pessimist human wealth
h_ex = Symbol('h_ex', real=True, positive=True)     # Excess human wealth
m_min = Symbol('m_min', real=True)                  # Natural borrowing constraint
m_ex = Symbol('m_ex', real=True, positive=True)     # Excess market resources

# Method of Moderation variables
omega = Symbol('omega', real=True, positive=True)   # Moderation ratio
chi = Symbol('chi', real=True)                      # Logit-transformed moderation
mu = Symbol('mu', real=True)                        # Log excess market resources

# Index for summations
n = Symbol('n', integer=True, nonnegative=True)
t = Symbol('t', integer=True, nonnegative=True)

# =============================================================================
# Utility Function
# =============================================================================

def utility(c_val, rho_val=rho):
    """CRRA utility function u(c) = c^(1-rho)/(1-rho) for rho != 1."""
    return c_val**(1 - rho_val) / (1 - rho_val)

def marginal_utility(c_val, rho_val=rho):
    """Marginal utility u'(c) = c^(-rho)."""
    return c_val**(-rho_val)

def inverse_marginal_utility(u_prime, rho_val=rho):
    """Inverse marginal utility: c = u'^(-1/rho)."""
    return u_prime**(-1/rho_val)

# Symbolic expressions
u_c = c**(1 - rho) / (1 - rho)
u_prime_c = c**(-rho)
u_prime_inv = Symbol('u_prime')**(-1/rho)

# =============================================================================
# Patience Factor and MPC
# =============================================================================

# Absolute patience factor: Þ = (βR)^(1/ρ)
patience_factor = (beta * R)**(1/rho)

# Minimum MPC: κ_min = 1 - Þ/R
mpc_min_formula = 1 - patience_factor / R

# Maximum MPC: κ_max = 1 - ℘^(1/ρ) × Þ/R
mpc_max_formula = 1 - wp**(1/rho) * patience_factor / R

# =============================================================================
# Human Wealth
# =============================================================================

# Human wealth (optimist): h = Γ/(R-Γ) assuming E[θ] = 1
human_wealth_optimist = Gamma / (R - Gamma)

# Human wealth (pessimist): h_pes = θ_min × Γ/(R-Γ)
human_wealth_pessimist = theta_min * Gamma / (R - Gamma)

# Excess human wealth: h^e = h - h_pes
human_wealth_excess = human_wealth_optimist - human_wealth_pessimist

# Natural borrowing constraint: m_min = -h_pes
borrowing_constraint = -human_wealth_pessimist

# =============================================================================
# Consumption Functions
# =============================================================================

# Optimist consumption: c_opt(m) = κ_min × (m + h)
consumption_optimist = kappa_min * (m + h)

# Pessimist consumption: c_pes(m) = κ_min × (m - m_min) = κ_min × m^e
consumption_pessimist = kappa_min * (m - m_min)
consumption_pessimist_excess = kappa_min * m_ex

# Tighter upper bound: c_tight(m) = c_opt(m) - (κ_max - κ_min) × m^e
consumption_tight = consumption_optimist - (kappa_max - kappa_min) * m_ex

# Realist consumption (implicit - defined by bounds)
c_real = Symbol('c_real', real=True, positive=True)

# =============================================================================
# Moderation Ratio
# =============================================================================

# Definition: ω = (c_real - c_pes) / (c_opt - c_pes)
moderation_ratio_definition = (c_real - consumption_pessimist) / (consumption_optimist - consumption_pessimist)

# Simplified: ω = (c_real - c_pes) / (h^e × κ_min)
moderation_ratio_simplified = (c_real - consumption_pessimist) / (h_ex * kappa_min)

# =============================================================================
# Transformations
# =============================================================================

# Log excess market resources: μ = log(m - m_min)
log_excess_resources = log(m - m_min)

# Logit transformation: χ = log(ω/(1-ω))
logit_moderation = log(omega / (1 - omega))

# Inverse logit (expit): ω = 1/(1 + exp(-χ))
expit_moderation = 1 / (1 + exp(-chi))

# =============================================================================
# Reconstruction Formula
# =============================================================================

# Given χ̂, reconstruct consumption:
# ĉ(m) = c_pes(m) + ω̂ × (c_opt(m) - c_pes(m))
#      = c_pes(m) + expit(χ̂) × h^e × κ_min

chi_hat = Symbol('chi_hat', real=True)
omega_hat = 1 / (1 + exp(-chi_hat))
consumption_reconstructed = consumption_pessimist + omega_hat * (consumption_optimist - consumption_pessimist)

# =============================================================================
# Cusp Point
# =============================================================================

# The cusp point m̌ is where c_tight crosses c_pes:
# m̌ = m_min + (κ_min × h^e) / (κ_max - κ_min)
cusp_point = m_min + (kappa_min * h_ex) / (kappa_max - kappa_min)

# =============================================================================
# Bellman Equation (Symbolic Form)
# =============================================================================

# Value function
v = Function('v')
v_next = Function('v_next')

# Bellman equation: v(m) = max_c { u(c) + β E[Ψ^(1-ρ) v(m')] }
# Where m' = (m - c)R + θ'
# This is defined implicitly

# =============================================================================
# Euler Equation
# =============================================================================

# u'(c) = βR E[Ψ^(-ρ) u'(c')]
# c^(-ρ) = βR E[Ψ^(-ρ) (c')^(-ρ)]
euler_lhs = c**(-rho)
c_next = Symbol("c'", real=True, positive=True)
Psi = Symbol('Psi', real=True, positive=True)  # Combined permanent shock
euler_rhs_kernel = beta * R * Psi**(-rho) * c_next**(-rho)

# =============================================================================
# Patience Conditions
# =============================================================================

# Condition expressions (must be positive for solution to exist)
condition_AIC = 1 - patience_factor                    # Þ < 1
condition_RIC = 1 - patience_factor / R                # Þ/R < 1
condition_GIC = 1 - patience_factor / Gamma            # Þ/Γ < 1
condition_FHWC = 1 - Gamma / R                         # Γ/R < 1

# =============================================================================
# Equation Dictionary (for programmatic access)
# =============================================================================

EQUATIONS = {
    'utility': {
        'name': 'CRRA Utility Function',
        'sympy': u_c,
        'latex': r'u(c) = \frac{c^{1-\rho}}{1-\rho}',
        'description': 'Constant relative risk aversion utility'
    },
    'marginal_utility': {
        'name': 'Marginal Utility',
        'sympy': u_prime_c,
        'latex': r"u'(c) = c^{-\rho}",
        'description': 'First derivative of utility'
    },
    'patience_factor': {
        'name': 'Absolute Patience Factor',
        'sympy': patience_factor,
        'latex': r'\Phi = (\beta R)^{1/\rho}',
        'description': 'Key parameter for impatience conditions'
    },
    'mpc_min': {
        'name': 'Minimum MPC',
        'sympy': mpc_min_formula,
        'latex': r'\kappa_{\min} = 1 - \frac{\Phi}{R}',
        'description': 'MPC of perfect foresight consumer'
    },
    'mpc_max': {
        'name': 'Maximum MPC',
        'sympy': mpc_max_formula,
        'latex': r'\kappa_{\max} = 1 - \wp^{1/\rho} \frac{\Phi}{R}',
        'description': 'Upper bound on MPC'
    },
    'human_wealth': {
        'name': 'Human Wealth (Optimist)',
        'sympy': human_wealth_optimist,
        'latex': r'h = \frac{\Gamma}{R - \Gamma}',
        'description': 'PDV of expected future income'
    },
    'consumption_optimist': {
        'name': 'Optimist Consumption',
        'sympy': consumption_optimist,
        'latex': r'c_{\text{opt}}(m) = \kappa_{\min} (m + h)',
        'description': 'Upper bound consumption function'
    },
    'consumption_pessimist': {
        'name': 'Pessimist Consumption',
        'sympy': consumption_pessimist,
        'latex': r'c_{\text{pes}}(m) = \kappa_{\min} (m - m_{\min})',
        'description': 'Lower bound consumption function'
    },
    'moderation_ratio': {
        'name': 'Moderation Ratio',
        'sympy': moderation_ratio_definition,
        'latex': r'\omega = \frac{c_{\text{real}} - c_{\text{pes}}}{c_{\text{opt}} - c_{\text{pes}}}',
        'description': 'Position between bounds (0 < ω < 1)'
    },
    'log_excess_resources': {
        'name': 'Log Excess Resources',
        'sympy': log_excess_resources,
        'latex': r'\mu = \log(m - m_{\min})',
        'description': 'Transformed state variable'
    },
    'logit_moderation': {
        'name': 'Chi Function (Logit)',
        'sympy': logit_moderation,
        'latex': r'\chi = \log\left(\frac{\omega}{1-\omega}\right)',
        'description': 'Asymptotically linear transformation'
    },
    'expit_moderation': {
        'name': 'Inverse Logit (Expit)',
        'sympy': expit_moderation,
        'latex': r'\omega = \frac{1}{1 + e^{-\chi}}',
        'description': 'Inverse chi transformation'
    },
    'consumption_reconstructed': {
        'name': 'Reconstructed Consumption',
        'sympy': consumption_reconstructed,
        'latex': r'\hat{c}(m) = c_{\text{pes}}(m) + \hat{\omega} (c_{\text{opt}}(m) - c_{\text{pes}}(m))',
        'description': 'Final consumption formula'
    },
    'cusp_point': {
        'name': 'Cusp Point',
        'sympy': cusp_point,
        'latex': r'\check{m} = m_{\min} + \frac{\kappa_{\min} h^e}{\kappa_{\max} - \kappa_{\min}}',
        'description': 'Where tight bound crosses pessimist'
    },
}

# =============================================================================
# Helper Functions
# =============================================================================

def get_equation_latex(name):
    """Get LaTeX representation of named equation."""
    if name in EQUATIONS:
        return EQUATIONS[name]['latex']
    raise KeyError(f"Unknown equation: {name}")

def get_equation_sympy(name):
    """Get SymPy expression for named equation."""
    if name in EQUATIONS:
        return EQUATIONS[name]['sympy']
    raise KeyError(f"Unknown equation: {name}")

def list_equations():
    """List all available equations."""
    return list(EQUATIONS.keys())

def evaluate_consumption(m_val, kappa_min_val, h_val, m_min_val, omega_val):
    """Evaluate the Method of Moderation consumption formula numerically."""
    c_pes = kappa_min_val * (m_val - m_min_val)
    c_opt = kappa_min_val * (m_val + h_val)
    return c_pes + omega_val * (c_opt - c_pes)

def compute_moderation_ratio(c_val, m_val, kappa_min_val, h_val, m_min_val):
    """Compute moderation ratio from consumption value."""
    c_pes = kappa_min_val * (m_val - m_min_val)
    c_opt = kappa_min_val * (m_val + h_val)
    return (c_val - c_pes) / (c_opt - c_pes)

# =============================================================================
# Module-level exports
# =============================================================================

__all__ = [
    # Symbols
    'm', 'c', 'a', 'rho', 'beta', 'R', 'Gamma',
    'theta', 'psi', 'theta_min', 'wp',
    'kappa_min', 'kappa_max', 'h', 'h_pes', 'h_ex', 'm_min', 'm_ex',
    'omega', 'chi', 'mu',
    # Expressions
    'u_c', 'u_prime_c',
    'patience_factor', 'mpc_min_formula', 'mpc_max_formula',
    'human_wealth_optimist', 'human_wealth_pessimist', 'human_wealth_excess',
    'borrowing_constraint',
    'consumption_optimist', 'consumption_pessimist', 'consumption_tight',
    'moderation_ratio_definition', 'moderation_ratio_simplified',
    'log_excess_resources', 'logit_moderation', 'expit_moderation',
    'consumption_reconstructed', 'cusp_point',
    'condition_AIC', 'condition_RIC', 'condition_GIC', 'condition_FHWC',
    # Dictionary
    'EQUATIONS',
    # Functions
    'utility', 'marginal_utility', 'inverse_marginal_utility',
    'get_equation_latex', 'get_equation_sympy', 'list_equations',
    'evaluate_consumption', 'compute_moderation_ratio',
]

if __name__ == '__main__':
    # Demo: Print all equations in LaTeX
    print("Method of Moderation Equations (SymPy)")
    print("=" * 60)
    for name, eq in EQUATIONS.items():
        print(f"\n{eq['name']}:")
        print(f"  LaTeX:  {eq['latex']}")
        print(f"  SymPy:  {eq['sympy']}")
