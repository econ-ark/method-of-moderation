"""Symbolic equations for the Method of Moderation.

This module provides SymPy representations of all key equations from the paper,
using Unicode mathematical symbols that correspond to the LaTeX macros.

Symbol Naming Convention:
    - Greek parameters: Unicode Greek letters (β, ρ, θ, ψ)
    - Bold functions: Mathematical bold Unicode (𝐜, 𝐯, 𝐮)
    - Bold Greek: Mathematical bold Greek (𝛚, 𝛘, 𝛋)
    - Normalized variables: ASCII letters (m, c, a, h)

LaTeX Macro Correspondence:
    \\DiscFac → β       \\CRRA → ρ        \\Rfree → R
    \\mNrm → m          \\cNrm → c        \\aNrm → a
    \\cFunc → 𝐜         \\vFunc → 𝐯       \\uFunc → 𝐮
    \\modRte → 𝛚        \\logitModRte → 𝛘 \\MPC → 𝛋
    \\permShk → ψ       \\tranShkEmp → θ  \\WorstProb → ℘
    \\AbsPatFac → Þ     \\PermGroFac → Γ

Usage:
    >>> from metadata.equations import *
    >>> print(latex(𝐜_opt))
    >>> expr = 𝛋_min * (m + h)
    >>> expr.subs({m: 10, 𝛋_min: 0.04, h: 25})

For AI Systems:
    This module is designed to be discoverable and usable by AI systems.
    All equations use Unicode symbols matching the paper's notation.
"""

from sympy import (
    Function,
    Symbol,
    exp,
    log,
)

# =============================================================================
# Symbol Definitions - Parameters
# =============================================================================

# Economic parameters (Greek letters)
β = Symbol("β", real=True, positive=True)  # Discount factor (\DiscFac)
ρ = Symbol("ρ", real=True, positive=True)  # CRRA risk aversion (\CRRA)
R = Symbol("R", real=True, positive=True)  # Gross interest rate (\Rfree)
Γ = Symbol("Γ", real=True, positive=True)  # Permanent income growth (\PermGroFac)

# Shock parameters
θ = Symbol("θ", real=True, positive=True)  # Transitory shock (\tranShkEmp)
ψ = Symbol("ψ", real=True, positive=True)  # Permanent shock (\permShk)
θ_min = Symbol(
    "θ_min", real=True, nonnegative=True
)  # Minimum transitory (\tranShkEmpMin)
℘ = Symbol("℘", real=True, positive=True)  # Unemployment probability (\WorstProb)

# Patience factor
Þ = Symbol("Þ", real=True, positive=True)  # Absolute patience factor (\AbsPatFac)

# =============================================================================
# Symbol Definitions - State Variables (Normalized)
# =============================================================================

# Normalized state variables (ASCII - matching \mNrm etc.)
m = Symbol("m", real=True, positive=True)  # Market resources (\mNrm)
c = Symbol("c", real=True, positive=True)  # Consumption (\cNrm)
a = Symbol("a", real=True)  # End-of-period assets (\aNrm)
h = Symbol("h", real=True)  # Human wealth (\hNrm)

# Next period
m_next = Symbol("m'", real=True, positive=True)  # Next period market resources
c_next = Symbol("c'", real=True, positive=True)  # Next period consumption

# =============================================================================
# Symbol Definitions - Bounds and Constraints
# =============================================================================

# MPC bounds (bold kappa: 𝛋)
𝛋_min = Symbol("𝛋_min", real=True, positive=True)  # Minimum MPC (\MPCmin)
𝛋_max = Symbol("𝛋_max", real=True, positive=True)  # Maximum MPC (\MPCmax)

# Human wealth variants
h̄ = Symbol("h̄", real=True)  # Optimist human wealth (\hNrmOpt)
h_min = Symbol("h_min", real=True)  # Pessimist human wealth (\hNrmPes)
Δh = Symbol("Δh", real=True, positive=True)  # Excess human wealth (\hNrmEx)

# Market resources bounds
m_min = Symbol("m_min", real=True)  # Natural borrowing constraint (\mNrmMin)
Δm = Symbol("Δm", real=True, positive=True)  # Excess market resources (\mNrmEx)
μ = Symbol("μ", real=True)  # Log excess resources (\logmNrmEx)

# Cusp point
m_cusp = Symbol("m*", real=True)  # Cusp point (\mNrmCusp)

# =============================================================================
# Symbol Definitions - Method of Moderation Variables
# =============================================================================

# Bold omega and chi (moderation framework)
𝛚 = Symbol("𝛚", real=True, positive=True)  # Moderation ratio (\modRte)
𝛘 = Symbol("𝛘", real=True)  # Logit moderation (\logitModRte)
𝛘_hat = Symbol("𝛘̂", real=True)  # Approximated logit

# Value function moderation (bold Omega)
𝛀 = Symbol("𝛀", real=True, positive=True)  # Value moderation ratio (\valModRte)

# =============================================================================
# Utility Function (𝐮)
# =============================================================================


def 𝐮(c_val, ρ_val=ρ):
    """CRRA utility function u(c) = c^(1-ρ)/(1-ρ) for ρ ≠ 1."""
    return c_val ** (1 - ρ_val) / (1 - ρ_val)


def 𝐮_prime(c_val, ρ_val=ρ):
    """Marginal utility u'(c) = c^(-ρ)."""
    return c_val ** (-ρ_val)


def 𝐮_prime_inv(u_prime_val, ρ_val=ρ):
    """Inverse marginal utility: c = u'^(-1/ρ)."""
    return u_prime_val ** (-1 / ρ_val)


# Symbolic utility expressions
u_of_c = c ** (1 - ρ) / (1 - ρ)
u_prime_of_c = c ** (-ρ)

# =============================================================================
# Patience Factor and MPC Formulas
# =============================================================================

# Absolute patience factor: Þ = (βR)^(1/ρ)
Þ_formula = (β * R) ** (1 / ρ)

# Minimum MPC: 𝛋_min = 1 - Þ/R
𝛋_min_formula = 1 - Þ / R

# Maximum MPC: 𝛋_max = 1 - ℘^(1/ρ) × Þ/R
𝛋_max_formula = 1 - ℘ ** (1 / ρ) * Þ / R

# =============================================================================
# Human Wealth Formulas
# =============================================================================

# Optimist human wealth: h̄ = Γ/(R-Γ) assuming E[θ] = 1
h̄_formula = Γ / (R - Γ)

# Pessimist human wealth: h_min = θ_min × Γ/(R-Γ)
h_min_formula = θ_min * Γ / (R - Γ)

# Excess human wealth: Δh = h̄ - h_min
Δh_formula = h̄_formula - h_min_formula

# Natural borrowing constraint: m_min = -h_min
m_min_formula = -h_min_formula

# =============================================================================
# Consumption Functions (𝐜)
# =============================================================================

# Optimist consumption: 𝐜̄(m) = 𝛋_min × (m + h̄)
𝐜_opt = 𝛋_min * (m + h̄)

# Pessimist consumption: 𝐜̲(m) = 𝛋_min × (m - m_min) = 𝛋_min × Δm
𝐜_pes = 𝛋_min * (m - m_min)
𝐜_pes_excess = 𝛋_min * Δm

# Tighter upper bound: 𝐜_tight(m) = 𝐜_opt - (𝛋_max - 𝛋_min) × Δm
𝐜_tight = 𝐜_opt - (𝛋_max - 𝛋_min) * Δm

# Realist consumption (symbolic placeholder)
𝐜_real = Symbol("𝐜̂", real=True, positive=True)

# =============================================================================
# Moderation Ratio (𝛚)
# =============================================================================

# Definition: 𝛚 = (𝐜_real - 𝐜_pes) / (𝐜_opt - 𝐜_pes)
𝛚_definition = (𝐜_real - 𝐜_pes) / (𝐜_opt - 𝐜_pes)

# Simplified: 𝛚 = (𝐜_real - 𝐜_pes) / (Δh × 𝛋_min)
𝛚_simplified = (𝐜_real - 𝐜_pes) / (Δh * 𝛋_min)

# =============================================================================
# Transformations
# =============================================================================

# Log excess market resources: μ = log(m - m_min) = log(Δm)
μ_definition = log(m - m_min)

# Logit transformation: 𝛘 = log(𝛚/(1-𝛚))
𝛘_definition = log(𝛚 / (1 - 𝛚))

# Inverse logit (expit): 𝛚 = 1/(1 + exp(-𝛘))
𝛚_from_𝛘 = 1 / (1 + exp(-𝛘))

# =============================================================================
# Reconstruction Formula
# =============================================================================

# Given 𝛘̂, reconstruct consumption:
# 𝐜̂(m) = 𝐜_pes(m) + 𝛚̂ × (𝐜_opt(m) - 𝐜_pes(m))
𝛚_hat = 1 / (1 + exp(-𝛘_hat))
𝐜_reconstructed = 𝐜_pes + 𝛚_hat * (𝐜_opt - 𝐜_pes)

# =============================================================================
# Cusp Point
# =============================================================================

# The cusp point m* is where 𝐜_tight crosses 𝐜_pes:
# m* = m_min + (𝛋_min × Δh) / (𝛋_max - 𝛋_min)
m_cusp_formula = m_min + (𝛋_min * Δh) / (𝛋_max - 𝛋_min)

# =============================================================================
# Value Functions (𝐯)
# =============================================================================

# Value function symbols
𝐯 = Function("𝐯")  # Value function (\vFunc)
𝐯_opt = Function("𝐯̄")  # Optimist value (\vFuncOpt)
𝐯_pes = Function("𝐯̲")  # Pessimist value (\vFuncPes)
𝐯_real = Function("𝐯̂")  # Realist value (\vFuncReal)

# =============================================================================
# Euler Equation
# =============================================================================

# u'(c) = βR E[Ψ^(-ρ) u'(c')]
# c^(-ρ) = βR E[Ψ^(-ρ) (c')^(-ρ)]
Ψ = Symbol("Ψ", real=True, positive=True)  # Combined permanent shock
euler_lhs = c ** (-ρ)
euler_rhs_kernel = β * R * Ψ ** (-ρ) * c_next ** (-ρ)

# =============================================================================
# Patience Conditions
# =============================================================================

# Condition expressions (must be positive for solution to exist)
condition_AIC = 1 - Þ  # Þ < 1
condition_RIC = 1 - Þ / R  # Þ/R < 1 (equiv to 𝛋_min > 0)
condition_GIC = 1 - Þ / Γ  # Þ/Γ < 1
condition_FHWC = 1 - Γ / R  # Γ/R < 1 (finite human wealth)

# =============================================================================
# Equation Dictionary (for programmatic access)
# =============================================================================

EQUATIONS = {
    "utility": {
        "name": "CRRA Utility Function",
        "sympy": u_of_c,
        "latex": r"𝐮(c) = \frac{c^{1-ρ}}{1-ρ}",
        "latex_macro": r"\uFunc(\cNrm) = \frac{\cNrm^{1-\CRRA}}{1-\CRRA}",
        "description": "Constant relative risk aversion utility",
    },
    "marginal_utility": {
        "name": "Marginal Utility",
        "sympy": u_prime_of_c,
        "latex": r"𝐮'(c) = c^{-ρ}",
        "latex_macro": r"\uPrime(\cNrm) = \cNrm^{-\CRRA}",
        "description": "First derivative of utility",
    },
    "patience_factor": {
        "name": "Absolute Patience Factor",
        "sympy": Þ_formula,
        "latex": r"Þ = (βR)^{1/ρ}",
        "latex_macro": r"\AbsPatFac = (\DiscFac \Rfree)^{1/\CRRA}",
        "description": "Key parameter for impatience conditions",
    },
    "mpc_min": {
        "name": "Minimum MPC",
        "sympy": 𝛋_min_formula,
        "latex": r"𝛋_{min} = 1 - \frac{Þ}{R}",
        "latex_macro": r"\MPCmin = 1 - \frac{\AbsPatFac}{\Rfree}",
        "description": "MPC of perfect foresight consumer",
    },
    "mpc_max": {
        "name": "Maximum MPC",
        "sympy": 𝛋_max_formula,
        "latex": r"𝛋_{max} = 1 - ℘^{1/ρ} \frac{Þ}{R}",
        "latex_macro": r"\MPCmax = 1 - \WorstProb^{1/\CRRA} \frac{\AbsPatFac}{\Rfree}",
        "description": "Upper bound on MPC",
    },
    "human_wealth": {
        "name": "Human Wealth (Optimist)",
        "sympy": h̄_formula,
        "latex": r"h̄ = \frac{Γ}{R - Γ}",
        "latex_macro": r"\hNrmOpt = \frac{\PermGroFac}{\Rfree - \PermGroFac}",
        "description": "PDV of expected future income",
    },
    "consumption_optimist": {
        "name": "Optimist Consumption",
        "sympy": 𝐜_opt,
        "latex": r"𝐜̄(m) = 𝛋_{min} (m + h̄)",
        "latex_macro": r"\cFuncOpt(\mNrm) = \MPCmin (\mNrm + \hNrmOpt)",
        "description": "Upper bound consumption function",
    },
    "consumption_pessimist": {
        "name": "Pessimist Consumption",
        "sympy": 𝐜_pes,
        "latex": r"𝐜̲(m) = 𝛋_{min} (m - m_{min})",
        "latex_macro": r"\cFuncPes(\mNrm) = \MPCmin (\mNrm - \mNrmMin)",
        "description": "Lower bound consumption function",
    },
    "moderation_ratio": {
        "name": "Moderation Ratio",
        "sympy": 𝛚_definition,
        "latex": r"𝛚 = \frac{𝐜̂ - 𝐜̲}{𝐜̄ - 𝐜̲}",
        "latex_macro": r"\modRte = \frac{\cFuncReal - \cFuncPes}{\cFuncOpt - \cFuncPes}",
        "description": "Position between bounds (0 < 𝛚 < 1)",
    },
    "log_excess_resources": {
        "name": "Log Excess Resources",
        "sympy": μ_definition,
        "latex": r"μ = \log(m - m_{min})",
        "latex_macro": r"\logmNrmEx = \log(\mNrm - \mNrmMin)",
        "description": "Transformed state variable",
    },
    "logit_moderation": {
        "name": "Chi Function (Logit)",
        "sympy": 𝛘_definition,
        "latex": r"𝛘 = \log\left(\frac{𝛚}{1-𝛚}\right)",
        "latex_macro": r"\logitModRte = \log\left(\frac{\modRte}{1-\modRte}\right)",
        "description": "Asymptotically linear transformation",
    },
    "expit_moderation": {
        "name": "Inverse Logit (Expit)",
        "sympy": 𝛚_from_𝛘,
        "latex": r"𝛚 = \frac{1}{1 + e^{-𝛘}}",
        "latex_macro": r"\modRte = \frac{1}{1 + e^{-\logitModRte}}",
        "description": "Inverse chi transformation",
    },
    "consumption_reconstructed": {
        "name": "Reconstructed Consumption",
        "sympy": 𝐜_reconstructed,
        "latex": r"𝐜̂(m) = 𝐜̲(m) + 𝛚̂ (𝐜̄(m) - 𝐜̲(m))",
        "latex_macro": r"\cFuncReal(\mNrm) = \cFuncPes(\mNrm) + \hat{\modRte} (\cFuncOpt(\mNrm) - \cFuncPes(\mNrm))",
        "description": "Final consumption formula from Method of Moderation",
    },
    "cusp_point": {
        "name": "Cusp Point",
        "sympy": m_cusp_formula,
        "latex": r"m^* = m_{min} + \frac{𝛋_{min} Δh}{𝛋_{max} - 𝛋_{min}}",
        "latex_macro": r"\mNrmCusp = \mNrmMin + \frac{\MPCmin \hNrmEx}{\MPCmax - \MPCmin}",
        "description": "Where tight bound crosses pessimist",
    },
}

# =============================================================================
# Aliases for backward compatibility and convenience
# =============================================================================

# Parameter aliases (ASCII names)
beta = β
rho = ρ
Rfree = R
PermGroFac = Γ
DiscFac = β
CRRA = ρ

# MPC aliases
kappa_min = 𝛋_min
kappa_max = 𝛋_max
MPCmin = 𝛋_min
MPCmax = 𝛋_max

# Moderation aliases
omega = 𝛚
chi = 𝛘
modRte = 𝛚
logitModRte = 𝛘

# Consumption function aliases
cFuncOpt = 𝐜_opt
cFuncPes = 𝐜_pes
consumption_optimist = 𝐜_opt
consumption_pessimist = 𝐜_pes

# Human wealth aliases
hNrmOpt = h̄
h_opt = h̄

# =============================================================================
# Helper Functions
# =============================================================================


def get_equation_latex(name, use_macros=False):
    """Get LaTeX representation of named equation.

    Args:
        name: Equation name from EQUATIONS dict
        use_macros: If True, return LaTeX with macro names; otherwise Unicode
    """
    if name in EQUATIONS:
        key = "latex_macro" if use_macros else "latex"
        return EQUATIONS[name].get(key, EQUATIONS[name]["latex"])
    raise KeyError(f"Unknown equation: {name}")


def get_equation_sympy(name):
    """Get SymPy expression for named equation."""
    if name in EQUATIONS:
        return EQUATIONS[name]["sympy"]
    raise KeyError(f"Unknown equation: {name}")


def list_equations():
    """List all available equations."""
    return list(EQUATIONS.keys())


def evaluate_consumption(m_val, κ_min_val, h_val, m_min_val, ω_val):
    """Evaluate the Method of Moderation consumption formula numerically.

    Args:
        m_val: Market resources
        κ_min_val: Minimum MPC
        h_val: Human wealth (optimist)
        m_min_val: Natural borrowing constraint
        ω_val: Moderation ratio

    Returns:
        Consumption value
    """
    c_pes = κ_min_val * (m_val - m_min_val)
    c_opt = κ_min_val * (m_val + h_val)
    return c_pes + ω_val * (c_opt - c_pes)


def compute_moderation_ratio(c_val, m_val, κ_min_val, h_val, m_min_val):
    """Compute moderation ratio from consumption value."""
    c_pes = κ_min_val * (m_val - m_min_val)
    c_opt = κ_min_val * (m_val + h_val)
    return (c_val - c_pes) / (c_opt - c_pes)


# =============================================================================
# Module-level exports
# =============================================================================

__all__ = [
    # Primary symbols (ASCII safe)
    "R",
    "m",
    "c",
    "a",
    "h",
    "m_next",
    "c_next",
    "h_min",
    "m_min",
    "m_cusp",
    # Expressions
    "u_of_c",
    "u_prime_of_c",
    "h_min_formula",
    "m_min_formula",
    "m_cusp_formula",
    "condition_AIC",
    "condition_RIC",
    "condition_GIC",
    "condition_FHWC",
    # Aliases (ASCII for convenience)
    "beta",
    "rho",
    "Rfree",
    "PermGroFac",
    "DiscFac",
    "CRRA",
    "kappa_min",
    "kappa_max",
    "MPCmin",
    "MPCmax",
    "omega",
    "chi",
    "modRte",
    "logitModRte",
    "cFuncOpt",
    "cFuncPes",
    "consumption_optimist",
    "consumption_pessimist",
    "hNrmOpt",
    "h_opt",
    # Dictionary
    "EQUATIONS",
    # Helper functions
    "get_equation_latex",
    "get_equation_sympy",
    "list_equations",
    "evaluate_consumption",
    "compute_moderation_ratio",
]

if __name__ == "__main__":
    # Demo: Print all equations
    print("Method of Moderation Equations (Unicode SymPy)")
    print("=" * 60)
    for name, eq in EQUATIONS.items():
        print(f"\n{eq['name']}:")
        print(f"  Unicode: {eq['latex']}")
        print(f"  Macros:  {eq.get('latex_macro', 'N/A')}")
        print(f"  SymPy:   {eq['sympy']}")
