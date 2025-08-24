"""Plotting functions for Method of Moderation notebook and dashboard.

This module contains all the plotting code for the five key figures that demonstrate
the Method of Moderation's superiority over standard EGM approaches.

Official ECON-ARK Brand Colors Available (ONLY APPROVED):
- ARK_BLUE (#1f476b) - Primary brand blue
- ARK_LIGHTBLUE (#00aeef) - Accent light blue
- ARK_PINK (#ed217c) - Brand pink
- ARK_GREEN (#39b54a) - Brand green
- ARK_YELLOW (#fcb040) - Brand yellow
- ARK_GREY (#676470) - Brand grey

CONSISTENT COLOR ASSIGNMENTS FOR ECONOMIC CONCEPTS:
- Truth/Truth (High-Precision): ARK_BLUE (authoritative, primary)
- MoM (Method of Moderation): ARK_GREEN (success, good method)
- EGM (Endogenous Grid Method): ARK_PINK (problematic method)
- Optimist (Perfect Foresight): ARK_LIGHTBLUE (light, optimistic)
- Pessimist (Worst Case): ARK_YELLOW (warning color)
- Tight (Upper Bound): ARK_GREY (neutral, secondary bound)

CONSISTENT LINE STYLE ASSIGNMENTS FOR FIGURES 1 & 3:
- EGM in Figure 1 (failure case): Dashed line ("--")
- MoM in Figure 3 (success case): Dashed line ("--")
- All other methods: Solid line ("-")

The get_concept_color() and get_concept_linestyle() functions automatically assign
consistent colors and line styles based on method names and figure context,
ensuring visual consistency across all plots in both plotting.py and dashboard.md.
This creates a unified visual language for the research.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from moderation import expit_moderate
from style import (
    ALPHA_HIGH,
    ALPHA_LOW,
    ALPHA_MEDIUM,
    ALPHA_MEDIUM_LOW,
    ALPHA_OPAQUE,
    ALPHA_VERY_HIGH,
    ARK_BLUE,
    ARK_GREEN,
    ARK_GREY,
    ARK_LIGHTBLUE,
    ARK_PINK,
    ARK_YELLOW,
    BBOX_PADDING,
    CONCEPT_COLORS,
    FONT_SIZE_LARGE,
    FONT_SIZE_SMALL,
    FONT_SIZE_XLARGE,
    GRID_ALPHA,
    LINE_STYLE_DASHDOT,
    LINE_STYLE_DASHED,
    LINE_STYLE_DOTTED,
    LINE_WIDTH_EXTRA_THICK,
    LINE_WIDTH_MEDIUM,
    LINE_WIDTH_THICK,
    LINE_WIDTH_THIN,
    MARKER_EDGE_COLOR,
    MARKER_EDGE_WIDTH_THICK,
    MARKER_EDGE_WIDTH_THIN,
    MARKER_SIZE_LARGE,
    MARKER_SIZE_STANDARD,
    PADDING_RATIO,
    REFERENCE_LINE_ALPHA,
    REFERENCE_LINE_COLOR,
    REFERENCE_LINE_WIDTH,
    YLIM_MODERATION_RATIO,
    YLIM_PRECAUTIONARY_GAPS,
    YLIM_VALUE_FUNCTION,
    setup_figure,
)


def extract_mom_grid_points(solution, grid_type="consumption"):
    """Extract interpolation grid points from MoM solution.

    Parameters
    ----------
    solution : ConsumerSolution
        MoM solution containing TransformedFunctionMoM
    grid_type : str
        Either "consumption", "value", or "mpc" to specify which grid to extract

    Returns
    -------
    tuple
        (grid_points_m, grid_points_y) where y is consumption, value, or mpc

    """
    from moderation import exp_mu

    try:
        if grid_type == "consumption":
            # MoM cFunc is directly a TransformedFunctionMoM
            mu_grid = solution.cFunc.logitModRteFunc.x_list
            m_min = solution.cFunc.mNrmMin

            # Convert mu back to market resources using proper inverse function
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_c = solution.cFunc(grid_points_m)
            return grid_points_m, grid_points_c

        if grid_type == "value":
            # MoM vFunc is ValueFuncCRRA containing TransformedFunctionMoM
            mu_grid = solution.vFunc.vFuncNvrs.logitModRteFunc.x_list
            m_min = solution.vFunc.vFuncNvrs.mNrmMin

            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_v = solution.vFunc(grid_points_m)
            return grid_points_m, grid_points_v

        if grid_type == "mpc":
            # For MPC, use consumption function grid points and evaluate derivative at those points
            mu_grid = solution.cFunc.logitModRteFunc.x_list
            m_min = solution.cFunc.mNrmMin

            # Convert mu back to market resources using proper inverse function
            grid_points_m = exp_mu(mu_grid, m_min)
            # Get MPC values by evaluating derivative at the grid points
            grid_points_mpc = solution.cFunc.derivative(grid_points_m)
            return grid_points_m, grid_points_mpc

    except Exception:
        pass

    return None, None


def extract_egm_grid_points(solution, grid_type="consumption"):
    """Extract interpolation grid points from EGM solution.

    Parameters
    ----------
    solution : ConsumerSolution
        EGM solution containing CubicInterp or LinearInterp functions
    grid_type : str
        Either "consumption", "value", or "mpc" to specify which grid to extract

    Returns
    -------
    tuple
        (grid_points_m, grid_points_y) where y is consumption, value, or mpc

    """
    if grid_type == "consumption":
        # EGM cFunc is CubicInterp or LinearInterp - both have x_list, y_list
        return solution.cFunc.x_list.copy(), solution.cFunc.y_list.copy()

    if grid_type == "value":
        # EGM vFunc is ValueFuncCRRA with vFuncNvrs attribute containing the interpolant
        # Use x_list from vFuncNvrs but evaluate the actual value function at those points
        grid_points_m = solution.vFunc.vFuncNvrs.x_list.copy()
        grid_points_v = solution.vFunc(grid_points_m)
        return grid_points_m, grid_points_v

    if grid_type == "mpc":
        # For EGM MPC, use consumption grid and evaluate MPC function
        grid_m = solution.cFunc.x_list.copy()
        grid_mpc = solution.vPfunc(grid_m)
        return grid_m, grid_mpc

    return None, None


def plot_moderation_ratio(
    m_grid: np.ndarray,
    omega_values: np.ndarray,
    title: str,
    subtitle: str,
    solution=None,
    grid_points: bool = True,
    grid_type: str = "consumption",
) -> None:
    r"""Plot moderation ratio $\\omega(m)$ showing how realist moderates between bounds.

    Parameters
    ----------
    m_grid : np.ndarray
        Market resources grid for evaluation
    omega_values : np.ndarray
        Moderation ratio values $\\omega(m)$
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    solution : ConsumerSolution, optional
        Solution object to extract grid points from, by default None
    grid_points : bool, optional
        Whether to show grid points, by default True
    grid_type : str, optional
        Type of grid to extract ("consumption" or "value"), by default "consumption"

    """
    # Validate parameters
    if grid_points and solution is None:
        msg = "solution parameter is required when grid_points=True"
        raise ValueError(msg)

    fig, ax = setup_figure(title=title)

    # Plot moderation ratio
    ax.plot(
        m_grid,
        omega_values,
        label="Moderation Ratio $\\omega(m)$",
        color=CONCEPT_COLORS["mom"],
        linewidth=LINE_WIDTH_THICK,
    )

    # Extract and plot interpolation grid points if solution provided
    if solution is not None:
        grid_points_m, grid_points_y = extract_mom_grid_points(solution, grid_type)
        if grid_points_m is not None:
            # For moderation ratio plots, we need to calculate omega from the grid points
            if grid_type == "consumption":
                # MoM cFunc is TransformedFunctionMoM - get chi values directly
                chi_values = solution.cFunc.logitModRteFunc.y_list
                # Convert chi to omega using expit_moderate
                grid_points_omega = expit_moderate(chi_values)

                ax.scatter(
                    grid_points_m,
                    grid_points_omega,
                    label="Grid Points",
                    color=CONCEPT_COLORS["mom"],
                    s=MARKER_SIZE_STANDARD,
                    zorder=5,
                    edgecolors=MARKER_EDGE_COLOR,
                    linewidths=MARKER_EDGE_WIDTH_THIN,
                )
            elif grid_type == "value":
                # MoM vFunc contains TransformedFunctionMoM
                chi_values = solution.vFunc.vFuncNvrs.logitModRteFunc.y_list
                grid_points_omega = expit_moderate(chi_values)

                ax.scatter(
                    grid_points_m,
                    grid_points_omega,
                    label="Grid Points",
                    color=CONCEPT_COLORS["mom"],
                    s=MARKER_SIZE_STANDARD,
                    zorder=5,
                    edgecolors=MARKER_EDGE_COLOR,
                    linewidths=MARKER_EDGE_WIDTH_THIN,
                )

    # Add reference lines with concept colors
    ax.axhline(
        y=0,
        color=CONCEPT_COLORS["optimist"],
        linestyle="-",
        linewidth=LINE_WIDTH_THIN,
        alpha=ALPHA_HIGH,
        label="$\\omega = 0$ (Optimist behavior)",
    )
    ax.axhline(
        y=1,
        color=CONCEPT_COLORS["pessimist"],
        linestyle=LINE_STYLE_DASHED,
        linewidth=LINE_WIDTH_THIN,
        alpha=ALPHA_HIGH,
        label="$\\omega = 1$ (Pessimist behavior)",
    )

    ax.set_xlabel("Normalized Market Resources (m)", fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel("Moderation Ratio $\\omega(m)$", fontsize=FONT_SIZE_LARGE)
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference lines at x=0 and y=0
    ax.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    ax.set_ylim(*YLIM_MODERATION_RATIO)

    # Automatically set x-axis limits with padding
    x_range = m_grid.max() - m_grid.min()
    padding = PADDING_RATIO * x_range  # 5% padding on each side
    ax.set_xlim(m_grid.min() - padding, m_grid.max() + padding)

    plt.tight_layout()


def plot_chi_function(
    mu_grid: np.ndarray,
    chi_values: np.ndarray,
    title: str,
    subtitle: str,
    solution=None,
    grid_points: bool = True,
    grid_type: str = "consumption",
) -> None:
    r"""Plot chi function $\\chi(\\mu)$ showing the logit-transformed moderation ratio.

    The chi function is always plotted in $\\mu$ space (log excess market resources)
    as this is its natural mathematical domain.

    Parameters
    ----------
    mu_grid : np.ndarray
        Log excess market resources grid ($\\mu$) for evaluation
    chi_values : np.ndarray
        Chi function values $\\chi(\\mu)$
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    solution : ConsumerSolution, optional
        Solution object to extract grid points from, by default None
    grid_points : bool, optional
        Whether to show grid points, by default True
    grid_type : str, optional
        Type of grid to extract ("consumption" or "value"), by default "consumption"

    """
    # Validate parameters
    if grid_points and solution is None:
        msg = "solution parameter is required when grid_points=True"
        raise ValueError(msg)

    fig, ax = setup_figure(title=title)

    # Plot chi function
    ax.plot(
        mu_grid,
        chi_values,
        label="Chi Function $\\chi(\\mu) = \\text{logit}(\\omega)$",
        color=CONCEPT_COLORS["mom"],
        linewidth=LINE_WIDTH_THICK,
    )

    # Extract and plot interpolation grid points if requested
    if grid_points and solution is not None:
        # Extract grid points directly from the logitModRteFunc (always in $\\mu$ space)
        if grid_type == "consumption":
            # MoM cFunc is TransformedFunctionMoM
            grid_points_x = solution.cFunc.logitModRteFunc.x_list  # $\\mu$ values
            grid_points_chi = solution.cFunc.logitModRteFunc.y_list  # $\\chi$ values
        elif grid_type == "value":
            # MoM vFunc contains TransformedFunctionMoM
            grid_points_x = solution.vFunc.vFuncNvrs.logitModRteFunc.x_list
            grid_points_chi = solution.vFunc.vFuncNvrs.logitModRteFunc.y_list
        else:
            grid_points_x = None
            grid_points_chi = None

        if grid_points_x is not None and grid_points_chi is not None:
            ax.scatter(
                grid_points_x,
                grid_points_chi,
                label="Grid Points",
                color=CONCEPT_COLORS["mom"],
                s=MARKER_SIZE_STANDARD,
                zorder=5,
                edgecolors=MARKER_EDGE_COLOR,
                linewidths=MARKER_EDGE_WIDTH_THIN,
            )

    ax.set_xlabel("Log Excess Market Resources ($\\mu$)", fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel("Chi Function $\\chi(\\mu)$", fontsize=FONT_SIZE_LARGE)
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference lines at x=0 and y=0
    ax.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
        label="$\\chi = 0$ ($\\omega = 0.5$, balanced moderation)",
    )
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    plt.tight_layout()


def get_concept_color(method_name: str) -> str:
    """Get consistent color for economic concept/method.

    Parameters
    ----------
    method_name : str
        Name of the method/concept (case-insensitive)

    Returns
    -------
    str
        Hex color code for the concept

    """
    name_lower = method_name.lower()

    # Map various method name variants to core concepts
    if "truth" in name_lower or "high-precision" in name_lower:
        return CONCEPT_COLORS["truth"]
    if "mom" in name_lower or "moderation" in name_lower:
        return CONCEPT_COLORS["mom"]
    if "egm" in name_lower or "endogenous" in name_lower:
        return CONCEPT_COLORS["egm"]
    if "optimist" in name_lower or "perfect" in name_lower:
        return CONCEPT_COLORS["optimist"]
    if "pessimist" in name_lower or "worst" in name_lower:
        return CONCEPT_COLORS["pessimist"]
    if "tight" in name_lower:
        return CONCEPT_COLORS["tight"]
    # Fallback to cycling through remaining colors for unknown methods
    fallback_colors = [
        ARK_BLUE,
        ARK_GREEN,
        ARK_PINK,
        ARK_LIGHTBLUE,
        ARK_YELLOW,
        ARK_GREY,
    ]
    return fallback_colors[hash(name_lower) % len(fallback_colors)]


def get_concept_linestyle(method_name: str, figure_num: int) -> str:
    """Get appropriate line style for economic concept/method in specific figure.

    Parameters
    ----------
    method_name : str
        Name of the method/concept (case-insensitive)
    figure_num : int
        Figure number (1 or 3 for precautionary gaps plots)

    Returns
    -------
    str
        Line style string ("-", "--", "-.", ":")

    """
    name_lower = method_name.lower()

    # Specific dashed line requirements for Figures 1 and 3
    if figure_num == 1 and ("egm" in name_lower or "endogenous" in name_lower):
        return "--"  # EGM dashed in Figure 1
    if figure_num == 3 and ("mom" in name_lower or "moderation" in name_lower):
        return "--"  # MoM dashed in Figure 3
    return "-"  # Default solid line for all other cases


def plot_precautionary_gaps(
    m_grid: np.ndarray,
    truth_gap: np.ndarray,
    approx_gap: np.ndarray | list[np.ndarray],
    legend: str | list[str],
    title: str,
    subtitle: str,
    solution=None,
    grid_points: bool = True,
) -> None:
    """Plot precautionary saving gaps comparing truth vs approximation(s).

    Parameters
    ----------
    m_grid : np.ndarray
        Wealth grid for evaluation
    truth_gap : np.ndarray
        Truth precautionary saving gap values
    approx_gap : np.ndarray | list[np.ndarray]
        Approximation gap values (single array or list of arrays for multiple methods)
    legend : str | list[str]
        Legend labels (single string or list of strings for multiple methods)
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    solution : object
        Solution object to extract grid points from
    grid_points : bool, optional
        Whether to show grid points, by default True

    """
    # Validate parameters
    if grid_points and solution is None:
        msg = "solution parameter is required when grid_points=True"
        raise ValueError(msg)

    fig, ax = setup_figure(title=title)

    # Plot truth gap with consistent color
    ax.plot(
        m_grid,
        truth_gap,
        label="Truth",
        color=CONCEPT_COLORS["truth"],
        linewidth=LINE_WIDTH_THICK,
    )

    # Handle single or multiple approximation methods
    if not isinstance(approx_gap, list):
        approx_gap = [approx_gap]
    if not isinstance(legend, list):
        legend = [legend]

    # Plot each approximation method
    for _i, (gap_vals, method_label) in enumerate(
        zip(approx_gap, legend, strict=False),
    ):
        color = get_concept_color(method_label)
        # Use dashed line for approximations to distinguish from truth
        # Centralize linestyle policy
        linestyle = get_concept_linestyle(method_label, figure_num=1)

        ax.plot(
            m_grid,
            gap_vals,
            label=method_label,
            color=color,
            linewidth=LINE_WIDTH_THICK,
            linestyle=linestyle,
        )

    # Extract and plot grid points if requested
    if grid_points and solution is not None:
        try:
            # Determine solution type and extract appropriate grid points
            if type(solution.cFunc).__name__ == "TransformedFunctionMoM":
                # This is a MoM solution
                grid_points_m, grid_points_c = extract_mom_grid_points(
                    solution,
                    "consumption",
                )
                if grid_points_m is not None and len(grid_points_m) > 1:
                    grid_boundary = grid_points_m[-2]  # MoM: second-to-last point
                else:
                    grid_boundary = (
                        grid_points_m[-1] if grid_points_m is not None else None
                    )
            else:
                # This is an EGM solution (CubicInterp or LinearInterp)
                grid_points_m, grid_points_c = extract_egm_grid_points(
                    solution,
                    "consumption",
                )
                grid_boundary = (
                    grid_points_m[-1] if grid_points_m is not None else None
                )  # EGM: last point

            # Plot grid points if successfully extracted
            if grid_points_m is not None and grid_points_c is not None:
                # For grid points on a gap plot, we need to show where the interpolation nodes
                # are located on the gap curve. We interpolate the gap values at grid point locations.
                method_label = legend[0] if isinstance(legend, list) else legend

                # Get the gap values at grid point locations by interpolation
                # This shows where the actual EGM/MoM interpolation nodes fall on the gap curve
                gap_vals = (
                    approx_gap if not isinstance(approx_gap, list) else approx_gap[0]
                )
                gap_at_grid_points = np.interp(grid_points_m, m_grid, gap_vals)

                # Plot actual grid points as scatter
                grid_color = get_concept_color(method_label)
                ax.scatter(
                    grid_points_m,
                    gap_at_grid_points,
                    label="Grid Points",
                    color=grid_color,
                    s=MARKER_SIZE_STANDARD,
                    zorder=5,
                    edgecolors=MARKER_EDGE_COLOR,
                    linewidths=MARKER_EDGE_WIDTH_THIN,
                )

                # Also plot grid boundary line
                if grid_boundary is not None:
                    ax.axvline(
                        x=grid_boundary,
                        color="gray",
                        linestyle=LINE_STYLE_DASHED,
                        alpha=ALPHA_MEDIUM,
                        label="Grid boundary",
                    )

        except Exception:
            # If grid extraction fails, continue without grid points
            pass

    ax.set_xlabel("Normalized Market Resources (m)", fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel("Precautionary Saving Gap", fontsize=FONT_SIZE_LARGE)
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference lines at x=0 and y=0
    ax.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    ax.set_ylim(*YLIM_PRECAUTIONARY_GAPS)

    # Automatically set x-axis limits with padding
    x_range = m_grid.max() - m_grid.min()
    padding = PADDING_RATIO * x_range  # 5% padding on each side
    ax.set_xlim(m_grid.min() - padding, m_grid.max() + padding)

    plt.tight_layout()


def plot_consumption_bounds(
    m_grid: np.ndarray,
    c_main: np.ndarray,
    c_opt: np.ndarray,
    c_pes: np.ndarray,
    title: str,
    subtitle: str,
    legend: str,
    c_tight: np.ndarray | None = None,
    solution=None,
    grid_points: bool = True,
    grid_type: str = "consumption",
) -> None:
    """Plot consumption function with theoretical bounds.

    Unified function for both Figure 2 (Truth bounds) and Figure 4 (MoM bounds).
    Automatically shows tight upper bound if c_tight is provided.

    Parameters
    ----------
    m_grid : np.ndarray
        Wealth grid for evaluation
    c_main : np.ndarray
        Main consumption function to plot (truth or MoM)
    c_opt : np.ndarray
        Optimist consumption values
    c_pes : np.ndarray
        Pessimist consumption values
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    legend : str
        Legend label for the main consumption function
    c_tight : np.ndarray | None, optional
        Tight upper bound consumption values, by default None. If provided, will be plotted automatically.
    solution : ConsumerSolution, optional
        Solution object to extract grid points from, by default None
    grid_points : bool, optional
        Whether to show grid points, by default True
    grid_type : str, optional
        Type of grid to extract ("consumption"), by default "consumption"

    """
    # Validate parameters
    if grid_points and solution is None:
        msg = "solution parameter is required when grid_points=True"
        raise ValueError(msg)

    fig, ax = setup_figure(title=title)

    # Plot bounds first with consistent colors
    ax.plot(
        m_grid,
        c_opt,
        label="Optimist",
        color=CONCEPT_COLORS["optimist"],
        linewidth=LINE_WIDTH_THICK,
        linestyle=LINE_STYLE_DASHED,
        alpha=ALPHA_OPAQUE,
    )
    ax.plot(
        m_grid,
        c_pes,
        label="Pessimist",
        color=CONCEPT_COLORS["pessimist"],
        linewidth=LINE_WIDTH_THICK,
        linestyle=LINE_STYLE_DOTTED,
        alpha=ALPHA_OPAQUE,
    )

    # Plot tight bound if provided
    if c_tight is not None:
        ax.plot(
            m_grid,
            c_tight,
            label="Tighter Upper Bound",
            color=CONCEPT_COLORS["tight"],
            linewidth=LINE_WIDTH_THIN,
            linestyle=LINE_STYLE_DASHDOT,
            alpha=ALPHA_HIGH,
        )

    # Plot main consumption function with appropriate color based on label
    main_color = get_concept_color(legend)
    ax.plot(
        m_grid,
        c_main,
        label=legend,
        color=main_color,
        linewidth=LINE_WIDTH_EXTRA_THICK,
    )

    # Extract and plot interpolation grid points if requested
    if grid_points and solution is not None:
        grid_points_m, grid_points_c = extract_mom_grid_points(solution, grid_type)
        if grid_points_m is not None and grid_points_c is not None:
            ax.scatter(
                grid_points_m,
                grid_points_c,
                label="Grid Points",
                color=main_color,
                s=MARKER_SIZE_STANDARD,
                zorder=5,
                edgecolors=MARKER_EDGE_COLOR,
                linewidths=MARKER_EDGE_WIDTH_THIN,
            )

    # Fill regions to show bounds
    ax.fill_between(
        m_grid,
        c_pes,
        c_opt,
        alpha=ALPHA_MEDIUM_LOW,
        color=main_color,
        label="Feasible region",
    )

    ax.set_xlabel("Normalized Market Resources (m)", fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel("Consumption (c)", fontsize=FONT_SIZE_LARGE)
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference lines at x=0 and y=0
    ax.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    # Automatically set x-axis limits with padding
    x_range = m_grid.max() - m_grid.min()
    padding = PADDING_RATIO * x_range  # 5% padding on each side
    x_min = m_grid.min() - padding
    x_max = m_grid.max() + padding
    ax.set_xlim(x_min, x_max)

    # Set y-axis limits based only on data within visible x-range
    # Find indices of data points within the visible x-range
    visible_mask = (m_grid >= x_min) & (m_grid <= x_max)
    if visible_mask.any():
        # Get all y-values within the visible range
        y_values_visible = []
        y_values_visible.extend(c_main[visible_mask])
        y_values_visible.extend(c_opt[visible_mask])
        y_values_visible.extend(c_pes[visible_mask])
        if c_tight is not None:
            y_values_visible.extend(c_tight[visible_mask])

        # Calculate y-range with some padding
        y_min = min(y_values_visible)
        y_max = max(y_values_visible)
        y_range = y_max - y_min
        y_padding = PADDING_RATIO * y_range  # 5% padding on each side
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.tight_layout()


def plot_mom_mpc(
    m_grid: np.ndarray,
    mpc_values: np.ndarray,
    mpc_opt_vals: np.ndarray,
    mpc_tight_vals: np.ndarray,
    title: str,
    subtitle: str,
    mpc_label: str = "MoM MPC",
    solution=None,
    grid_points: bool = True,
    grid_type: str = "mpc",
) -> None:
    """Plot Figure 5: MoM MPC bounded by theory.

    Parameters
    ----------
    m_grid : np.ndarray
        Wealth grid for evaluation
    mpc_values : np.ndarray
        MPC values to plot (can be truth or MoM when implemented)
    mpc_opt_vals : np.ndarray
        Optimist MPC values (constant array)
    mpc_tight_vals : np.ndarray
        Tight bound MPC values (constant array)
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    mpc_label : str, optional
        Label for the main MPC line, by default "MoM MPC"
    solution : ConsumerSolution, optional
        Solution object to extract grid points from, by default None
    grid_points : bool, optional
        Whether to show grid points, by default True
    grid_type : str, optional
        Type of grid to extract ("mpc"), by default "mpc"

    """
    # Validate parameters
    if grid_points and solution is None:
        msg = "solution parameter is required when grid_points=True"
        raise ValueError(msg)

    fig, ax = setup_figure(title=title)

    # Plot MPC bounds with consistent colors
    ax.plot(
        m_grid,
        mpc_opt_vals,
        label="Optimist",
        color=CONCEPT_COLORS["optimist"],
        linewidth=LINE_WIDTH_MEDIUM,
        linestyle=LINE_STYLE_DASHED,
    )
    ax.plot(
        m_grid,
        mpc_tight_vals,
        label="Tighter Upper Bound",
        color=CONCEPT_COLORS["tight"],
        linewidth=LINE_WIDTH_MEDIUM,
        linestyle=LINE_STYLE_DASHDOT,
    )

    # Plot main MPC with appropriate color based on label
    main_color = get_concept_color(mpc_label)
    ax.plot(
        m_grid,
        mpc_values,
        label=mpc_label,
        color=main_color,
        linewidth=LINE_WIDTH_EXTRA_THICK,
    )

    # Extract and plot interpolation grid points if requested
    if grid_points and solution is not None:
        grid_points_m, grid_points_mpc = extract_mom_grid_points(solution, grid_type)
        if grid_points_m is not None and grid_points_mpc is not None:
            ax.scatter(
                grid_points_m,
                grid_points_mpc,
                label="Grid Points",
                color=main_color,
                s=MARKER_SIZE_STANDARD,
                zorder=5,
                edgecolors=MARKER_EDGE_COLOR,
                linewidths=MARKER_EDGE_WIDTH_THIN,
            )

    # Fill bound region
    ax.fill_between(
        m_grid,
        mpc_opt_vals,
        mpc_tight_vals,
        alpha=ALPHA_LOW,
        color=main_color,
        label="MPC bounds",
    )

    ax.set_xlabel("Normalized Market Resources (m)", fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel("Marginal Propensity to Consume (MPC)", fontsize=FONT_SIZE_LARGE)
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference line at x=0 only (not y=0 since MPC is always positive)
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    # Automatically set x-axis limits with padding
    x_range = m_grid.max() - m_grid.min()
    padding = PADDING_RATIO * x_range  # 5% padding on each side
    ax.set_xlim(m_grid.min() - padding, m_grid.max() + padding)

    # Set y-axis limits based on theoretical MPC bounds with padding
    mpc_min = min(mpc_opt_vals.min(), mpc_tight_vals.min())
    mpc_max = max(mpc_opt_vals.max(), mpc_tight_vals.max())
    y_range = mpc_max - mpc_min
    y_padding = PADDING_RATIO * y_range  # 5% padding on each side
    ax.set_ylim(mpc_min - y_padding, mpc_max + y_padding)

    plt.tight_layout()


def plot_value_functions(
    m_grid: np.ndarray,
    title: str,
    subtitle: str,
    v_truth: np.ndarray | None = None,
    v_opt: np.ndarray | None = None,
    v_pes: np.ndarray | None = None,
    v_tight: np.ndarray | None = None,
    v_egm_sparse: np.ndarray | None = None,
    v_mom_sparse: np.ndarray | None = None,
    figure_num: int = 7,
    mom_solution=None,
    egm_solution=None,
    grid_points: bool = True,
) -> None:
    """Plot value functions for optimist, pessimist, truth, tight bound, EGM, and MoM.

    Parameters
    ----------
    m_grid : np.ndarray
        Wealth grid for evaluation
    v_truth : np.ndarray | None, optional
        True (realist) value function values, by default None
    v_opt : np.ndarray | None, optional
        Optimist value function values, by default None
    v_pes : np.ndarray | None, optional
        Pessimist value function values, by default None
    v_tight : np.ndarray | None, optional
        Tight upper bound value function values, by default None
    v_egm_sparse : np.ndarray | None, optional
        EGM value function values (sparse grid), by default None
    v_mom_sparse : np.ndarray | None, optional
        MoM value function values (sparse grid), by default None
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    figure_num : int, optional
        Figure number, by default 7
    mom_solution : ConsumerSolution, optional
        MoM solution object to extract grid points from, by default None
    egm_solution : ConsumerSolution, optional
        EGM solution object to extract grid points from, by default None
    grid_points : bool, optional
        Whether to show grid points, by default True

    """
    # Validate parameters
    if grid_points and (mom_solution is None and egm_solution is None):
        msg = "At least one solution parameter is required when grid_points=True"
        raise ValueError(
            msg,
        )

    fig, ax = setup_figure(title=title)

    # Plot bounds first with consistent colors
    if v_opt is not None:
        ax.plot(
            m_grid,
            v_opt,
            label="Optimist",
            color=CONCEPT_COLORS["optimist"],
            linewidth=LINE_WIDTH_THICK,
            linestyle=LINE_STYLE_DASHED,
            alpha=ALPHA_HIGH,
        )

    if v_pes is not None:
        ax.plot(
            m_grid,
            v_pes,
            label="Pessimist",
            color=CONCEPT_COLORS["pessimist"],
            linewidth=LINE_WIDTH_THICK,
            linestyle=LINE_STYLE_DOTTED,
            alpha=ALPHA_HIGH,
        )

    # Plot tight bound if provided
    if v_tight is not None:
        ax.plot(
            m_grid,
            v_tight,
            label="Tighter Upper Bound",
            color=CONCEPT_COLORS["tight"],
            linewidth=LINE_WIDTH_THIN,
            linestyle=LINE_STYLE_DASHDOT,
            alpha=ALPHA_HIGH,
        )

    # Plot truth value function if provided
    if v_truth is not None:
        ax.plot(
            m_grid,
            v_truth,
            label="Truth",
            color=CONCEPT_COLORS["truth"],
            linewidth=LINE_WIDTH_EXTRA_THICK,
        )

    # Plot sparse EGM approximation if provided
    if v_egm_sparse is not None:
        ax.plot(
            m_grid,
            v_egm_sparse,
            label="EGM Approximation",
            color=CONCEPT_COLORS["egm"],
            linewidth=LINE_WIDTH_THICK,
            linestyle=LINE_STYLE_DASHED,  # Dashed line to distinguish as approximation
            alpha=ALPHA_HIGH,
        )

    # Plot sparse MoM approximation if provided
    if v_mom_sparse is not None:
        ax.plot(
            m_grid,
            v_mom_sparse,
            label="MoM Approximation",
            color=CONCEPT_COLORS["mom"],
            linewidth=LINE_WIDTH_THICK,
            linestyle=LINE_STYLE_DASHED,  # Dashed line to distinguish as approximation
            alpha=ALPHA_HIGH,
        )

    # Extract and plot grid points only for Figure 7 (not for Figure 8 inverse values)
    if grid_points and figure_num != 8:
        # Extract and plot MoM interpolation grid points if solution provided
        if mom_solution is not None:
            mom_grid_points_m, mom_grid_points_v = extract_mom_grid_points(
                mom_solution,
                "value",
            )
            if mom_grid_points_m is not None and mom_grid_points_v is not None:
                ax.scatter(
                    mom_grid_points_m,
                    mom_grid_points_v,
                    label="MoM Grid Points",
                    color=CONCEPT_COLORS["mom"],
                    s=MARKER_SIZE_STANDARD,
                    zorder=5,
                    edgecolors=MARKER_EDGE_COLOR,
                    linewidths=MARKER_EDGE_WIDTH_THIN,
                )

        # Extract and plot EGM interpolation grid points if solution provided
        if egm_solution is not None:
            egm_grid_points_m, egm_grid_points_v = extract_egm_grid_points(
                egm_solution,
                "value",
            )
            if egm_grid_points_m is not None and egm_grid_points_v is not None:
                ax.scatter(
                    egm_grid_points_m,
                    egm_grid_points_v,
                    label="EGM Grid Points",
                    color=CONCEPT_COLORS["egm"],
                    s=MARKER_SIZE_STANDARD,
                    zorder=5,
                    edgecolors=MARKER_EDGE_COLOR,
                    linewidths=MARKER_EDGE_WIDTH_THIN,
                )

    # Fill region to show bounds if both optimist and pessimist are provided
    if v_pes is not None and v_opt is not None:
        ax.fill_between(
            m_grid,
            v_pes,
            v_opt,
            alpha=ALPHA_MEDIUM_LOW,
            color=CONCEPT_COLORS["truth"],
            label="Feasible value region",
        )

    ax.set_xlabel("Normalized Market Resources (m)", fontsize=FONT_SIZE_LARGE)

    # Set labels and title based on figure type
    if figure_num == 8:
        # Figure 8: Inverse value functions
        ax.set_ylabel("Inverse Value Function (vNvrs)", fontsize=FONT_SIZE_LARGE)
    else:
        # Figure 7: Regular value functions
        ax.set_ylabel("Value Function (v)", fontsize=FONT_SIZE_LARGE)

    # Set title using the provided subtitle parameter
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")

    # Set specific axis limits based on figure type
    if figure_num == 8:
        # Figure 8: Inverse value functions - let matplotlib auto-scale
        pass  # Auto y-limits for inverse value functions (positive values)
    else:
        # Figure 7: Regular value functions - fixed limits for negative values
        ax.set_ylim(*YLIM_VALUE_FUNCTION)

    # Automatically set x-axis limits with padding
    x_range = m_grid.max() - m_grid.min()
    padding = PADDING_RATIO * x_range  # 5% padding on each side
    ax.set_xlim(m_grid.min() - padding, m_grid.max() + padding)

    ax.legend(loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference lines at x=0 and y=0
    ax.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    plt.tight_layout()


def plot_mom_value_debug(
    m_grid: np.ndarray,
    v_mom_sparse: np.ndarray,
    v_optimist: np.ndarray | None = None,
    v_pessimist: np.ndarray | None = None,
    grid_points_m: np.ndarray | None = None,
    grid_points_v: np.ndarray | None = None,
    figure_num: int = 7,
) -> None:
    """Debug plot for MoM value function with bounds to see actual scale and values.

    Parameters
    ----------
    m_grid : np.ndarray
        Wealth grid for evaluation
    v_mom_sparse : np.ndarray
        MoM value function values (sparse grid)
    v_optimist : np.ndarray | None, optional
        Optimist value function values, by default None
    v_pessimist : np.ndarray | None, optional
        Pessimist value function values, by default None
    grid_points_m : np.ndarray | None, optional
        Market resource values at interpolation grid points, by default None
    grid_points_v : np.ndarray | None, optional
        Value function values at interpolation grid points, by default None
    figure_num : int, optional
        Figure number, by default 7

    """
    fig, ax = setup_figure(
        title=f"Figure {figure_num}: MoM Value Function Debug with Bounds",
    )

    # Plot optimist bound if provided
    if v_optimist is not None:
        ax.plot(
            m_grid,
            v_optimist,
            label="Optimist",
            color=CONCEPT_COLORS["optimist"],
            linewidth=LINE_WIDTH_THIN,
            linestyle="-",
            alpha=ALPHA_HIGH,
        )

    # Plot pessimist bound if provided
    if v_pessimist is not None:
        ax.plot(
            m_grid,
            v_pessimist,
            label="Pessimist",
            color=CONCEPT_COLORS["pessimist"],
            linewidth=LINE_WIDTH_THIN,
            linestyle="-",
            alpha=ALPHA_HIGH,
        )

    # Plot interpolation grid points if provided
    if grid_points_m is not None and grid_points_v is not None:
        ax.scatter(
            grid_points_m,
            grid_points_v,
            label="Grid Points",
            color=CONCEPT_COLORS["mom"],
            s=MARKER_SIZE_LARGE,
            zorder=5,
            edgecolors=MARKER_EDGE_COLOR,
            linewidths=MARKER_EDGE_WIDTH_THICK,
        )

    # Plot MoM value function (main focus)
    ax.plot(
        m_grid,
        v_mom_sparse,
        label="MoM Approximation",
        color=CONCEPT_COLORS["mom"],
        linewidth=LINE_WIDTH_EXTRA_THICK,
        linestyle="-",
    )

    ax.set_xlabel("Normalized Market Resources (m)", fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel("Value Function (v)", fontsize=FONT_SIZE_LARGE)
    ax.set_title(
        "MoM Value Function: Full Scale View with Theoretical Bounds",
        fontsize=FONT_SIZE_XLARGE,
        fontweight="bold",
    )

    # Show actual value ranges
    v_min, v_max = v_mom_sparse.min(), v_mom_sparse.max()
    info_text = f"MoM range: [{v_min:.2e}, {v_max:.2e}]"

    if v_optimist is not None:
        opt_min, opt_max = v_optimist.min(), v_optimist.max()
        info_text += f"\nOpt range: [{opt_min:.2e}, {opt_max:.2e}]"

    if v_pessimist is not None:
        pes_min, pes_max = v_pessimist.min(), v_pessimist.max()
        info_text += f"\nPes range: [{pes_min:.2e}, {pes_max:.2e}]"

    if grid_points_m is not None:
        info_text += f"\nGrid points: {len(grid_points_m)}"

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_SMALL,
        verticalalignment="top",
        bbox={
            "boxstyle": f"round,pad={BBOX_PADDING}",
            "facecolor": "white",
            "alpha": ALPHA_VERY_HIGH,
        },
    )

    ax.legend(loc="best")
    ax.grid(True, alpha=GRID_ALPHA)

    # Add reference lines at x=0 and y=0
    ax.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )
    ax.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    # Don't set any axis limits - let matplotlib auto-scale
    plt.tight_layout()
