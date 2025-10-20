"""Plotting functions for Method of Moderation notebook and dashboard.

This module implements all plot generation for the Method of Moderation project,
using style definitions and theming logic from style.py. It contains plotting
functions for the key figures that demonstrate the Method of Moderation's
performance characteristics.

Responsibilities
----------------
- Grid point extraction from MoM and EGM solutions
- Plot generation for moderation ratios, logit functions, consumption bounds
- Plot generation for precautionary gaps, MPCs, and value functions
- Visualization of theoretical bounds and approximation quality
- Grid point and interpolation boundary visualization

The module relies on style.py for all visual styling decisions including:
- Color assignments for economic concepts (via get_concept_color)
- Line style assignments (via get_concept_linestyle)
- Font sizes, line widths, alpha values, and other visual constants
- Grid type constants for consistent data extraction

Key Figures
-----------
- Moderation Ratio: Shows how realist moderates between optimist/pessimist bounds
- Logit Function: Shows chi transformation for numerical stability
- Consumption Bounds: Shows consumption function bounded by theory
- Precautionary Gaps: Compares approximation quality vs truth
- MPC Bounds: Shows MPC bounded by theoretical limits
- Value Functions: Shows value function approximations vs truth
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from moderation import expit_moderate

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Public API exports
__all__ = [
    # Grid type enum and configuration
    "GridType",
    "PlotConfig",
    "extract_egm_grid_points",
    # Grid point extraction functions
    "extract_mom_grid_points",
    "plot_consumption_bounds",
    "plot_logit_function",
    # Main plotting functions
    "plot_moderation_ratio",
    "plot_mom_mpc",
    "plot_precautionary_gaps",
    "plot_value_functions",
]

# =========================================================================
# Plot-Specific Constants
# =========================================================================


class GridType(str, Enum):
    """Grid types for data extraction from solutions.

    Attributes
    ----------
    CONSUMPTION : str
        Extract consumption function grid points
    VALUE : str
        Extract value function grid points
    MPC : str
        Extract marginal propensity to consume grid points

    """

    CONSUMPTION = "consumption"
    VALUE = "value"
    MPC = "mpc"


@dataclass
class PlotConfig:
    """Configuration for plot metadata and display options.

    This dataclass encapsulates common plotting parameters to reduce
    function signature clutter and make plots easier to configure.

    Attributes
    ----------
    title : str
        Main figure title (suptitle)
    subtitle : str
        Subplot title (axes title)
    xlabel : str
        X-axis label, by default "Normalized Market Resources (m)"
    ylabel : str
        Y-axis label (required, no default)
    legend_loc : str
        Legend location, by default "upper right"
    grid_points : bool
        Whether to display interpolation grid points, by default True
    grid_type : GridType
        Type of grid to extract for display, by default GridType.CONSUMPTION
    solution : object | None
        Solution object for grid point extraction, by default None

    Examples
    --------
    >>> config = PlotConfig(
    ...     title="Figure 1",
    ...     subtitle="Consumption Function",
    ...     ylabel="Consumption (c)",
    ... )

    """

    title: str
    subtitle: str
    ylabel: str
    xlabel: str = "Normalized Market Resources (m)"
    legend_loc: str = "upper right"
    grid_points: bool = True
    grid_type: GridType = GridType.CONSUMPTION
    solution: object | None = None


# Y-axis limits for different plot types
YLIM_MODERATION_RATIO = (-0.1, 1.1)
YLIM_PRECAUTIONARY_GAPS = (-0.15, 0.35)
YLIM_VALUE_FUNCTION = (-6, 0)

from style import (
    ALPHA_HIGH,
    ALPHA_LOW,
    ALPHA_MEDIUM,
    ALPHA_MEDIUM_LOW,
    ALPHA_OPAQUE,
    FONT_SIZE_LARGE,
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
    MARKER_EDGE_WIDTH_THIN,
    MARKER_SIZE_STANDARD,
    PADDING_RATIO,
    REFERENCE_LINE_ALPHA,
    REFERENCE_LINE_COLOR,
    REFERENCE_LINE_WIDTH,
    get_concept_color,
    get_concept_linestyle,
    setup_figure,
)

# =========================================================================
# Helper Functions for Common Plotting Patterns
# =========================================================================


def _validate_grid_points_params(grid_points: bool, solution: object | None) -> None:
    """Validate parameters for grid point plotting.

    Parameters
    ----------
    grid_points : bool
        Whether grid points should be shown
    solution : object | None
        Solution object to extract grid points from

    Raises
    ------
    ValueError
        If grid_points is True but solution is None

    """
    if grid_points and solution is None:
        msg = "solution parameter is required when grid_points=True"
        raise ValueError(msg)


def _add_reference_lines(
    ax: Axes,
    *,
    add_horizontal: bool = True,
    add_vertical: bool = True,
) -> None:
    """Add reference lines at x=0 and y=0.

    Parameters
    ----------
    ax : Axes
        Axes object to add reference lines to
    add_horizontal : bool, optional
        Whether to add horizontal reference line at y=0, by default True
    add_vertical : bool, optional
        Whether to add vertical reference line at x=0, by default True

    """
    if add_horizontal:
        ax.axhline(
            y=0,
            color=REFERENCE_LINE_COLOR,
            linewidth=REFERENCE_LINE_WIDTH,
            alpha=REFERENCE_LINE_ALPHA,
        )
    if add_vertical:
        ax.axvline(
            x=0,
            color=REFERENCE_LINE_COLOR,
            linewidth=REFERENCE_LINE_WIDTH,
            alpha=REFERENCE_LINE_ALPHA,
        )


def _set_xlim_with_padding(ax: Axes, m_grid: np.ndarray) -> None:
    """Set x-axis limits with automatic padding.

    Parameters
    ----------
    ax : Axes
        Axes object to set limits on
    m_grid : np.ndarray
        Market resources grid for evaluation

    """
    x_range = m_grid.max() - m_grid.min()
    padding = PADDING_RATIO * x_range
    ax.set_xlim(m_grid.min() - padding, m_grid.max() + padding)


def _plot_grid_points_scatter(
    ax: Axes,
    grid_points_m: np.ndarray,
    grid_points_y: np.ndarray,
    color: str,
    *,
    label: str = "Grid Points",
) -> None:
    """Plot grid points as scatter markers.

    Parameters
    ----------
    ax : Axes
        Axes object to plot on
    grid_points_m : np.ndarray
        X-coordinates of grid points
    grid_points_y : np.ndarray
        Y-coordinates of grid points
    color : str
        Color for the scatter markers
    label : str, optional
        Label for the legend, by default "Grid Points"

    """
    ax.scatter(
        grid_points_m,
        grid_points_y,
        label=label,
        color=color,
        s=MARKER_SIZE_STANDARD,
        zorder=5,
        edgecolors=MARKER_EDGE_COLOR,
        linewidths=MARKER_EDGE_WIDTH_THIN,
    )


def _configure_standard_axes(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    subtitle: str,
    *,
    legend_loc: str = "upper right",
) -> None:
    """Configure standard axis properties for plots.

    Parameters
    ----------
    ax : Axes
        Axes object to configure
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    subtitle : str
        Subplot title
    legend_loc : str, optional
        Legend location, by default "upper right"

    """
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_LARGE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LARGE)
    ax.set_title(subtitle, fontsize=FONT_SIZE_XLARGE, fontweight="bold")
    ax.legend(loc=legend_loc)
    ax.grid(True, alpha=GRID_ALPHA)


# =========================================================================
# Grid Point Extraction from Solutions
# =========================================================================


def extract_mom_grid_points(
    solution,
    grid_type: GridType = GridType.CONSUMPTION,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract interpolation grid points from MoM solution.

    Parameters
    ----------
    solution : ConsumerSolution
        MoM solution containing TransformedFunctionMoM
    grid_type : GridType, optional
        Type of grid to extract, by default GridType.CONSUMPTION

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (grid_points_m, grid_points_y) where y is consumption, value, or mpc.
        Returns (None, None) if extraction fails.

    """
    from moderation import exp_mu

    try:
        if grid_type == GridType.CONSUMPTION:
            # MoM cFunc is directly a TransformedFunctionMoM
            mu_grid = solution.cFunc.logitModRteFunc.x_list
            m_min = solution.cFunc.mNrmMin
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_c = solution.cFunc(grid_points_m)
            return grid_points_m, grid_points_c

        if grid_type == GridType.VALUE:
            # MoM vFunc is ValueFuncCRRA containing TransformedFunctionMoM
            mu_grid = solution.vFunc.vFuncNvrs.logitModRteFunc.x_list
            m_min = solution.vFunc.vFuncNvrs.mNrmMin
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_v = solution.vFunc(grid_points_m)
            return grid_points_m, grid_points_v

        if grid_type == GridType.MPC:
            # For MPC, use consumption function grid points and evaluate derivative
            mu_grid = solution.cFunc.logitModRteFunc.x_list
            m_min = solution.cFunc.mNrmMin
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_mpc = solution.cFunc.derivative(grid_points_m)
            return grid_points_m, grid_points_mpc

    except (AttributeError, KeyError, IndexError):
        # Catch specific exceptions instead of bare except
        pass

    return None, None


def extract_egm_grid_points(
    solution,
    grid_type: GridType = GridType.CONSUMPTION,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract interpolation grid points from EGM solution.

    Parameters
    ----------
    solution : ConsumerSolution
        EGM solution containing CubicInterp or LinearInterp functions
    grid_type : GridType, optional
        Type of grid to extract, by default GridType.CONSUMPTION

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (grid_points_m, grid_points_y) where y is consumption, value, or mpc.
        Returns (None, None) if extraction fails.

    """
    try:
        if grid_type == GridType.CONSUMPTION:
            # EGM cFunc is CubicInterp or LinearInterp - both have x_list, y_list
            return solution.cFunc.x_list.copy(), solution.cFunc.y_list.copy()

        if grid_type == GridType.VALUE:
            # EGM vFunc is ValueFuncCRRA with vFuncNvrs attribute
            grid_points_m = solution.vFunc.vFuncNvrs.x_list.copy()
            grid_points_v = solution.vFunc(grid_points_m)
            return grid_points_m, grid_points_v

        if grid_type == GridType.MPC:
            # For EGM MPC, use consumption grid and evaluate MPC function
            grid_m = solution.cFunc.x_list.copy()
            grid_mpc = solution.vPfunc(grid_m)
            return grid_m, grid_mpc

    except (AttributeError, KeyError, IndexError):
        # Catch specific exceptions instead of bare except
        pass

    return None, None


# =========================================================================
# Plotting Functions
# =========================================================================


def plot_moderation_ratio(
    solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 50.0,
    n_points: int = 200,
    grid_type: GridType = GridType.CONSUMPTION,
) -> None:
    r"""Plot moderation ratio $\\omega(m)$ showing how realist moderates between bounds.

    Parameters
    ----------
    solution : ConsumerSolution
        MoM solution containing moderation functions
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 50.0
    n_points : int, optional
        Number of points in evaluation grid, by default 200
    grid_type : GridType, optional
        Type of grid to extract, by default GridType.CONSUMPTION

    """
    from moderation import log_mnrm_ex

    # Extract moderation functions based on grid type
    if grid_type == GridType.VALUE:
        # Value function moderation
        transformed_func = solution.vFunc.vFuncNvrs
    else:
        # Consumption function moderation (default)
        transformed_func = solution.cFunc

    m_min = transformed_func.mNrmMin
    logitModRteFunc = transformed_func.logitModRteFunc

    # Create evaluation grid
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)
    mu_grid = log_mnrm_ex(m_grid, m_min)

    # Evaluate moderation ratio
    chi_values = logitModRteFunc(mu_grid)
    omega_values = expit_moderate(chi_values)

    fig, ax = setup_figure(title=title)

    # Plot moderation ratio
    mom_color = get_concept_color("MoM")
    ax.plot(
        m_grid,
        omega_values,
        label="Moderation Ratio $\\omega(m)$",
        color=mom_color,
        linewidth=LINE_WIDTH_THICK,
    )

    # Extract and plot interpolation grid points if solution provided
    if solution is not None:
        grid_points_m, grid_points_y = extract_mom_grid_points(solution, grid_type)
        if grid_points_m is not None:
            # For moderation ratio plots, we need to calculate omega from the grid points
            if grid_type == GridType.CONSUMPTION:
                # MoM cFunc is TransformedFunctionMoM - get chi values directly
                chi_values = solution.cFunc.logitModRteFunc.y_list
                # Convert chi to omega using expit_moderate
                grid_points_omega = expit_moderate(chi_values)
                _plot_grid_points_scatter(
                    ax,
                    grid_points_m,
                    grid_points_omega,
                    mom_color,
                )
            elif grid_type == GridType.VALUE:
                # MoM vFunc contains TransformedFunctionMoM
                chi_values = solution.vFunc.vFuncNvrs.logitModRteFunc.y_list
                grid_points_omega = expit_moderate(chi_values)
                _plot_grid_points_scatter(
                    ax,
                    grid_points_m,
                    grid_points_omega,
                    mom_color,
                )

    # Add reference lines with concept colors
    ax.axhline(
        y=0,
        color=get_concept_color("Pessimist"),
        linestyle=LINE_STYLE_DASHED,
        linewidth=LINE_WIDTH_THIN,
        alpha=ALPHA_HIGH,
        label="$\\omega = 0$ (Pessimist behavior, low wealth)",
    )
    ax.axhline(
        y=1,
        color=get_concept_color("Optimist"),
        linestyle="-",
        linewidth=LINE_WIDTH_THIN,
        alpha=ALPHA_HIGH,
        label="$\\omega = 1$ (Optimist behavior, high wealth)",
    )

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Moderation Ratio $\\omega(m)$",
        subtitle=subtitle,
    )
    _add_reference_lines(ax)
    ax.set_ylim(*YLIM_MODERATION_RATIO)
    _set_xlim_with_padding(ax, m_grid)

    plt.tight_layout()


def plot_logit_function(
    solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 50.0,
    n_points: int = 200,
    grid_type: GridType = GridType.CONSUMPTION,
) -> None:
    r"""Plot chi function $\\chi(\\mu)$ showing the logit-transformed moderation ratio.

    The chi function is always plotted in $\\mu$ space (log excess market resources)
    as this is its natural mathematical domain.

    Parameters
    ----------
    solution : ConsumerSolution
        MoM solution containing moderation functions
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 50.0
    n_points : int, optional
        Number of points in evaluation grid, by default 200
    grid_type : GridType, optional
        Type of grid to extract, by default GridType.CONSUMPTION

    """
    from moderation import log_mnrm_ex

    # Extract moderation functions from solution
    transformed_func = solution.cFunc
    m_min = transformed_func.mNrmMin
    logitModRteFunc = transformed_func.logitModRteFunc

    # Create evaluation grid in m space, convert to mu space
    m_grid = np.linspace(m_min + 0.001, m_max, n_points)
    mu_grid = log_mnrm_ex(m_grid, m_min)

    # Evaluate chi function
    chi_values = logitModRteFunc(mu_grid)

    fig, ax = setup_figure(title=title)

    # Plot chi function
    mom_color = get_concept_color("MoM")
    ax.plot(
        mu_grid,
        chi_values,
        label="Chi Function $\\chi(\\mu) = \\text{logit}(\\omega)$",
        color=mom_color,
        linewidth=LINE_WIDTH_THICK,
    )

    # Extract and plot interpolation grid points
    # Extract grid points directly from the logitModRteFunc (always in $\\mu$ space)
    if grid_type == GridType.CONSUMPTION:
        # MoM cFunc is TransformedFunctionMoM
        grid_points_x = solution.cFunc.logitModRteFunc.x_list  # $\\mu$ values
        grid_points_chi = solution.cFunc.logitModRteFunc.y_list  # $\\chi$ values
    elif grid_type == GridType.VALUE:
        # MoM vFunc contains TransformedFunctionMoM
        grid_points_x = solution.vFunc.vFuncNvrs.logitModRteFunc.x_list
        grid_points_chi = solution.vFunc.vFuncNvrs.logitModRteFunc.y_list
    else:
        grid_points_x = None
        grid_points_chi = None

    if grid_points_x is not None and grid_points_chi is not None:
        _plot_grid_points_scatter(ax, grid_points_x, grid_points_chi, mom_color)

    _configure_standard_axes(
        ax,
        xlabel="Log Excess Market Resources ($\\mu$)",
        ylabel="Logit Transformation $\\chi(\\mu)$",
        subtitle=subtitle,
        legend_loc="lower right",
    )

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


def plot_precautionary_gaps(
    truth_solution,
    approx_solutions,
    title: str,
    subtitle: str,
    *,
    m_max: float = 30.0,
    n_points: int = 100,
    legend: str | list[str] | None = None,
) -> None:
    """Plot precautionary saving gaps comparing truth vs approximation(s).

    Parameters
    ----------
    truth_solution : ConsumerSolution
        High-precision "truth" solution for comparison
    approx_solutions : ConsumerSolution or list[ConsumerSolution]
        Approximation solution(s) to compare (single or list for multiple methods)
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 30.0
    n_points : int, optional
        Number of points in evaluation grid, by default 100
    legend : str or list[str], optional
        Legend labels for approximation(s). If None, auto-generates from solution type.

    """
    # Ensure approx_solutions is a list
    if not isinstance(approx_solutions, list):
        approx_solutions = [approx_solutions]

    # Auto-generate legend labels if not provided
    if legend is None:
        legend = []
        for sol in approx_solutions:
            if type(sol.cFunc).__name__ == "TransformedFunctionMoM":
                legend.append("MoM Approximation")
            else:
                legend.append("EGM Approximation")
    elif not isinstance(legend, list):
        legend = [legend]

    # Create evaluation grid
    m_min = truth_solution.mNrmMin
    m_grid = np.linspace(m_min + 0.001, m_max, n_points)

    # Compute truth gap (Optimist - Truth)
    truth_gap = truth_solution.Optimist.cFunc(m_grid) - truth_solution.cFunc(m_grid)

    # Compute approximation gaps
    approx_gaps = []
    for sol in approx_solutions:
        # Use the same optimist as truth solution
        gap = truth_solution.Optimist.cFunc(m_grid) - sol.cFunc(m_grid)
        approx_gaps.append(gap)

    fig, ax = setup_figure(title=title)

    # Plot truth gap with consistent color
    ax.plot(
        m_grid,
        truth_gap,
        label="Truth",
        color=get_concept_color("Truth"),
        linewidth=LINE_WIDTH_THICK,
    )

    # Plot each approximation method
    for gap_vals, method_label, sol in zip(
        approx_gaps,
        legend,
        approx_solutions,
        strict=False,
    ):
        color = get_concept_color(method_label)
        linestyle = get_concept_linestyle(method_label)

        ax.plot(
            m_grid,
            gap_vals,
            label=method_label,
            color=color,
            linewidth=LINE_WIDTH_THICK,
            linestyle=linestyle,
        )

        # Extract and plot grid points for this solution
        try:
            # Determine solution type and extract appropriate grid points
            if type(sol.cFunc).__name__ == "TransformedFunctionMoM":
                # This is a MoM solution
                grid_points_m, grid_points_c = extract_mom_grid_points(
                    sol,
                    GridType.CONSUMPTION,
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
                    sol,
                    GridType.CONSUMPTION,
                )
                grid_boundary = (
                    grid_points_m[-1] if grid_points_m is not None else None
                )  # EGM: last point

            # Plot grid points if successfully extracted
            if grid_points_m is not None and grid_points_c is not None:
                # Get the gap values at grid point locations by interpolation
                gap_at_grid_points = np.interp(grid_points_m, m_grid, gap_vals)

                # Plot actual grid points as scatter
                _plot_grid_points_scatter(ax, grid_points_m, gap_at_grid_points, color)

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

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Precautionary Saving Gap",
        subtitle=subtitle,
    )
    _add_reference_lines(ax)
    ax.set_ylim(*YLIM_PRECAUTIONARY_GAPS)
    _set_xlim_with_padding(ax, m_grid)

    plt.tight_layout()


def plot_consumption_bounds(
    solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 10.0,
    n_points: int = 100,
    show_tight_bound: bool = False,
    show_grid_points: bool = True,
    legend: str | None = None,
) -> None:
    """Plot consumption function with theoretical bounds.

    Unified function for both Figure 2 (Truth bounds) and Figure 4 (MoM bounds).

    Parameters
    ----------
    solution : ConsumerSolution
        Solution containing consumption function and theoretical bounds
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 10.0
    n_points : int, optional
        Number of points in evaluation grid, by default 100
    show_tight_bound : bool, optional
        Whether to show tighter upper bound, by default False
    show_grid_points : bool, optional
        Whether to show approximation grid points, by default True
    legend : str, optional
        Legend label for the main consumption function. If None, auto-generates.

    """
    # Auto-generate legend if not provided
    if legend is None:
        if type(solution.cFunc).__name__ == "TransformedFunctionMoM":
            legend = "MoM Approximation"
        else:
            legend = "Truth"

    # Create evaluation grid
    m_min = solution.mNrmMin
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)

    # Evaluate consumption functions
    c_main = solution.cFunc(m_grid)
    c_opt = solution.Optimist.cFunc(m_grid)
    c_pes = solution.Pessimist.cFunc(m_grid)
    c_tight = solution.TighterUpperBound.cFunc(m_grid) if show_tight_bound else None

    fig, ax = setup_figure(title=title)

    # Plot bounds first with consistent colors
    ax.plot(
        m_grid,
        c_opt,
        label="Optimist",
        color=get_concept_color("Optimist"),
        linewidth=LINE_WIDTH_THICK,
        linestyle=LINE_STYLE_DASHED,
        alpha=ALPHA_OPAQUE,
    )
    ax.plot(
        m_grid,
        c_pes,
        label="Pessimist",
        color=get_concept_color("Pessimist"),
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
            color=get_concept_color("Tight"),
            linewidth=LINE_WIDTH_THIN,
            linestyle=LINE_STYLE_DASHDOT,
            alpha=ALPHA_HIGH,
        )

    # Plot main consumption function with appropriate color and line style based on label
    main_color = get_concept_color(legend)
    main_linestyle = get_concept_linestyle(legend)
    ax.plot(
        m_grid,
        c_main,
        label=legend,
        color=main_color,
        linewidth=LINE_WIDTH_EXTRA_THICK,
        linestyle=main_linestyle,
    )

    # Extract and plot interpolation grid points if requested
    if show_grid_points:
        # Determine which extractor to use based on solution type
        if type(solution.cFunc).__name__ == "TransformedFunctionMoM":
            grid_points_m, grid_points_c = extract_mom_grid_points(
                solution,
                GridType.CONSUMPTION,
            )
        else:
            grid_points_m, grid_points_c = extract_egm_grid_points(
                solution,
                GridType.CONSUMPTION,
            )

        if grid_points_m is not None and grid_points_c is not None:
            _plot_grid_points_scatter(ax, grid_points_m, grid_points_c, main_color)

    # Fill regions to show bounds
    ax.fill_between(
        m_grid,
        c_pes,
        c_opt,
        alpha=ALPHA_MEDIUM_LOW,
        color=main_color,
        label="Feasible region",
    )

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Consumption (c)",
        subtitle=subtitle,
        legend_loc="lower right",
    )
    _add_reference_lines(ax)

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
    solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 10.0,
    n_points: int = 100,
    mpc_label: str | None = None,
) -> None:
    """Plot MPC bounded by theory.

    Parameters
    ----------
    solution : ConsumerSolution
        Solution containing MPC and theoretical bounds
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 10.0
    n_points : int, optional
        Number of points in evaluation grid, by default 100
    mpc_label : str, optional
        Label for the main MPC line. If None, auto-generates.

    """
    # Auto-generate label if not provided
    if mpc_label is None:
        if type(solution.cFunc).__name__ == "TransformedFunctionMoM":
            mpc_label = "MoM MPC"
        else:
            mpc_label = "Truth MPC"

    # Create evaluation grid
    m_min = solution.mNrmMin
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)

    # Evaluate MPC
    mpc_values = solution.cFunc.derivative(m_grid)

    # Get constant bounds
    mpc_opt_vals = np.full_like(m_grid, solution.MPCmin)
    mpc_tight_vals = np.full_like(m_grid, solution.MPCmax)

    fig, ax = setup_figure(title=title)

    # Plot MPC bounds with consistent colors
    ax.plot(
        m_grid,
        mpc_opt_vals,
        label="Optimist",
        color=get_concept_color("Optimist"),
        linewidth=LINE_WIDTH_MEDIUM,
        linestyle=LINE_STYLE_DASHED,
    )
    ax.plot(
        m_grid,
        mpc_tight_vals,
        label="Tighter Upper Bound",
        color=get_concept_color("Tight"),
        linewidth=LINE_WIDTH_MEDIUM,
        linestyle=LINE_STYLE_DASHDOT,
    )

    # Plot main MPC with appropriate color and line style based on label
    main_color = get_concept_color(mpc_label)
    main_linestyle = get_concept_linestyle(mpc_label)
    ax.plot(
        m_grid,
        mpc_values,
        label=mpc_label,
        color=main_color,
        linewidth=LINE_WIDTH_EXTRA_THICK,
        linestyle=main_linestyle,
    )

    # Extract and plot interpolation grid points
    if type(solution.cFunc).__name__ == "TransformedFunctionMoM":
        grid_points_m, grid_points_mpc = extract_mom_grid_points(solution, GridType.MPC)
    else:
        grid_points_m, grid_points_mpc = extract_egm_grid_points(solution, GridType.MPC)

    if grid_points_m is not None and grid_points_mpc is not None:
        _plot_grid_points_scatter(ax, grid_points_m, grid_points_mpc, main_color)

    # Fill bound region
    ax.fill_between(
        m_grid,
        mpc_opt_vals,
        mpc_tight_vals,
        alpha=ALPHA_LOW,
        color=main_color,
        label="MPC bounds",
    )

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Marginal Propensity to Consume (MPC)",
        subtitle=subtitle,
    )
    _add_reference_lines(
        ax,
        add_horizontal=False,
        add_vertical=True,
    )  # MPC always positive
    _set_xlim_with_padding(ax, m_grid)

    # Set y-axis limits based on theoretical MPC bounds with padding
    mpc_min = min(mpc_opt_vals.min(), mpc_tight_vals.min())
    mpc_max = max(mpc_opt_vals.max(), mpc_tight_vals.max())
    y_range = mpc_max - mpc_min
    y_padding = PADDING_RATIO * y_range  # 5% padding on each side
    ax.set_ylim(mpc_min - y_padding, mpc_max + y_padding)

    plt.tight_layout()


def plot_value_functions(
    truth_solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 3.0,
    n_points: int = 100,
    inverse: bool = False,
    egm_solution=None,
    mom_solution=None,
) -> None:
    """Plot value functions with theoretical bounds and approximations.

    Parameters
    ----------
    truth_solution : ConsumerSolution
        High-precision "truth" solution for comparison
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 3.0
    n_points : int, optional
        Number of points in evaluation grid, by default 100
    inverse : bool, optional
        If True, plot inverse value functions (vNvrs), by default False
    egm_solution : ConsumerSolution, optional
        EGM approximation solution to plot, by default None
    mom_solution : ConsumerSolution, optional
        MoM approximation solution to plot, by default None

    """
    # Create evaluation grid
    m_min = truth_solution.mNrmMin
    m_grid = np.linspace(m_min + 0.001, m_max, n_points)

    # Evaluate value functions based on mode
    if inverse:
        # Inverse value functions
        v_truth = truth_solution.vFunc.vFuncNvrs(m_grid)
        v_opt = truth_solution.Optimist.vFunc.vFuncNvrs(m_grid)
        v_pes = truth_solution.Pessimist.vFunc.vFuncNvrs(m_grid)
        v_tight = None
        v_egm_sparse = egm_solution.vFunc.vFuncNvrs(m_grid) if egm_solution else None
        v_mom_sparse = mom_solution.vFunc.vFuncNvrs(m_grid) if mom_solution else None
    else:
        # Regular value functions
        v_truth = truth_solution.vFunc(m_grid)
        v_opt = truth_solution.Optimist.vFunc(m_grid)
        v_pes = truth_solution.Pessimist.vFunc(m_grid)
        v_tight = truth_solution.TighterUpperBound.vFunc(m_grid)
        v_egm_sparse = egm_solution.vFunc(m_grid) if egm_solution else None
        v_mom_sparse = mom_solution.vFunc(m_grid) if mom_solution else None

    fig, ax = setup_figure(title=title)

    # Plot bounds first with consistent colors
    if v_opt is not None:
        ax.plot(
            m_grid,
            v_opt,
            label="Optimist",
            color=get_concept_color("Optimist"),
            linewidth=LINE_WIDTH_THICK,
            linestyle=LINE_STYLE_DASHED,
            alpha=ALPHA_HIGH,
        )

    if v_pes is not None:
        ax.plot(
            m_grid,
            v_pes,
            label="Pessimist",
            color=get_concept_color("Pessimist"),
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
            color=get_concept_color("Tight"),
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
            color=get_concept_color("Truth"),
            linewidth=LINE_WIDTH_EXTRA_THICK,
        )

    # Plot sparse EGM approximation if provided
    if v_egm_sparse is not None:
        egm_linestyle = get_concept_linestyle("EGM Approximation")
        ax.plot(
            m_grid,
            v_egm_sparse,
            label="EGM Approximation",
            color=get_concept_color("EGM"),
            linewidth=LINE_WIDTH_THICK,
            linestyle=egm_linestyle,
            alpha=ALPHA_HIGH,
        )

    # Plot sparse MoM approximation if provided
    if v_mom_sparse is not None:
        mom_linestyle = get_concept_linestyle("MoM Approximation")
        ax.plot(
            m_grid,
            v_mom_sparse,
            label="MoM Approximation",
            color=get_concept_color("MoM"),
            linewidth=LINE_WIDTH_THICK,
            linestyle=mom_linestyle,
            alpha=ALPHA_HIGH,
        )

    # Extract and plot grid points (only for regular value functions, not inverse)
    # Only show MoM grid points as they are the same as EGM grid points
    if not inverse and mom_solution is not None:
        mom_grid_points_m, mom_grid_points_v = extract_mom_grid_points(
            mom_solution,
            GridType.VALUE,
        )
        if mom_grid_points_m is not None and mom_grid_points_v is not None:
            _plot_grid_points_scatter(
                ax,
                mom_grid_points_m,
                mom_grid_points_v,
                get_concept_color("MoM"),
                label="Grid Points",
            )

    # Fill region to show bounds if both optimist and pessimist are provided
    if v_pes is not None and v_opt is not None:
        ax.fill_between(
            m_grid,
            v_pes,
            v_opt,
            alpha=ALPHA_MEDIUM_LOW,
            color=get_concept_color("Truth"),
            label="Feasible value region",
        )

    # Set labels and title based on function type
    ylabel = "Inverse Value Function (vNvrs)" if inverse else "Value Function (v)"

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel=ylabel,
        subtitle=subtitle,
        legend_loc="lower right",
    )

    # Set specific axis limits based on function type
    if not inverse:
        # Regular value functions - fixed limits for negative values
        ax.set_ylim(*YLIM_VALUE_FUNCTION)
    # Inverse value functions auto-scale (no fixed limits)

    _set_xlim_with_padding(ax, m_grid)

    _add_reference_lines(ax)

    plt.tight_layout()
