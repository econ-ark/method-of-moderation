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
- Logit Function (`plot_logit_function`): Shows chi(mu) transformation for
  numerical stability and asymptotic linearity
- Consumption Bounds: Shows consumption function bounded by theory
- Precautionary Gaps: Compares approximation quality vs truth
- MPC Bounds (`plot_mom_mpc`): Shows MPC bounded by theoretical limits
- Value Functions: Shows value function approximations vs truth
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from HARK.interpolation import LinearInterp
from moderation import calc_cusp_point, exp_mu, expit_moderate, log_mnrm_ex
from style import (
    ALPHA_HIGH,
    ALPHA_LOW,
    ALPHA_MEDIUM,
    ALPHA_MEDIUM_LOW,
    ALPHA_OPAQUE,
    FONT_SIZE_LARGE,
    GRID_ALPHA,
    LINE_STYLE_DASHDOT,
    LINE_STYLE_DASHED,
    LINE_STYLE_DOTTED,
    LINE_STYLE_SOLID,
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

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Public API exports
__all__ = [
    # Grid type enum
    "GridType",
    # Grid point extraction functions
    "extract_egm_grid_points",
    "extract_grid_points",
    "extract_mom_grid_points",
    # Main plotting functions
    "plot_consumption_bounds",
    "plot_cusp_point",
    "plot_logit_function",
    "plot_moderation_ratio",
    "plot_mom_mpc",
    "plot_precautionary_gaps",
    "plot_share_bounds",
    "plot_share_error",
    "plot_share_extrapolation",
    "plot_share_logit",
    "plot_share_sparse_accuracy",
    "plot_stochastic_bounds",
    "plot_value_functions",
]

# =========================================================================
# Plot-Specific Constants
# =========================================================================


class GridType(StrEnum):
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


# Y-axis limits for different plot types
YLIM_MODERATION_RATIO = (-0.1, 1.1)
# Top clears the near-constraint peak (measured 0.43003 at the lowest EGM
# gridpoint). At 0.35 matplotlib clipped the marker for a real gridpoint, so the
# figure showed 4 of the 5 the abstract promises.
YLIM_PRECAUTIONARY_GAPS = (-0.15, 0.46)
YLIM_VALUE_FUNCTION = (-6, 0)

# =========================================================================
# Helper Functions for Common Plotting Patterns
# =========================================================================


def _is_mom_solution(solution) -> bool:
    """Check if solution uses Method of Moderation.

    Parameters
    ----------
    solution : ConsumerSolution
        Solution object to check

    Returns
    -------
    bool
        True if solution uses TransformedFunctionMoM, False otherwise

    """
    return type(solution.cFunc).__name__ == "TransformedFunctionMoM"


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


def _normalize_series_labels(
    approx_solutions,
    legend: str | list[str] | None,
) -> tuple[list, list[str]]:
    """Coerce approximations and their labels into matched lists.

    A missing legend is inferred from each solution's own type, which is how
    both gap figures distinguish MoM from EGM without the caller restating it.
    """
    if not isinstance(approx_solutions, list):
        approx_solutions = [approx_solutions]
    if legend is None:
        legend = [
            "MoM Approximation" if _is_mom_solution(sol) else "EGM Approximation"
            for sol in approx_solutions
        ]
    elif not isinstance(legend, list):
        legend = [legend]
    return approx_solutions, legend


def _evaluate_value_series(truth_solution, m_grid, inverse, egm_solution, mom_solution):
    """Evaluate every value-function series on one grid, in draw order.

    The inverse transform has no tighter-upper-bound counterpart, so that series
    is None there. Optional solutions keep their original truthiness guard rather
    than an identity check, since a falsy solution should be skipped either way.
    """

    def evaluate(solution, *, optional=False):
        if optional and not solution:
            return None
        vfunc = solution.vFunc
        return vfunc.vFuncNvrs(m_grid) if inverse else vfunc(m_grid)

    return (
        evaluate(truth_solution),
        evaluate(truth_solution.Optimist),
        evaluate(truth_solution.Pessimist),
        None if inverse else evaluate(truth_solution.TighterUpperBound),
        evaluate(egm_solution, optional=True),
        evaluate(mom_solution, optional=True),
    )


def _plot_value_grid_points(ax: Axes, inverse: bool, mom_solution) -> None:
    """Overlay MoM value-function gridpoints, when there are any to show.

    Skipped for inverse value functions. Only MoM points are drawn, since EGM
    shares the same grid and a second identical scatter adds nothing.
    """
    if inverse or mom_solution is None:
        return
    points_m, points_v = extract_grid_points(mom_solution, GridType.VALUE)
    if points_m is None or points_v is None:
        return
    _plot_grid_points_scatter(
        ax,
        points_m,
        points_v,
        get_concept_color("MoM"),
        label="Grid Points",
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
        Accepted for backward compatibility but intentionally not rendered:
        the manuscript caption is the figure title, so an in-image title only
        duplicates the caption (and the journal's own figure numbering). This
        matches ``setup_figure``'s convention.
    legend_loc : str, optional
        Legend location, by default "upper right"

    """
    del subtitle  # not rendered; the LaTeX caption is the title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    try:
        if grid_type == GridType.CONSUMPTION:
            # MoM cFunc is directly a TransformedFunctionMoM; drop the synthetic
            # extrapolation knot appended at each end of the mu grid so markers
            # show only true solution gridpoints.
            mu_grid = solution.cFunc.logitModRteFunc.x_list[1:-1]
            m_min = solution.cFunc.mNrmMin
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_c = solution.cFunc(grid_points_m)
            return grid_points_m, grid_points_c

        if grid_type == GridType.VALUE:
            # MoM vFunc is ValueFuncCRRA containing TransformedFunctionMoM
            mu_grid = solution.vFunc.vFuncNvrs.logitModRteFunc.x_list[1:-1]
            m_min = solution.vFunc.vFuncNvrs.mNrmMin
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_v = solution.vFunc(grid_points_m)
            return grid_points_m, grid_points_v

        if grid_type == GridType.MPC:
            # For MPC, use consumption function grid points and evaluate derivative
            mu_grid = solution.cFunc.logitModRteFunc.x_list[1:-1]
            m_min = solution.cFunc.mNrmMin
            grid_points_m = exp_mu(mu_grid, m_min)
            grid_points_mpc = solution.cFunc.derivative(grid_points_m)
            return grid_points_m, grid_points_mpc

    except (AttributeError, KeyError, IndexError) as exc:
        # Grid extraction can fail for various solution types or incomplete
        # solutions. WARNING, not debug: these markers carry the argument of a
        # published figure, and WARNING reaches stderr with no logging config.
        logger.warning(
            "MoM grid extraction failed for %s solution: %s",
            grid_type.value,
            exc,
        )

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
            # For EGM MPC, use consumption grid and evaluate derivative
            grid_m = solution.cFunc.x_list.copy()
            grid_mpc = solution.cFunc.derivative(grid_m)
            return grid_m, grid_mpc

    except (AttributeError, KeyError, IndexError) as exc:
        # See extract_mom_grid_points for the rationale. HARK is pinned to a
        # moving git rev; surface API drift loudly.
        logger.warning(
            "EGM grid extraction failed for %s solution: %s",
            grid_type.value,
            exc,
        )

    return None, None


def extract_grid_points(
    solution,
    grid_type: GridType = GridType.CONSUMPTION,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract grid points from either MoM or EGM solution.

    Unified dispatcher that selects the appropriate extraction method
    based on the solution type.

    Parameters
    ----------
    solution : ConsumerSolution
        Solution object (MoM or EGM)
    grid_type : GridType, optional
        Type of grid to extract, by default GridType.CONSUMPTION

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (grid_points_m, grid_points_y) where y is consumption, value, or mpc.
        Returns (None, None) if extraction fails.

    """
    if _is_mom_solution(solution):
        return extract_mom_grid_points(solution, grid_type)
    return extract_egm_grid_points(solution, grid_type)


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

    _fig, ax = setup_figure(title=title)

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
        grid_points_m, _grid_points_y = extract_mom_grid_points(solution, grid_type)
        if grid_points_m is not None:
            # Evaluate the logit at the extracted points. Reading
            # `logitModRteFunc.y_list` instead crashed the scatter: extraction
            # drops synthetic knots, so the raw list is longer.
            mu_points = log_mnrm_ex(np.asarray(grid_points_m, dtype=float), m_min)
            grid_points_omega = expit_moderate(logitModRteFunc(mu_points))
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
    # Extract moderation functions from solution
    transformed_func = solution.cFunc
    m_min = transformed_func.mNrmMin
    logitModRteFunc = transformed_func.logitModRteFunc

    # Create evaluation grid in m space, convert to mu space
    m_grid = np.linspace(m_min + 0.001, m_max, n_points)
    mu_grid = log_mnrm_ex(m_grid, m_min)

    # Evaluate chi function
    chi_values = logitModRteFunc(mu_grid)

    _fig, ax = setup_figure(title=title)

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
        # Trim the synthetic extrapolation knots, matching extract_mom_grid_points:
        # untrimmed this draws 7 markers for a grid the paper calls five points.
        grid_points_x = solution.cFunc.logitModRteFunc.x_list[1:-1]
        grid_points_chi = solution.cFunc.logitModRteFunc.y_list[1:-1]
    elif grid_type == GridType.VALUE:
        # MoM vFunc contains TransformedFunctionMoM
        grid_points_x = solution.vFunc.vFuncNvrs.logitModRteFunc.x_list[1:-1]
        grid_points_chi = solution.vFunc.vFuncNvrs.logitModRteFunc.y_list[1:-1]
    else:
        grid_points_x = None
        grid_points_chi = None

    if grid_points_x is not None and grid_points_chi is not None:
        _plot_grid_points_scatter(ax, grid_points_x, grid_points_chi, mom_color)

    # Reference lines go in BEFORE the axes are configured: that helper calls
    # ax.legend(), which snapshots labelled artists, so a line added afterwards
    # never reaches the legend and reads as an unexplained crosshair.
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

    _configure_standard_axes(
        ax,
        xlabel="Log Excess Market Resources ($\\mu$)",
        ylabel="Logit Transformation $\\chi(\\mu)$",
        subtitle=subtitle,
        legend_loc="lower right",
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
    approx_solutions, legend = _normalize_series_labels(approx_solutions, legend)

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

    _fig, ax = setup_figure(title=title)

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
        strict=True,
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
            # Use unified grid extraction
            grid_points_m, grid_points_c = extract_grid_points(
                sol,
                GridType.CONSUMPTION,
            )
            # extract_grid_points already trims MoM's synthetic extrapolation
            # knots, so the last point is the true top gridpoint for both.
            grid_boundary = grid_points_m[-1] if grid_points_m is not None else None

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

        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            # Warning, not debug: the figure still plots without markers, but
            # a gap figure with no gridpoints marked argues nothing.
            logger.warning("Inline grid extraction failed: %s", exc)

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


def plot_solution_gaps(
    truth_solution,
    approx_solutions,
    title: str,
    subtitle: str,
    *,
    m_max: float = 30.0,
    n_points: int = 400,
    legend: str | list[str] | None = None,
) -> None:
    r"""Plot the absolute consumption error against a high-precision truth.

    This is the visual companion to the accuracy table: it plots
    :math:`|c_{\\text{truth}}(m) - c_{\\text{approx}}(m)|` on a logarithmic
    axis across the whole domain for each approximation. The error dips toward
    zero at the sparse gridpoints and peaks between them (the interval maxima
    the table reports), and beyond the top gridpoint (marked) the endogenous
    gridpoints error explodes while the method of moderation stays controlled.

    Parameters
    ----------
    truth_solution : ConsumerSolution
        High-precision "truth" solution for comparison.
    approx_solutions : ConsumerSolution or list[ConsumerSolution]
        Approximation solution(s) to compare, typically the sparse EGM and MoM
        solutions on a shared grid.
    title : str
        Figure title (kept for signature parity; not rendered in-image).
    subtitle : str
        Figure subtitle (kept for signature parity; not rendered in-image).
    m_max : float, optional
        Maximum market resources for the plot range, by default 30.0.
    n_points : int, optional
        Number of evaluation points, by default 400.
    legend : str or list[str], optional
        Legend labels for the approximation(s). If None, auto-generated from
        the solution type.

    """
    approx_solutions, legend = _normalize_series_labels(approx_solutions, legend)

    m_min = truth_solution.mNrmMin
    m_grid = np.linspace(m_min + 0.001, m_max, n_points)

    _fig, ax = setup_figure(title=title)

    # Distinct dash patterns per series. EGM and MoM both resolve to dashed via
    # get_concept_linestyle, and their green/pink separation falls 238 -> 95
    # under deuteranopia, so colour alone cannot carry this comparison.
    dash_cycle = (LINE_STYLE_DASHED, LINE_STYLE_DASHDOT, LINE_STYLE_DOTTED)

    boundary_labeled = False
    for series_i, (sol, method_label) in enumerate(
        zip(approx_solutions, legend, strict=True),
    ):
        abs_error = np.abs(truth_solution.cFunc(m_grid) - sol.cFunc(m_grid))
        color = get_concept_color(method_label)
        linestyle = dash_cycle[series_i % len(dash_cycle)]

        ax.plot(
            m_grid,
            abs_error,
            label=method_label,
            color=color,
            linewidth=LINE_WIDTH_THICK,
            linestyle=linestyle,
        )

        # Mark the top gridpoint: to its right the approximation extrapolates.
        try:
            grid_points_m, _ = extract_grid_points(sol, GridType.CONSUMPTION)
            if grid_points_m is not None and len(grid_points_m) > 1:
                # extract_grid_points already trims MoM's synthetic knots, so
                # the last point is the true top gridpoint for both methods.
                grid_boundary = grid_points_m[-1]
                ax.axvline(
                    x=grid_boundary,
                    color="gray",
                    linestyle=LINE_STYLE_DASHED,
                    alpha=ALPHA_MEDIUM,
                    label=None if boundary_labeled else "Top gridpoint",
                )
                boundary_labeled = True
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Inline grid extraction failed: %s", exc)

    ax.set_yscale("log")
    # Floor the axis just below the smallest interval maximum the table reports
    # (~1e-7); this trims the distracting near-machine-zero spikes at the nodes
    # without hiding any interval- or extrapolation-region error.
    ax.set_ylim(bottom=1e-9)
    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Absolute Consumption Error",
        subtitle=subtitle,
        legend_loc="lower right",
    )
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
        legend = "MoM Approximation" if _is_mom_solution(solution) else "Truth"

    # Create evaluation grid
    m_min = solution.mNrmMin
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)

    # Evaluate consumption functions
    c_main = solution.cFunc(m_grid)
    c_opt = solution.Optimist.cFunc(m_grid)
    c_pes = solution.Pessimist.cFunc(m_grid)
    c_tight = solution.TighterUpperBound.cFunc(m_grid) if show_tight_bound else None

    _fig, ax = setup_figure(title=title)

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
        grid_points_m, grid_points_c = extract_grid_points(
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
        mpc_label = "MoM MPC" if _is_mom_solution(solution) else "Truth MPC"

    # Create evaluation grid
    m_min = solution.mNrmMin
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)

    # Evaluate MPC
    mpc_values = solution.cFunc.derivative(m_grid)

    # Get constant bounds
    mpc_opt_vals = np.full_like(m_grid, solution.MPCmin)
    mpc_tight_vals = np.full_like(m_grid, solution.MPCmax)

    _fig, ax = setup_figure(title=title)

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
    grid_points_m, grid_points_mpc = extract_grid_points(solution, GridType.MPC)

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

    v_truth, v_opt, v_pes, v_tight, v_egm_sparse, v_mom_sparse = _evaluate_value_series(
        truth_solution,
        m_grid,
        inverse,
        egm_solution,
        mom_solution,
    )

    _fig, ax = setup_figure(title=title)

    # Series in draw order: bounds beneath, truth above, approximations last.
    for values, label, lw, ls, alpha in (
        (v_opt, "Optimist", LINE_WIDTH_THICK, LINE_STYLE_DASHED, ALPHA_HIGH),
        (v_pes, "Pessimist", LINE_WIDTH_THICK, LINE_STYLE_DOTTED, ALPHA_HIGH),
        (
            v_tight,
            "Tighter Upper Bound",
            LINE_WIDTH_THIN,
            LINE_STYLE_DASHDOT,
            ALPHA_HIGH,
        ),
        (v_truth, "Truth", LINE_WIDTH_EXTRA_THICK, LINE_STYLE_SOLID, ALPHA_OPAQUE),
        (v_egm_sparse, "EGM Approximation", LINE_WIDTH_THICK, None, ALPHA_HIGH),
        (v_mom_sparse, "MoM Approximation", LINE_WIDTH_THICK, None, ALPHA_HIGH),
    ):
        if values is None:
            continue
        ax.plot(
            m_grid,
            values,
            label=label,
            color=get_concept_color(label),
            linewidth=lw,
            linestyle=ls if ls is not None else get_concept_linestyle(label),
            alpha=alpha,
        )

    _plot_value_grid_points(ax, inverse, mom_solution)

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


def plot_cusp_point(
    solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 10.0,
    n_points: int = 200,
) -> None:
    r"""Plot consumption bounds showing the cusp point intersection.

    This figure visualizes where the optimist and tighter upper bounds
    intersect (the cusp point), as described in {eq}`eq:mNrmCusp`.
    Below the cusp, the tighter bound provides a better upper constraint;
    above the cusp, the optimist bound is tighter.

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
        Number of points in evaluation grid, by default 200

    """
    # Extract parameters for cusp calculation
    m_min = solution.mNrmMin
    hNrm = solution.hNrm
    MPCmin = solution.MPCmin
    MPCmax = solution.MPCmax

    # Calculate cusp point
    mNrmCusp = calc_cusp_point(hNrm, m_min, MPCmin, MPCmax)

    # Create evaluation grid
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)

    # Evaluate bounds
    c_opt = solution.Optimist.cFunc(m_grid)
    c_pes = solution.Pessimist.cFunc(m_grid)
    c_tight = solution.TighterUpperBound.cFunc(m_grid)

    # Consumption at cusp point
    c_cusp = solution.Optimist.cFunc(mNrmCusp)

    _fig, ax = setup_figure(title=title)

    # Plot pessimist (lower bound)
    ax.plot(
        m_grid,
        c_pes,
        label="Pessimist",
        color=get_concept_color("Pessimist"),
        linewidth=LINE_WIDTH_THICK,
        linestyle=LINE_STYLE_DOTTED,
        alpha=ALPHA_OPAQUE,
    )

    # Plot optimist (upper bound for high wealth)
    ax.plot(
        m_grid,
        c_opt,
        label="Optimist",
        color=get_concept_color("Optimist"),
        linewidth=LINE_WIDTH_THICK,
        linestyle=LINE_STYLE_DASHED,
        alpha=ALPHA_OPAQUE,
    )

    # Plot tighter upper bound (upper bound for low wealth)
    ax.plot(
        m_grid,
        c_tight,
        label="Tighter Upper Bound",
        color=get_concept_color("Tight"),
        linewidth=LINE_WIDTH_THICK,
        linestyle=LINE_STYLE_DASHDOT,
        alpha=ALPHA_OPAQUE,
    )

    # Plot the envelope of upper bounds (min of optimist and tight)
    c_envelope = np.minimum(c_opt, c_tight)
    ax.plot(
        m_grid,
        c_envelope,
        label="Upper Bound Envelope",
        color=get_concept_color("MoM"),
        linewidth=LINE_WIDTH_EXTRA_THICK,
        linestyle="-",
        alpha=ALPHA_HIGH,
    )

    # Mark the cusp point with a prominent marker
    ax.scatter(
        [mNrmCusp],
        [c_cusp],
        color=get_concept_color("MoM"),
        s=MARKER_SIZE_STANDARD * 1.5,
        zorder=10,
        edgecolors=MARKER_EDGE_COLOR,
        linewidths=MARKER_EDGE_WIDTH_THIN * 1.5,
        label=f"Cusp Point ($m$ = {mNrmCusp:.2f})",
    )

    # Add vertical line at cusp point
    ax.axvline(
        x=mNrmCusp,
        color=get_concept_color("MoM"),
        linestyle=LINE_STYLE_DASHED,
        linewidth=LINE_WIDTH_THIN,
        alpha=ALPHA_MEDIUM,
    )

    # Add annotation
    ax.annotate(
        f"$m^{{cusp}}$ = {mNrmCusp:.2f}",
        xy=(mNrmCusp, c_cusp),
        xytext=(mNrmCusp + 0.5, c_cusp + 0.3),
        fontsize=FONT_SIZE_LARGE,
        ha="left",
        arrowprops={
            "arrowstyle": "->",
            "color": get_concept_color("MoM"),
            "lw": 1.5,
        },
    )

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Consumption (c)",
        subtitle=subtitle,
        legend_loc="lower right",
    )
    _add_reference_lines(ax)
    _set_xlim_with_padding(ax, m_grid)

    # Set y-limits based on visible data
    y_min = min(c_pes.min(), 0)
    y_max = max(c_opt.max(), c_tight.max()) * 1.05
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()


def plot_stochastic_bounds(
    solution,
    title: str,
    subtitle: str,
    *,
    m_max: float = 10.0,
    n_points: int = 200,
) -> None:
    r"""Plot comparison of deterministic vs stochastic optimist bounds.

    This figure visualizes how stochastic returns affect the theoretical
    bounds on consumption. For CRRA > 1, the Merton-Samuelson stochastic
    MPC is *lower* than the deterministic MPC because a mean-preserving
    spread of returns raises E[R^(1-CRRA)] (Jensen's inequality, since
    R^(1-CRRA) is convex in R for CRRA > 1). With the same human wealth,
    a lower MPC at any given level of market resources implies lower
    consumption: the optimist facing return risk consumes slightly less
    than the deterministic optimist.

    Parameters
    ----------
    solution : ConsumerSolution
        Solution from stochastic returns solver containing both
        deterministic and stochastic bounds
    title : str
        Figure title
    subtitle : str
        Figure subtitle
    m_max : float, optional
        Maximum market resources for plot range, by default 10.0
    n_points : int, optional
        Number of points in evaluation grid, by default 200

    Notes
    -----
    This function expects a solution from RiskyAssetMoMConsumerType
    which includes OptimistStochastic and PessimistStochastic attributes.

    """
    m_min = solution.mNrmMin

    # Create evaluation grid
    m_grid = np.linspace(m_min + 0.01, m_max, n_points)

    # Evaluate deterministic bounds
    c_opt_det = solution.Optimist.cFunc(m_grid)
    c_pes_det = solution.Pessimist.cFunc(m_grid)

    # Evaluate stochastic bounds (if available). The values themselves carry
    # the availability flag, so the two cannot fall out of step.
    c_opt_stoch = c_pes_stoch = None
    if hasattr(solution, "OptimistStochastic"):
        c_opt_stoch = solution.OptimistStochastic.cFunc(m_grid)
        c_pes_stoch = solution.PessimistStochastic.cFunc(m_grid)

    # Main consumption function
    c_main = solution.cFunc(m_grid)

    _fig, ax = setup_figure(title=title)

    # Plot deterministic pessimist
    ax.plot(
        m_grid,
        c_pes_det,
        label="Pessimist (Deterministic)",
        color=get_concept_color("Pessimist"),
        linewidth=LINE_WIDTH_MEDIUM,
        linestyle=LINE_STYLE_DOTTED,
        alpha=ALPHA_HIGH,
    )

    # Plot deterministic optimist
    ax.plot(
        m_grid,
        c_opt_det,
        label=f"Optimist (Det., $\\kappa$ = {solution.MPCmin_deterministic:.3f})",
        color=get_concept_color("Optimist"),
        linewidth=LINE_WIDTH_MEDIUM,
        linestyle=LINE_STYLE_DASHED,
        alpha=ALPHA_HIGH,
    )

    if c_pes_stoch is not None:
        # Plot stochastic pessimist
        ax.plot(
            m_grid,
            c_pes_stoch,
            label="Pessimist (Stochastic)",
            color=get_concept_color("Pessimist"),
            linewidth=LINE_WIDTH_THICK,
            linestyle="-",
            alpha=ALPHA_OPAQUE,
        )

        # Plot stochastic optimist
        ax.plot(
            m_grid,
            c_opt_stoch,
            label=f"Optimist (Stoch., $\\kappa$ = {solution.MPCmin_stochastic:.3f})",
            color=get_concept_color("Optimist"),
            linewidth=LINE_WIDTH_THICK,
            linestyle="-",
            alpha=ALPHA_OPAQUE,
        )

        # Fill between deterministic and stochastic optimist to show difference
        ax.fill_between(
            m_grid,
            c_opt_stoch,
            c_opt_det,
            alpha=ALPHA_LOW,
            color=get_concept_color("Optimist"),
            label="Stochastic precautionary effect",
        )

    # Plot main consumption function
    main_linestyle = get_concept_linestyle("MoM Approximation")
    ax.plot(
        m_grid,
        c_main,
        label="Realist (MoM)",
        color=get_concept_color("MoM"),
        linewidth=LINE_WIDTH_EXTRA_THICK,
        linestyle=main_linestyle,
    )

    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m)",
        ylabel="Consumption (c)",
        subtitle=subtitle,
        legend_loc="lower right",
    )
    _add_reference_lines(ax)
    _set_xlim_with_padding(ax, m_grid)

    # Set y-limits based on visible data
    all_c = [c_pes_det, c_opt_det, c_main]
    if c_pes_stoch is not None:
        all_c.extend([c_opt_stoch, c_pes_stoch])
    y_min = min(c.min() for c in all_c)
    y_max = max(c.max() for c in all_c) * 1.05
    ax.set_ylim(y_min - 0.1, y_max)

    plt.tight_layout()


# =========================================================================
# Portfolio-Choice Risky Share
# =========================================================================


def plot_share_bounds(
    solution,
    title: str,
    subtitle: str,
    *,
    m_min: float = 1.0,
    m_max: float = 20000.0,
    n_points: int = 800,
) -> None:
    """Plot the risky share inside its bracket, with the cap release marked.

    The axis is log ``m`` from ``m_min``; starting at 1 rather than 0.1 keeps
    the flat constrained segment to a fifth of the width instead of a third.
    A symlog axis centered on the kink was tried and rejected (2026-08-25): the
    share is steepest just past the kink, so the edge of the linear zone
    renders as a second bend wherever it is placed.

    The bracket is the 'myopic' agent's share below and the leverage cap above.
    Quoted everywhere, in prose and in legends, on Alan's instruction: it is
    Campbell and Viceira's word for the mean-variance component rather than
    ours, unlike optimist and pessimist, and it implies a disability the fully
    rational investor it names does not have.
    Unlike the consumption bracket, whose bounds are functions of ``m``, both
    bounds here are constants, and the share approaches the LOWER one.

    Parameters
    ----------
    solution : PortfolioSolution
        A solved portfolio solution carrying ``ShareFuncAdj`` and ``ShareLimit``.
    title, subtitle : str
        Passed through; the title is discarded, since the caption is the title.
    m_max : float
        Right edge of the plotted range. Never exceed the solved range.
    n_points : int
        Evaluation points.

    """
    _fig, ax = setup_figure(title=title)
    m = np.geomspace(m_min, m_max, n_points)
    share = solution.ShareFuncAdj(m)
    sigma = solution.ShareLimit

    ax.plot(
        m,
        share,
        color=get_concept_color("Realist"),
        lw=LINE_WIDTH_THICK,
        label="Realist share",
    )
    ax.axhline(
        sigma,
        color=get_concept_color("Pessimist"),
        ls=LINE_STYLE_DASHED,
        lw=LINE_WIDTH_THIN,
        label=r"'Myopic' limit $\varsigma^{*}$",
    )
    ax.axhline(
        1.0,
        color=get_concept_color("Tight"),
        ls=LINE_STYLE_DASHED,
        lw=LINE_WIDTH_THIN,
        label="Leverage constraint",
    )
    ax.fill_between(
        m,
        sigma,
        1.0,
        color=get_concept_color("MoM"),
        alpha=ALPHA_LOW,
        label="Feasible share range",
    )

    release = solution.ShareFuncAdj.mRelease
    if np.isfinite(release):
        ax.axvline(
            release,
            color=get_concept_color("Tight"),
            ls=LINE_STYLE_DOTTED,
            lw=LINE_WIDTH_THIN,
        )
        ax.annotate(
            f"kink point at $m$ = {release:.2f}",
            xy=(release, 1.0),
            xytext=(release * 4.0, 0.88),
            fontsize=FONT_SIZE_LARGE,
            ha="left",
            arrowprops={"arrowstyle": "->", "color": get_concept_color("Tight")},
        )

    ax.set_xscale("log")
    ax.set_xlim(m_min, m_max)
    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m), log scale",
        ylabel=r"Risky Share $\varsigma$",
        subtitle=subtitle,
        legend_loc="lower left",
    )
    plt.tight_layout()


def plot_share_logit(
    solution,
    title: str,
    subtitle: str,
    *,
    m_min: float = 10.0,
    m_max: float = 100.0,
    n_points: int = 300,
) -> None:
    """Which transform straightens the share, and therefore which one to interpolate.

    Overlaying a fitted line on its own data shows nothing: the fit tracks the
    curve by construction. What decides the link is a COMPARISON. Each candidate
    is rescaled to a common vertical range and drawn against the straight chord
    through its own endpoints; whichever hugs its chord is the one linear
    interpolation can represent with few nodes.

    The share approaches its LOWER bound, so the linearizing transform is
    ``log(omega)``, not the logit that serves the consumption problem where the
    realist approaches the optimist from below.
    """
    _fig, ax = setup_figure(title=title)

    # Read the DIRECT solve, never the moderated representation. ShareFuncAdj is
    # piecewise linear in (log m, log omega) by construction, so using it here
    # would let the log link win because it drew the data.
    share_func = getattr(solution, "ShareFuncAdjDirect", solution.ShareFuncAdj)

    # Below the release omega is exactly 1 and log1p(-omega) is -inf, so the
    # logit's bow would become an artifact of the clip rather than a property
    # of the link. Refuse the window instead of quietly reporting that number.
    release = getattr(getattr(solution, "ShareFuncAdj", None), "mRelease", -np.inf)
    if np.isfinite(release) and m_min <= release:
        msg = (
            f"m_min={m_min} is at or below the kink point {release:.4g}; the "
            f"logit bow would be set by the omega clip, not by the link"
        )
        raise ValueError(msg)

    m = np.geomspace(m_min, m_max, n_points)
    share = share_func(m)
    omega = np.clip(
        (share - solution.ShareLimit) / (1.0 - solution.ShareLimit),
        1e-12,
        1 - 1e-12,
    )

    candidates = (
        ("Untransformed share", share, LINE_STYLE_DASHED, "Tight"),
        (
            r"logit $\omega_{\varsigma}$ (consumption link)",
            np.log(omega) - np.log1p(-omega),
            LINE_STYLE_DOTTED,
            "Linear",
        ),
        (
            r"$\log \omega_{\varsigma}$ (link used here)",
            np.log(omega),
            LINE_STYLE_SOLID,
            "MoM",
        ),
    )
    for i, (label, values, dash, concept) in enumerate(candidates):
        spread = values.max() - values.min()
        if spread <= 0.0:
            msg = f"candidate {label!r} is constant over [{m_min}, {m_max}]"
            raise ValueError(msg)
        scaled = (values - values.min()) / spread
        chord = np.linspace(scaled[0], scaled[-1], len(scaled))
        bow = 100.0 * np.abs(scaled - chord).max()
        ax.plot(
            m,
            scaled,
            color=get_concept_color(concept),
            ls=dash,
            lw=LINE_WIDTH_THICK if i == 2 else LINE_WIDTH_THIN,
            # Width and dash style already mark the focal series, so all three
            # run opaque: at ALPHA_MEDIUM the dotted comparator sits at 2.5:1
            # against the panel, against 3.9:1 opaque.
            alpha=ALPHA_OPAQUE,
            label=f"{label}: bows {bow:.1f}%",
        )
    ax.plot(
        m,
        np.linspace(1.0, 0.0, len(m)),  # all three descend; chord must too
        color=REFERENCE_LINE_COLOR,
        lw=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    ax.set_xscale("log")
    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m), log scale",
        ylabel="Rescaled to a common range",
        subtitle=subtitle,
        legend_loc="upper right",
    )
    plt.tight_layout()


def plot_share_sparse_accuracy(
    reference,
    moderated_factory,
    title: str,
    subtitle: str,
    *,
    n_nodes: int = 5,
    m_lo: float = 25.0,
    m_hi: float = 100.0,
    n_points: int = 400,
) -> None:
    """Compare moderated against linear interpolation on the same sparse nodes.

    Both schemes receive identical nodes and identical solved values, so the
    difference is attributable to the representation alone. The baseline is
    plain linear interpolation, matching the consumption sections.

    Parameters
    ----------
    reference : PortfolioSolution
        A converged dense solve supplying both the nodes' values and the truth.
    moderated_factory : callable
        ``(m_nodes, s_nodes) -> callable`` building the moderated share.

    """
    _fig, ax = setup_figure(title=title)
    probe = np.geomspace(m_lo, m_hi, n_points)
    truth = reference.ShareFuncAdj(probe)
    nodes = np.geomspace(m_lo, m_hi, n_nodes)
    s_nodes = reference.ShareFuncAdj(nodes)

    linear = LinearInterp(nodes, s_nodes, lower_extrap=True)(probe)
    moderated = moderated_factory(nodes, s_nodes)(probe)

    ax.plot(
        probe,
        truth,
        color=get_concept_color("Truth"),
        lw=LINE_WIDTH_THICK,
        label="Truth",
    )
    ax.plot(
        probe,
        linear,
        color=get_concept_color("Linear"),
        ls=LINE_STYLE_DASHED,
        lw=LINE_WIDTH_MEDIUM,
        label=f"Linear ({np.abs(linear - truth).max():.1e})",
    )
    ax.plot(
        probe,
        moderated,
        color=get_concept_color("MoM"),
        ls=LINE_STYLE_DASHDOT,
        lw=LINE_WIDTH_MEDIUM,
        label=f"MoM ({np.abs(moderated - truth).max():.1e})",
    )
    _plot_grid_points_scatter(ax, nodes, s_nodes, get_concept_color("MoM"))

    ax.set_xscale("log")
    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m), log scale",
        ylabel=r"Risky Share $\varsigma$",
        subtitle=subtitle,
        legend_loc="upper right",
    )
    plt.tight_layout()


def plot_share_error(
    reference,
    moderated_factory,
    title: str,
    subtitle: str,
    *,
    n_nodes: int = 5,
    m_lo: float = 25.0,
    m_hi: float = 100.0,
    n_points: int = 400,
) -> None:
    """Absolute share error for moderation against plain linear interpolation.

    The portfolio analogue of :func:`plot_solution_gaps`. Both schemes receive
    the same sparse nodes and the same solved values, so the separation is
    attributable to the representation alone. Plotted on a log axis because the
    gap spans two decades and is otherwise legible only from the legend.
    """
    _fig, ax = setup_figure(title=title)
    probe = np.geomspace(m_lo, m_hi, n_points)
    truth = reference.ShareFuncAdj(probe)
    nodes = np.geomspace(m_lo, m_hi, n_nodes)
    s_nodes = reference.ShareFuncAdj(nodes)

    series = (
        (
            "Linear",
            LinearInterp(nodes, s_nodes, lower_extrap=True)(probe),
            LINE_STYLE_DASHED,
        ),
        ("MoM", moderated_factory(nodes, s_nodes)(probe), LINE_STYLE_DASHDOT),
    )
    for label, values, dash in series:
        ax.plot(
            probe,
            np.abs(values - truth),
            label=label,
            color=get_concept_color(label),
            linewidth=LINE_WIDTH_THICK,
            linestyle=dash,
        )
    # No gridpoint scatter: their y-value here is an error, not a level, so any
    # marker would sit at a fabricated height. The dips locate them exactly, as
    # in the consumption error figure.

    ax.set_xscale("log")
    ax.set_yscale("log")
    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m), log scale",
        ylabel=r"$|\varsigma_{\rm approx} - \varsigma|$",
        subtitle=subtitle,
        legend_loc="lower left",
    )
    plt.tight_layout()


def plot_share_extrapolation(
    reference,
    sigma_star: float,
    moderated_factory,
    title: str,
    subtitle: str,
    *,
    m_lo: float = 25.0,
    m_top: float = 100.0,
    m_max: float = 400.0,
    n_points: int = 300,
) -> None:
    """Beyond the solved grid: what each scheme predicts for the risky share.

    The portfolio analogue of the consumption extrapolation pair. The plot starts
    inside the node range, where linear interpolation and moderation agree, so the
    divergence past the last node reads as a departure rather than as two
    unrelated curves. Linear extrapolation of a convex decreasing share overshoots
    downward and breaches the myopic limit; moderating in the log of the
    moderation ratio inherits the tail's measured slope instead of assuming one.
    """
    _fig, ax = setup_figure(title=title)
    probe = np.geomspace(m_lo, m_max, n_points)
    truth = reference.ShareFuncAdj(probe)

    nodes = np.geomspace(m_lo, m_top, 5)
    s_nodes = reference.ShareFuncAdj(nodes)
    series = (
        ("Truth", truth, LINE_STYLE_SOLID, LINE_WIDTH_EXTRA_THICK),
        (
            "Linear",
            LinearInterp(nodes, s_nodes, lower_extrap=True)(probe),
            LINE_STYLE_DASHED,
            LINE_WIDTH_THICK,
        ),
        (
            "MoM",
            moderated_factory(nodes, s_nodes)(probe),
            LINE_STYLE_DASHDOT,
            LINE_WIDTH_THICK,
        ),
    )
    for label, values, dash, width in series:
        ax.plot(
            probe,
            values,
            label=label,
            color=get_concept_color(label),
            linewidth=width,
            linestyle=dash,
        )
    ax.axhline(
        sigma_star,
        color=get_concept_color("Pessimist"),
        ls=LINE_STYLE_DASHED,
        lw=LINE_WIDTH_THIN,
        label=r"'Myopic' limit $\varsigma^{*}$",
    )
    ax.axvline(
        m_top,
        color=REFERENCE_LINE_COLOR,
        ls=LINE_STYLE_DOTTED,
        lw=REFERENCE_LINE_WIDTH,
        alpha=REFERENCE_LINE_ALPHA,
    )

    ax.set_xscale("log")
    _configure_standard_axes(
        ax,
        xlabel="Normalized Market Resources (m), log scale; grid ends at 100",
        ylabel=r"Risky Share $\varsigma$",
        subtitle=subtitle,
        legend_loc="lower left",
    )
    plt.tight_layout()
