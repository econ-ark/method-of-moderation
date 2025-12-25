"""Professional Econ-ARK branding and style definitions for Python applications.

This module defines all visual constants, colors, fonts, matplotlib configuration,
and theming logic used throughout the Method of Moderation project. It centralizes
all style-related decisions to ensure consistency across plots and notebooks.

Responsibilities
----------------
- Official ECON-ARK brand color definitions
- Consistent color mapping for economic concepts (truth, MoM, EGM, optimist, etc.)
- Matplotlib style configuration and theme application
- Typography and font settings
- Line styles, widths, alphas, and other visual constants
- Grid type constants for data extraction
- Helper functions for concept-based color and line style selection
- External CSS loading for Jupyter notebook styling

Built on normalize.css and HTML5 Boilerplate foundations for cross-browser
consistency. Contains official ECON-ARK color schemes, plot styles, and modern
HTML/CSS styling with responsive design and accessibility features. All CSS
styling is loaded from an external file for better maintainability and easier
customization.

Features
--------
- Ultra-minimal CSS with STRICT ECON-ARK brand compliance
- ONLY approved ECON-ARK brand colors (6 colors, no unauthorized additions)
- ONLY approved fonts: Roboto and Varela Round (no system font fallbacks)
- Professional typography hierarchy with full-width ARK blue underlines
- Clean text formatting and alignment utilities
- Extremely lightweight and brand-compliant (~3KB CSS file)
- Refined professional styling colors for enhanced visual polish
- Matplotlib plots with professional light panel background and refined colors

External Files
--------------
style.css : Ultra-minimal stylesheet (~3KB, 124 lines) including:
  * Minimal CSS reset (box-sizing, margin/padding reset)
  * STRICT ECON-ARK brand compliance (6 colors + 2 fonts ONLY)
  * Professional typography system using Varela Round headings and Roboto body
  * h1: Bold ARK blue titles, h2: ARK blue with full-width underlines
  * h3: ARK light blue subsections, h4-h6: ARK blue graduated sizes
  * Optional .ark-h2.lightblue variant for figure headings
  * Brand-colored strong/bold text and utility classes
  * Mobile typography scaling maintaining brand font compliance

The CSS file is automatically loaded when this module is imported, with
graceful fallback behavior if the file is not found.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler

# Public API exports
__all__ = [
    # Alpha values
    "ALPHA_HIGH",
    "ALPHA_LOW",
    "ALPHA_MEDIUM",
    "ALPHA_MEDIUM_LOW",
    "ALPHA_OPAQUE",
    # Official ECON-ARK brand colors
    "ARK_BLUE",
    "ARK_GREEN",
    "ARK_GREY",
    "ARK_LIGHTBLUE",
    "ARK_PINK",
    "ARK_YELLOW",
    # Refined styling colors
    "ARK_GRID_SOFT",
    "ARK_PANEL_LIGHT",
    "ARK_SPINE",
    "ARK_TEXT",
    # Concept colors mapping
    "CONCEPT_COLORS",
    # Font sizes
    "FONT_SIZE_LARGE",
    "FONT_SIZE_XLARGE",
    # Grid and padding
    "GRID_ALPHA",
    "PADDING_RATIO",
    # Line styles
    "LINE_STYLE_DASHDOT",
    "LINE_STYLE_DASHED",
    "LINE_STYLE_DOTTED",
    "LINE_STYLE_SOLID",
    # Line widths
    "LINE_WIDTH_EXTRA_THICK",
    "LINE_WIDTH_MEDIUM",
    "LINE_WIDTH_THICK",
    "LINE_WIDTH_THIN",
    # Marker styling
    "MARKER_EDGE_COLOR",
    "MARKER_EDGE_WIDTH_THIN",
    "MARKER_SIZE_STANDARD",
    # Reference line styling
    "REFERENCE_LINE_ALPHA",
    "REFERENCE_LINE_COLOR",
    "REFERENCE_LINE_WIDTH",
    # Matplotlib configuration
    "MATPLOTLIB_STYLE",
    # Notebook styling
    "HEADER_HTML_NOTEBOOK",
    "NOTEBOOK_CSS",
    # Theming functions
    "apply_ark_style",
    "apply_notebook_css",
    "get_concept_color",
    "get_concept_linestyle",
    "setup_figure",
]

# Official ECON-ARK brand colors - ONLY APPROVED DEFINITIONS
ARK_BLUE = "#1f476b"
ARK_LIGHTBLUE = "#00aeef"
ARK_PINK = "#ed217c"
ARK_GREEN = "#39b54a"
ARK_YELLOW = "#fcb040"
ARK_GREY = "#676470"

# Refined ECON-ARK styling colors (separate from official brand colors)
ARK_PANEL_LIGHT = "#f8fafc"  # Lighter panel background
ARK_GRID_SOFT = "#edf2f7"  # Softer grid lines
ARK_SPINE = "#94a3b8"  # Professional spine color
ARK_TEXT = "#334155"  # Clear, professional text color

# Reference line styling for x=0 and y=0 axes
REFERENCE_LINE_COLOR = ARK_GREY
REFERENCE_LINE_WIDTH = 0.8
REFERENCE_LINE_ALPHA = 0.7

# Grid styling constants
GRID_ALPHA = 0.3
PADDING_RATIO = 0.05  # 5% padding on each side of plots

# CONSISTENT COLOR MAPPING FOR ECONOMIC CONCEPTS
CONCEPT_COLORS = {
    "truth": ARK_BLUE,
    "mom": ARK_GREEN,
    "egm": ARK_PINK,
    "optimist": ARK_LIGHTBLUE,
    "pessimist": ARK_YELLOW,
    "tight": ARK_GREY,
}

# Plot styling constants
# Font sizes
FONT_SIZE_LARGE = 14
FONT_SIZE_XLARGE = 16

# Line widths
LINE_WIDTH_THIN = 2.0
LINE_WIDTH_MEDIUM = 2.5
LINE_WIDTH_THICK = 3.0
LINE_WIDTH_EXTRA_THICK = 4.0

# Alpha values for transparency
ALPHA_LOW = 0.1
ALPHA_MEDIUM_LOW = 0.2
ALPHA_MEDIUM = 0.7
ALPHA_HIGH = 0.8
ALPHA_OPAQUE = 1.0

# Marker styling (unify edge color with global style)
MARKER_SIZE_STANDARD = 100
MARKER_EDGE_WIDTH_THIN = 1.5
MARKER_EDGE_COLOR = ARK_GREY

# Line styles
LINE_STYLE_SOLID = "-"
LINE_STYLE_DASHED = "--"
LINE_STYLE_DASHDOT = "-."
LINE_STYLE_DOTTED = ":"

# =========================================================================
# Concept-Based Theming Functions
# =========================================================================


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
    # Fallback to cycling through colors for unknown methods
    # Use sum of character ordinals for deterministic indexing (hash() varies across sessions)
    fallback_colors = [
        ARK_BLUE,
        ARK_GREEN,
        ARK_PINK,
        ARK_LIGHTBLUE,
        ARK_YELLOW,
        ARK_GREY,
    ]
    return fallback_colors[sum(ord(c) for c in name_lower) % len(fallback_colors)]


def get_concept_linestyle(method_name: str) -> str:
    """Get appropriate line style for economic concept/method.

    Both EGM and MoM approximations use dashed lines to distinguish from truth.
    Truth and theoretical bounds use solid lines.

    Parameters
    ----------
    method_name : str
        Name of the method/concept (case-insensitive)

    Returns
    -------
    str
        Line style string ("-", "--", "-.", ":")

    """
    name_lower = method_name.lower()

    # Both EGM and MoM approximations always use dashed lines to distinguish from truth
    if (
        "egm" in name_lower
        or "endogenous" in name_lower
        or "mom" in name_lower
        or "moderation" in name_lower
        or "approximation" in name_lower
    ):
        return "--"  # Dashed line for all approximations
    return "-"  # Default solid line for truth and bounds


# =========================================================================
# Matplotlib Style Configuration
# =========================================================================

# Matplotlib style configuration
MATPLOTLIB_STYLE = {
    # --- Font & text ---
    "font.family": ["sans-serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "600",  # Bolder titles
    "axes.labelsize": 9,  # Smaller axis labels
    "axes.labelweight": "500",  # Slightly bolder labels
    "xtick.labelsize": 8.5,  # Slightly smaller tick labels
    "ytick.labelsize": 8.5,
    # Text colors - using refined professional colors
    "text.color": ARK_TEXT,
    "axes.labelcolor": ARK_TEXT,
    "axes.titlecolor": ARK_BLUE,  # Brand color for all titles including subplots
    "xtick.color": ARK_TEXT,
    "ytick.color": ARK_TEXT,
    # --- Colours & lines ---
    "axes.prop_cycle": cycler(
        color=[
            ARK_BLUE,
            ARK_LIGHTBLUE,
            ARK_GREEN,
            ARK_PINK,
            ARK_YELLOW,
            ARK_GREY,
        ],
    ),
    "axes.edgecolor": ARK_SPINE,
    "axes.linewidth": 1.2,  # Slightly thicker spines
    "grid.color": ARK_GRID_SOFT,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,  # Higher alpha for softer grid color
    # --- Background & figure ---
    # Professional light panel background for subtle visual refinement
    "axes.facecolor": ARK_PANEL_LIGHT,  # Clean, light background
    "figure.facecolor": "white",
    "figure.dpi": 110,
    # --- Spines ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    # --- Legend ---
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": ARK_GREY,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    # --- Lines & markers ---
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "lines.markeredgewidth": 1.5,
    "lines.markeredgecolor": MARKER_EDGE_COLOR,
    # --- Ticks ---
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
}

# =========================================================================
# External CSS File Loading
# =========================================================================


def _load_css_file(filename: str) -> str:
    """Load CSS content from an external file.

    Parameters
    ----------
    filename : str
        Name of the CSS file to load (relative to this module's directory)

    Returns
    -------
    str
        CSS content wrapped in <style> tags, or empty string if file not found

    """
    try:
        css_path = Path(__file__).parent / filename
        with css_path.open("r", encoding="utf-8") as f:
            css_content = f.read()
        return f"<style>\n{css_content}\n</style>"
    except FileNotFoundError:
        return ""


# =========================================================================
# Load CSS from External Files
# =========================================================================

# Simple notebook CSS (loaded from style.css)
NOTEBOOK_CSS = _load_css_file("style.css")

# Header HTML for notebook use only
HEADER_HTML_NOTEBOOK = f"""
<div style='
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, {ARK_BLUE}, {ARK_LIGHTBLUE});
    color: white;
    padding: 20px;
    margin: -8px -8px 20px -8px;
    border-radius: 0 0 8px 8px;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
'>
    Method of Moderation Illustrative Notebook
</div>
"""


def apply_ark_style() -> None:
    """Apply Econ-ARK matplotlib style to all plots."""
    plt.rcParams.update(MATPLOTLIB_STYLE)


def apply_notebook_css() -> None:
    """Apply simple notebook CSS styling for Jupyter notebooks."""
    try:
        from IPython.display import HTML, display

        display(HTML(NOTEBOOK_CSS))
    except ImportError:
        pass  # Running outside Jupyter environment


def setup_figure(figsize=(12, 8), title=None):
    """Create a figure with Econ-ARK styling applied."""
    apply_ark_style()
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="600", color=ARK_BLUE)
    return fig, ax
