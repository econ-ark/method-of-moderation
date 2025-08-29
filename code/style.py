"""Professional Econ-ARK branding and style definitions for Python applications.

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
MARKER_SIZE_STANDARD = 60
MARKER_EDGE_WIDTH_THIN = 1.2
MARKER_EDGE_COLOR = "white"

# Line styles
LINE_STYLE_DASHED = "--"
LINE_STYLE_DASHDOT = "-."
LINE_STYLE_DOTTED = ":"

# Y-axis limits for different plot types
YLIM_MODERATION_RATIO = (-0.1, 1.1)
YLIM_PRECAUTIONARY_GAPS = (-0.15, 0.35)
YLIM_VALUE_FUNCTION = (-6, 0)


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
HEADER_HTML_NOTEBOOK = """
<div style='
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #1f476b, #00aeef);
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
