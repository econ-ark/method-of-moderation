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

A purpose-built stylesheet using CSS custom properties for ECON-ARK brand
compliance. Contains official ECON-ARK color schemes, plot styles, and
notebook-friendly typography. CSS lives in `style.css`, loaded from an
external file for easier customization.

Features
--------
- Ultra-minimal CSS with STRICT ECON-ARK brand compliance
- ONLY approved ECON-ARK brand colors (6 colors, no unauthorized additions)
- ONLY approved fonts: Roboto and Varela Round (no system font fallbacks)
- Professional typography hierarchy with full-width ARK underlines
- Clean text formatting and alignment utilities
- Lightweight stylesheet, brand-compliant
- Refined professional styling colors for enhanced visual polish
- Matplotlib plots with professional light panel background and refined colors

External Files
--------------
See `style.css` for the live stylesheet. The notebook header uses ECON-ARK
brand colors and fonts; h2 and h3 selectors get distinct accent colors.

The CSS file is automatically loaded when this module is imported, with
graceful fallback behavior (warns and returns an empty `<style>` block) if
the file is not found.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from IPython.display import HTML, display

logger = logging.getLogger(__name__)

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
    # Refined styling colors
    "ARK_GRID_SOFT",
    "ARK_LIGHTBLUE",
    "ARK_PANEL_LIGHT",
    "ARK_PINK",
    "ARK_SPINE",
    "ARK_TEXT",
    "ARK_YELLOW",
    # Concept colors mapping
    "CONCEPT_COLORS",
    # Font sizes
    "FONT_SIZE_LARGE",
    "FONT_SIZE_XLARGE",
    # Grid and padding
    "GRID_ALPHA",
    # Notebook styling
    "HEADER_HTML_NOTEBOOK",
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
    # Matplotlib configuration
    "MATPLOTLIB_STYLE",
    "NOTEBOOK_CSS",
    "PADDING_RATIO",
    # Reference line styling
    "REFERENCE_LINE_ALPHA",
    "REFERENCE_LINE_COLOR",
    "REFERENCE_LINE_WIDTH",
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
# Violet replaced ARK_PINK for the comparator because pink against ARK_GREEN is
# a red-green pair: under simulated deuteranopia it keeps 21% of its CIELAB
# separation (dE 139.5 -> 29.1), against 66% for violet (150.0 -> 99.5).
ARK_VIOLET = "#9b5de5"
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
    "egm": ARK_VIOLET,
    "optimist": ARK_LIGHTBLUE,
    "pessimist": ARK_YELLOW,
    "tight": ARK_GREY,
    # Same hue as "egm" on purpose: both are the naive comparator MoM is scored
    # against. No figure draws both; one that did would render them
    # identically, so give "linear" its own hue before writing it.
    "linear": ARK_VIOLET,
}

# Substring aliases mapped to concepts, in priority order: the first entry whose
# alias appears in a label wins. Ordered data rather than an if-chain so adding a
# concept costs one line and no branching.
CONCEPT_ALIASES = (
    (("truth", "high-precision", "realist"), "truth"),
    (("mom", "moderation"), "mom"),
    (("egm", "endogenous"), "egm"),
    (("linear",), "linear"),
    (("optimist", "perfect"), "optimist"),
    (("pessimist", "worst"), "pessimist"),
    (("tight",), "tight"),
)

FALLBACK_COLORS = (
    ARK_BLUE,
    ARK_GREEN,
    ARK_PINK,
    ARK_LIGHTBLUE,
    ARK_YELLOW,
    ARK_GREY,
)

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
    for aliases, concept in CONCEPT_ALIASES:
        if any(alias in name_lower for alias in aliases):
            return CONCEPT_COLORS[concept]
    # Unknown label: cycle deterministically. Character ordinals rather than
    # hash(), which is salted per session and would recolor between runs.
    return FALLBACK_COLORS[sum(ord(c) for c in name_lower) % len(FALLBACK_COLORS)]


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
    "axes.titlesize": 12,
    "axes.titleweight": "600",  # Bolder titles
    "axes.labelsize": 11,  # Academic-modest axis labels
    "axes.labelweight": "500",  # Slightly bolder labels
    "xtick.labelsize": 8.5,  # Slightly smaller tick labels
    "ytick.labelsize": 8.5,
    # Text colors - using refined professional colors
    "text.color": ARK_TEXT,
    "axes.labelcolor": ARK_TEXT,
    "axes.titlecolor": ARK_TEXT,  # Neutral academic text; ARK palette kept on the plotted lines
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
    "figure.dpi": 150,
    "savefig.dpi": 300,  # High resolution for PDF/print quality
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
    css_path = Path(__file__).parent / filename
    try:
        with css_path.open("r", encoding="utf-8") as f:
            css_content = f.read()
        return f"<style>\n{css_content}\n</style>"
    except FileNotFoundError:
        warnings.warn(
            f"CSS file not found: {css_path}. Notebook styling will be missing.",
            stacklevel=2,
        )
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
    """Apply simple notebook CSS styling for Jupyter notebooks.

    IPython is a resolved dependency of this project (via ipywidgets and voila),
    so it is imported at module scope rather than guarded: the former ImportError
    branch could not be reached from any environment `pyproject.toml` describes.
    """
    display(HTML(NOTEBOOK_CSS))


def setup_figure(figsize=(7, 4.5), title=None):
    """Create a figure with Econ-ARK styling applied.

    ``title`` is accepted for backward compatibility but intentionally not
    rendered: in the manuscript the LaTeX caption is the figure title, so an
    in-image figure title would only duplicate the caption (and the journal's
    own figure numbering).
    """
    del title  # not rendered; the caption is the title
    apply_ark_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax
