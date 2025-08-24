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
BBOX_PADDING = 0.4  # Padding for text boxes

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
FONT_SIZE_SMALL = 11
FONT_SIZE_MEDIUM = 12
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
ALPHA_VERY_HIGH = 0.9
ALPHA_NEARLY_OPAQUE = 0.95
ALPHA_OPAQUE = 1.0

# Marker styling (unify edge color with global style)
MARKER_SIZE_STANDARD = 60
MARKER_SIZE_LARGE = 80
MARKER_EDGE_WIDTH_THIN = 1.2
MARKER_EDGE_WIDTH_THICK = 1.5
MARKER_EDGE_COLOR = "white"

# Line styles
LINE_STYLE_SOLID = "-"
LINE_STYLE_DASHED = "--"
LINE_STYLE_DASHDOT = "-."
LINE_STYLE_DOTTED = ":"

# Y-axis limits for different plot types
YLIM_MODERATION_RATIO = (-0.1, 1.1)
YLIM_PRECAUTIONARY_GAPS = (-0.15, 0.35)
YLIM_VALUE_FUNCTION = (-6, 0)

# Dashboard widget styling constants
WIDGET_DESCRIPTION_WIDTH = "120px"
WIDGET_LAYOUT_WIDTH = "95%"
WIDGET_LAYOUT_MARGIN = "1px 0"
WIDGET_OUTPUT_WIDTH = "100%"
WIDGET_OUTPUT_HEIGHT = "100%"
WIDGET_OUTPUT_OVERFLOW = "hidden"

# Dashboard layout constants
DASHBOARD_ROW_HEIGHT_VH = 0.475  # 47.5vh per row (95vh / 2)
DASHBOARD_FIGURE_HEIGHT_RATIO = 0.92  # 92% of row height for figure
DASHBOARD_FIGURE_WIDTH_RATIO = 0.95  # 95% of available width
DASHBOARD_LEFT_PANEL_WIDTH_FRACTION = (
    0.3  # *** CHANGE THIS FRACTION TO RESIZE LEFT PANEL *** (e.g., 1/3, 0.25, 0.4)
)
DASHBOARD_PANEL_SPACING = 20  # Fixed spacing in pixels between left and right panels
DASHBOARD_ASPECT_RATIO = 0.55  # Figure aspect ratio (height/width)
DASHBOARD_MIN_FIGURE_WIDTH = 8.0
DASHBOARD_MAX_FIGURE_WIDTH = 10.0

# Approximate sticky header height so the container fits in the viewport.
# Use a conservative value to avoid any page-level scrollbar from minor rounding.
HEADER_HEIGHT_PX = 110

# Author styling
AUTHOR_LABEL_COLOR = "black"
AUTHOR_NAME_COLOR = ARK_LIGHTBLUE  # Alan Lujan color
AUTHOR_INSTITUTION_COLOR = ARK_BLUE  # Johns Hopkins University color

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

# Complex dashboard CSS (loaded from style_dashboard.css)
DASHBOARD_CSS = _load_css_file("style_dashboard.css")

# Notebook-specific CSS overrides to re-enable scrolling and natural heights
# when the dashboard CSS has been loaded. These rules appear after the
# dashboard CSS and therefore take precedence.
NOTEBOOK_RESET_CSS = ""


USE_REMOTE_LOGOS = True  # toggle to False to avoid remote assets in headers


def _get_header_html(title: str) -> str:
    """Generate header HTML with dynamic title.

    Note: CSS styling is included in the main DASHBOARD_CSS.

    Parameters
    ----------
    title : str
        The title to display in the header

    Returns
    -------
    str
        Header HTML content (CSS loaded separately via DASHBOARD_CSS)

    """
    if USE_REMOTE_LOGOS:
        return f"""
<div class='ark-header'>
  <div class='ark-header-left'>
    <a href='https://econ-ark.org' target='_blank' style='border:0'>
      <img src='https://econ-ark.org/assets/img/econ-ark-logo-white.png'
           alt='Econ-ARK logo'>
    </a>
  </div>
  <div class='ark-header-center'>
    <span>{title}</span>
  </div>
  <div class='ark-header-right'>
    <a href='https://www.jhu.edu' target='_blank' style='border:0'>
      <img src='https://brand.jhu.edu/wp-content/uploads/2024/08/shield-jhu-scaled.jpg'
           alt='Johns Hopkins University logo'>
    </a>
  </div>
</div>
"""
    return f"""
<div class='ark-header'>
  <div class='ark-header-center'>
    <span>{title}</span>
  </div>
</div>
"""


# Header HTML with Econ-ARK logo (CSS included in DASHBOARD_CSS)
HEADER_HTML_NOTEBOOK = _get_header_html("Method of Moderation Illustrative Notebook")
HEADER_HTML_DASHBOARD = _get_header_html("Method of Moderation Interactive Dashboard")


def tidy_legend(fig) -> None:
    """Deprecated helper; no-op (kept for compatibility)."""
    return


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


def apply_dashboard_css() -> None:
    """Apply complex dashboard CSS styling for interactive dashboards."""
    try:
        from IPython.display import HTML, display

        display(HTML(DASHBOARD_CSS))
    except ImportError:
        pass  # Running outside Jupyter environment


def setup_figure(figsize=(12, 8), title=None):
    """Create a figure with Econ-ARK styling applied."""
    apply_ark_style()
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="600", color=ARK_BLUE)
    return fig, ax


def setup_subplots(*args, **kwargs):
    """Deprecated; use setup_figure instead."""
    return setup_figure()


def get_widget_layout():
    """Get consistent widget layout for dashboard controls."""
    import ipywidgets as widgets

    return widgets.Layout(width=WIDGET_LAYOUT_WIDTH, margin=WIDGET_LAYOUT_MARGIN)


def get_widget_style():
    """Get consistent widget style for dashboard controls."""
    return {"description_width": WIDGET_DESCRIPTION_WIDTH}


def get_output_widget_layout():
    """Get consistent layout for output widgets in dashboard."""
    import ipywidgets as widgets

    return widgets.Layout(
        width=WIDGET_OUTPUT_WIDTH,
        height=WIDGET_OUTPUT_HEIGHT,
        overflow=WIDGET_OUTPUT_OVERFLOW,
        padding="5px",
        flex="1 1 auto",
        min_height="0",
        min_width="0",
    )


def calculate_dashboard_figure_size():
    """Return static figure size suited for dashboard; drop tkinter dependence."""
    fallback_width = 9.0
    fallback_height = fallback_width * DASHBOARD_ASPECT_RATIO
    return (fallback_width, fallback_height)


def calculate_dynamic_panel_widths():
    """Return static panel widths based on a nominal screen width; drop tkinter."""
    fallback_screen_width = 1920
    left_panel_width_px = int(
        fallback_screen_width * DASHBOARD_LEFT_PANEL_WIDTH_FRACTION,
    )
    total_panel_space_px = left_panel_width_px + DASHBOARD_PANEL_SPACING
    return {
        "left_width": f"{left_panel_width_px}px",
        "right_width": f"calc(100% - {total_panel_space_px}px)",
        "left_width_px": left_panel_width_px,
        "total_space_px": total_panel_space_px,
    }


def get_author_html() -> str:
    """Get properly styled author HTML using style constants."""
    return f"""
                <span style="font-weight: bold; color: {AUTHOR_LABEL_COLOR};">Author:</span>
                <span style="font-weight: bold; color: {AUTHOR_NAME_COLOR};">Alan Lujan</span>,
                <span style="font-weight: bold; color: {AUTHOR_INSTITUTION_COLOR};">Johns Hopkins University</span>
    """


def create_dashboard_widget(widget_type, **kwargs):
    """Create a consistently styled dashboard widget.

    Parameters
    ----------
    widget_type : str
        Type of widget ('FloatSlider', 'IntSlider', 'Checkbox')
    **kwargs
        Widget-specific parameters (value, min, max, step, description, etc.)

    Returns
    -------
    widget
        Configured widget with consistent styling

    """
    import ipywidgets as widgets

    # Remove style and layout from kwargs if present (we'll override them)
    kwargs.pop("style", None)
    kwargs.pop("layout", None)

    # Create the widget with consistent styling
    widget_class = getattr(widgets, widget_type)
    return widget_class(
        style=get_widget_style(),
        layout=get_widget_layout(),
        **kwargs,
    )


# Dashboard layout constants for containers (static values only)
DASHBOARD_LEFT_PANEL_HEIGHT = "100%"
DASHBOARD_LEFT_PANEL_PADDING = "5px"
DASHBOARD_RIGHT_PANEL_HEIGHT = "100%"
DASHBOARD_CONTAINER_HEIGHT = f"calc(100vh - {HEADER_HEIGHT_PX}px)"
DASHBOARD_ROW_HEIGHT = "50%"
DASHBOARD_BORDER = "0"


def get_dashboard_left_panel_layout():
    """Get layout for dashboard left panel with dynamic width."""
    import ipywidgets as widgets

    panel_widths = calculate_dynamic_panel_widths()
    return widgets.Layout(
        width=panel_widths["left_width"],
        height=DASHBOARD_LEFT_PANEL_HEIGHT,
        padding=DASHBOARD_LEFT_PANEL_PADDING,
        overflow_y="auto",
        overflow_x="hidden",
        min_height="0",
    )


def get_dashboard_right_panel_layout():
    """Get layout for dashboard right panel with dynamic width."""
    import ipywidgets as widgets

    panel_widths = calculate_dynamic_panel_widths()
    return widgets.Layout(
        width=panel_widths["right_width"],
        height=DASHBOARD_RIGHT_PANEL_HEIGHT,
        overflow="hidden",  # prevent scrollbars on figures
        min_height="0",
        min_width="0",
    )


def get_dashboard_container_layout():
    """Get layout for main dashboard container."""
    import ipywidgets as widgets

    return widgets.Layout(
        width="100%",
        height="auto",
        overflow="hidden",
        border=DASHBOARD_BORDER,
        position="fixed",
        top=f"{HEADER_HEIGHT_PX}px",
        left="0",
        right="0",
        bottom="0",
    )


def get_dashboard_row_layout():
    """Get layout for dashboard figure rows."""
    import ipywidgets as widgets

    return widgets.Layout(
        width="100%",
        height=DASHBOARD_ROW_HEIGHT,
        align_items="stretch",
        overflow="hidden",
        min_height="0",
    )


# Dashboard figure management functions
def setup_dashboard_figures():
    """Set up matplotlib for dashboard figure display.

    Returns
    -------
    tuple
        (dynamic_figsize, dashboard_setup_figure_func, capture_output_context)

    """
    import sys
    from contextlib import contextmanager
    from io import StringIO

    import matplotlib.pyplot as plt

    # Get dynamic figure size
    dynamic_figsize = calculate_dashboard_figure_size()

    # Store original setup_figure for restoration
    original_setup_figure = setup_figure

    def dashboard_setup_figure(figsize=None, title="", **kwargs):
        """Override setup_figure to ALWAYS use dynamically calculated sizes."""
        # IGNORE the passed figsize - always use our calculated size
        figsize = dynamic_figsize

        # Create figure with our size
        fig, ax = original_setup_figure(figsize=figsize, title=title, **kwargs)

        # Skip tight_layout as it can cause inconsistent sizes
        # Force exact margins for all figures
        fig.subplots_adjust(
            bottom=0.12,  # Space for x-axis labels
            left=0.10,  # Space for y-axis labels
            right=0.98,  # Minimal right margin
            top=0.92,  # Space for title
        )

        # Force the figure canvas to update with exact size
        fig.set_size_inches(figsize, forward=True)

        # Ensure DPI is consistent
        fig.set_dpi(100)

        return fig, ax

    def enforce_figure_size() -> None:
        """Ensure current figure has the correct size."""
        fig = plt.gcf()
        fig.set_size_inches(dynamic_figsize, forward=True)
        fig.set_dpi(100)

    @contextmanager
    def capture_output():
        """Capture text output and ensure matplotlib figures display properly in widgets."""
        import matplotlib.pyplot as plt
        from IPython.display import display

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # Redirect text output to string buffers
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        # Store current matplotlib state and turn off interactive mode
        was_interactive = plt.isinteractive()
        plt.ioff()  # Prevent automatic figure display outside widget

        try:
            yield enforce_figure_size
            # After plotting function completes, display the current figure in widget
            current_fig = plt.gcf()
            if current_fig.get_axes():  # Only display if figure has content
                display(current_fig)
        finally:
            # Restore matplotlib interactive state
            if was_interactive:
                plt.ion()

            # Restore original streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return dynamic_figsize, dashboard_setup_figure, capture_output


def configure_dashboard_plotting():
    """Configure matplotlib and plotting module for dashboard use.

    Returns
    -------
    function
        capture_output context manager for use with plotting functions

    """
    # Set matplotlib backend to not print figure info
    import matplotlib as mpl
    import plotting

    mpl.rcParams["figure.max_open_warning"] = 0

    # Get dashboard figure configuration
    dynamic_figsize, dashboard_setup_figure, capture_output = setup_dashboard_figures()

    # Monkey-patch the setup_figure in plotting module
    plotting.setup_figure = dashboard_setup_figure

    return capture_output
