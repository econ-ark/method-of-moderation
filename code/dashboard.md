---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python tags=["hide-input"]
# Import Econ-ARK styling and display header
import ipywidgets as widgets
from IPython.display import display
from style import (
    DASHBOARD_CSS,
    HEADER_HTML_DASHBOARD,
    apply_ark_style,
    configure_dashboard_plotting,
    create_dashboard_widget,
    get_author_html,
    get_dashboard_container_layout,
    get_dashboard_left_panel_layout,
    get_dashboard_right_panel_layout,
    get_dashboard_row_layout,
    get_output_widget_layout,
    get_widget_layout,
    get_widget_style,
)

# Apply Econ-ARK branding and styling
apply_ark_style()

# Official ECON-ARK Brand Colors Available (ONLY APPROVED):
# ARK_BLUE (#1f476b) - Primary brand blue
# ARK_LIGHTBLUE (#00aeef) - Accent light blue
# ARK_PINK (#ed217c) - Brand pink
# ARK_GREEN (#39b54a) - Brand green
# ARK_YELLOW (#fcb040) - Brand yellow
# ARK_GREY (#676470) - Brand grey
#
# CONSISTENT COLOR ASSIGNMENTS (matches plotting.py):
# Truth → ARK_BLUE, MoM → ARK_GREEN, EGM → ARK_PINK
# Optimist → ARK_LIGHTBLUE, Pessimist → ARK_YELLOW, Tight → ARK_GREY
#
# CONSISTENT LINE STYLE ASSIGNMENTS (Figures 1 & 3):
# EGM in Figure 1 → dashed ("--"), MoM in Figure 3 → dashed ("--")

# Display Econ-ARK header (for dashboard)
try:
    display(widgets.HTML(HEADER_HTML_DASHBOARD))
except ImportError:
    pass  # Running outside Jupyter environment
```

```python
from __future__ import annotations

import builtins
import contextlib

# Import interactive widgets
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from moderation import (
    IndShockEGMConsumerType,
    IndShockMoMConsumerType,
    expit_moderate,
    log_mnrm_ex,
)
from plotting import (
    plot_chi_function,
    plot_moderation_ratio,
    plot_precautionary_gaps,
    plot_value_functions,
)
```

```python
def create_dashboard():
    """Create a dashboard with left panel controls and right panel 2x2 figure grid."""
    # Set matplotlib to not show figures inline initially
    plt.ioff()

    # Create output widgets for the four figures
    fig5_output = widgets.Output(layout=get_output_widget_layout())
    fig6_output = widgets.Output(layout=get_output_widget_layout())
    fig7_output = widgets.Output(layout=get_output_widget_layout())
    fig9_output = widgets.Output(layout=get_output_widget_layout())

    # Economic Parameters
    cycles_slider = widgets.IntSlider(
        value=1,
        min=0,  # 0 = infinite horizon
        max=10,
        step=1,
        description="Cycles:",
        style=get_widget_style(),
        layout=get_widget_layout(),
    )
    cycles_label = widgets.HTMLMath(
        value=r"Cycles (0=$\infty$):",
        layout=widgets.Layout(width="95%", margin="2px 0 0 2px"),
    )

    crra_slider = create_dashboard_widget(
        "FloatSlider",
        value=2.0,
        min=1.1,
        max=8.0,
        step=0.1,
        description="CRRA (rho):",
    )

    discfac_slider = widgets.FloatSlider(
        value=0.96,
        min=0.85,
        max=0.995,
        step=0.005,
        description="DiscFac (beta):",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    livprb_slider = widgets.FloatSlider(
        value=1.0,
        min=0.9,
        max=1.0,
        step=0.01,
        description="LivPrb:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    rfree_slider = widgets.FloatSlider(
        value=1.02,
        min=1.001,
        max=1.10,
        step=0.005,
        description="Rfree (R):",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    cubic_bool_checkbox = widgets.Checkbox(
        value=True,  # Default to cubic interpolation
        description="CubicBool:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    # Artificial Borrowing Constraint - checkbox + slider combo
    boro_cnst_enable_checkbox = widgets.Checkbox(
        value=False,  # Default to no borrowing constraint (BoroCnstArt = None)
        description="Enable BoroCnst:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    boro_cnst_value_slider = widgets.FloatSlider(
        value=0.0,
        min=-1.0,
        max=0.0,
        step=0.05,
        description="BoroCnstArt:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
        disabled=True,  # Start disabled since checkbox is False
    )

    # Link checkbox to slider enable/disable
    def on_boro_checkbox_change(change) -> None:
        boro_cnst_value_slider.disabled = not change["new"]

    boro_cnst_enable_checkbox.observe(on_boro_checkbox_change, names="value")

    # Income Process Parameters
    perm_gro_fac_slider = widgets.FloatSlider(
        value=1.0,
        min=0.98,
        max=1.05,
        step=0.005,
        description="PermGroFac:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    perm_shk_std_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.3,
        step=0.01,
        description="PermShkStd:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    perm_shk_count_slider = widgets.IntSlider(
        value=1,
        min=1,
        max=7,
        step=1,
        description="PermShkCount:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    tran_shk_std_slider = widgets.FloatSlider(
        value=1.0,
        min=0.1,
        max=3.0,
        step=0.1,
        description="TranShkStd (sigma):",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    tran_shk_count_slider = widgets.IntSlider(
        value=7,
        min=3,
        max=15,
        step=1,
        description="TranShkCount:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    unemp_prb_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=0.2,
        step=0.01,
        description="UnempPrb:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    inc_unemp_slider = widgets.FloatSlider(
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.05,
        description="IncUnemp:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    # Dense Grid Parameters (for "truth" solution)
    dense_axtra_min_slider = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.0005,
        description="Dense aMin:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    dense_axtra_max_slider = widgets.FloatSlider(
        value=40,  # Match notebook default
        min=10,
        max=100,
        step=5,
        description="Dense aMax:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    dense_axtra_count_slider = widgets.IntSlider(
        value=500,  # Match notebook default
        min=50,
        max=1000,
        step=50,
        description="Dense Count:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    dense_nest_fac_slider = widgets.IntSlider(
        value=3,  # Match notebook default
        min=1,
        max=5,
        step=1,
        description="Dense NestFac:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    # Sparse Grid Parameters
    sparse_axtra_min_slider = widgets.FloatSlider(
        value=0.001,
        min=0.0001,
        max=0.01,
        step=0.0005,
        description="Sparse aMin:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    sparse_axtra_max_slider = widgets.FloatSlider(
        value=4,  # Match notebook default (integer)
        min=1.0,
        max=15.0,
        step=0.5,
        description="Sparse aMax:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    sparse_axtra_count_slider = widgets.IntSlider(
        value=5,
        min=3,
        max=25,
        step=1,
        description="Sparse Count:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    sparse_nest_fac_slider = widgets.IntSlider(
        value=-1,
        min=-1,
        max=5,
        step=1,
        description="Sparse NestFac:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="95%", margin="1px 0"),
    )

    def update_all_figures(
        cycles,
        crra,
        discfac,
        livprb,
        rfree,
        boro_cnst_enable,
        boro_cnst_value,
        perm_gro_fac,
        perm_shk_std,
        perm_shk_count,
        tran_shk_std,
        tran_shk_count,
        unemp_prb,
        inc_unemp,
        dense_axtra_min,
        dense_axtra_max,
        dense_axtra_count,
        dense_nest_fac,
        sparse_axtra_min,
        sparse_axtra_max,
        sparse_axtra_count,
        sparse_nest_fac,
        cubic_bool,
    ) -> None:
        """Update all four figures based on current parameter values."""
        # Clear all outputs
        for output in [fig5_output, fig6_output, fig7_output, fig9_output]:
            output.clear_output(wait=True)

        # Set up parameters (matching notebook.md structure exactly)
        # Interpret slider value: 0 => infinite horizon; n>0 => finite horizon of length n
        if cycles and cycles > 0:
            cycles_flag = 1
            T_cycle = int(cycles)
        else:
            cycles_flag = 0
            T_cycle = 1

        # Core economic parameters
        params_current = {
            "cycles": cycles_flag,
            "T_cycle": T_cycle,
            "CRRA": crra,
            "DiscFac": discfac,
            "LivPrb": [livprb] * T_cycle,
            "Rfree": [rfree] * T_cycle,
            "BoroCnstArt": boro_cnst_value if boro_cnst_enable else None,
            "CubicBool": cubic_bool,
            "vFuncBool": True,  # Enable value function computation
        }

        # Income process parameters
        # Make the time-profile length match 'cycles' for finite-horizon models,
        # and use length 1 for infinite-horizon (cycles = 0).
        income_process_current = {
            "PermGroFac": [perm_gro_fac] * T_cycle,
            "PermShkStd": [perm_shk_std] * T_cycle,
            "PermShkCount": perm_shk_count,
            "TranShkStd": [tran_shk_std] * T_cycle,
            "TranShkCount": tran_shk_count,
            "UnempPrb": unemp_prb,
            "IncUnemp": inc_unemp,
        }

        # Combine all parameters into a single dictionary
        consumer_current = params_current | income_process_current

        # Grid configurations (matching notebook.md exactly)
        dense_grid = {
            "aXtraMin": dense_axtra_min,
            "aXtraMax": dense_axtra_max,
            "aXtraNestFac": dense_nest_fac,
            "aXtraCount": dense_axtra_count,
            "aXtraExtra": None,
        }

        sparse_grid = {
            "aXtraMin": sparse_axtra_min,
            "aXtraMax": sparse_axtra_max,
            "aXtraNestFac": sparse_nest_fac,
            "aXtraCount": sparse_axtra_count,
            "aXtraExtra": None,
        }

        try:
            # Solve models (vFuncBool already in consumer_current)
            truth_params = consumer_current | dense_grid
            truth_agent = IndShockEGMConsumerType(**truth_params)
            truth_agent.solve()
            truth_sol = truth_agent.solution[0]

            egm_params = consumer_current | sparse_grid
            egm_agent = IndShockEGMConsumerType(**egm_params)
            egm_agent.solve()
            egm_sol = egm_agent.solution[0]

            mom_params = consumer_current | sparse_grid
            mom_agent = IndShockMoMConsumerType(**mom_params)
            mom_agent.solve()
            mom_sol = mom_agent.solution[0]

            # Unpack bound solutions for easier access
            truth_opt = truth_sol.Optimist
            truth_pes = truth_sol.Pessimist
            truth_tight = truth_sol.TighterUpperBound

            # Define gap functions
            def truth_gap(m):
                return truth_opt.cFunc(m) - truth_sol.cFunc(m)

            def egm_gap(m):
                return truth_opt.cFunc(m) - egm_sol.cFunc(m)

            def mom_gap(m):
                return truth_opt.cFunc(m) - mom_sol.cFunc(m)

            # Create grids (matching notebook.md)
            m_grid_wide = np.linspace(egm_sol.mNrmMin + 0.001, 30, 100)
            m_grid_vfunc = np.linspace(egm_sol.mNrmMin + 0.001, 3.0, 100)

            # Configure matplotlib and plotting for dashboard display
            capture_output = configure_dashboard_plotting()
            # All figure sizing and management now handled by style.py configure_dashboard_plotting()
            # The above function sets up matplotlib backend, monkey-patches plotting.setup_figure,
            # and returns the capture_output context manager with enforce_size function

            # Figure 5: Direct Method Comparison
            with fig5_output, capture_output() as enforce_size:
                plot_precautionary_gaps(
                    m_grid=m_grid_wide,
                    truth_gap=truth_gap(m_grid_wide),
                    approx_gap=[egm_gap(m_grid_wide), mom_gap(m_grid_wide)],
                    legend=["EGM Approximation", "MoM Approximation"],
                    title="Figure 5: Direct Method Comparison",
                    subtitle=f"EGM vs MoM Extrapolation Performance (cycles={'∞' if cycles_flag == 0 else T_cycle})",
                    solution=mom_sol,
                    grid_points=True,
                )
                enforce_size()

            # Prepare shared MoM chi function and constants once
            logitModRteFunc = mom_sol.cFunc.logitModRteFunc
            m_min = mom_sol.cFunc.mNrmMin

            # Figure 6: Moderation Ratio Function
            with fig6_output, capture_output() as enforce_size:
                # Create specific grid for moderation ratio
                m_grid_fig6 = np.linspace(m_min + 0.01, 50, 200)
                # Convert market resources to mu (log excess resources)
                mu_grid_fig6 = log_mnrm_ex(m_grid_fig6, m_min)
                # Evaluate moderation ratio omega(mu)
                chi_values_fig6 = logitModRteFunc(mu_grid_fig6)
                omega_values_fig6 = expit_moderate(chi_values_fig6)

                plot_moderation_ratio(
                    m_grid=m_grid_fig6,
                    omega_values=omega_values_fig6,
                    title=r"Figure 6: Consumption Moderation Ratio $\omega(m)$",
                    subtitle="Wealth-Dependent Moderation Between Bounds",
                    solution=mom_sol,
                    grid_type="consumption",
                )
                enforce_size()

            # Figure 7: Logit Transformation
            with fig7_output:
                with capture_output() as enforce_size:
                    # Create market resources grid from near constraint to high wealth
                    m_grid_chi = np.linspace(m_min + 0.001, 50, 200)
                    # Convert to mu grid for x-axis (this is the natural domain for logit function)
                    mu_grid_chi = log_mnrm_ex(m_grid_chi, m_min)
                    # Evaluate logit function over this extended range
                    chi_values_fig7 = logitModRteFunc(mu_grid_chi)

                    plot_chi_function(
                        mu_grid=mu_grid_chi,
                        chi_values=chi_values_fig7,
                        title="Figure 7: Logit Transformation for Stable Extrapolation",
                        subtitle="Unbounded Transformation for Stable Extrapolation",
                        solution=mom_sol,
                        grid_points=True,
                    )
                    enforce_size()

            # Figure 9: Value Functions
            with fig9_output, capture_output() as enforce_size:
                # Evaluate value functions on the grid
                v_truth = truth_sol.vFunc(m_grid_vfunc)  # Truth
                v_opt = truth_opt.vFunc(m_grid_vfunc)  # Optimist
                v_pes = truth_pes.vFunc(m_grid_vfunc)  # Pessimist
                v_tight = truth_tight.vFunc(m_grid_vfunc)  # Tight upper bound
                v_egm_sparse = egm_sol.vFunc(m_grid_vfunc)  # EGM Approximation
                v_mom_sparse = mom_sol.vFunc(m_grid_vfunc)  # MoM Approximation

                plot_value_functions(
                    m_grid=m_grid_vfunc,
                    title="Figure 9: Value Functions Bounded by Economic Theory",
                    subtitle="Value Function: True Solution vs Sparse Approximations",
                    v_truth=v_truth,
                    v_opt=v_opt,
                    v_pes=v_pes,
                    v_tight=v_tight,
                    v_egm_sparse=v_egm_sparse,
                    v_mom_sparse=v_mom_sparse,
                    figure_num=7,
                    mom_solution=mom_sol,
                    egm_solution=None,
                )
                enforce_size()

        except Exception:
            debug = False  # set True to print tracebacks in the UI
            if debug:
                import traceback as _tb

                _err = _tb.format_exc()
                for output in [fig5_output, fig6_output, fig7_output, fig9_output]:
                    with output:
                        pass
        finally:
            # Restore original setup_figure if it was patched
            with contextlib.suppress(builtins.BaseException):
                plotting.setup_figure = original_setup_figure

    # Create left panel with narrative and controls + smooth accordion animations
    narrative_html = widgets.HTML(
        value=f"""
        <div class="dashboard-narrative">
            <h2 class="dashboard-title">Method of Moderation Dashboard</h2>
            <p class="dashboard-author">{get_author_html()}</p>
            <p class="dashboard-description">Explore how the Method of Moderation solves extrapolation problems in sparse EGM grids. This dashboard shows the key figures demonstrating MoM's theoretical foundation and performance.</p>
            <small>Note: Greek letters shown as: rho (&rho;), beta (&beta;), sigma (&sigma;)</small>
        </div>
        """,
    )

    # Create collapsible sections for better organization
    econ_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.HBox(
                        [
                            cycles_label,
                        ],
                    ),
                    cycles_slider,
                    crra_slider,
                    discfac_slider,
                    livprb_slider,
                    rfree_slider,
                    cubic_bool_checkbox,
                    boro_cnst_enable_checkbox,
                    boro_cnst_value_slider,
                ],
            ),
        ],
    )
    econ_accordion.set_title(0, "Economic Parameters")
    econ_accordion.selected_index = 0  # Start expanded

    income_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    perm_gro_fac_slider,
                    perm_shk_std_slider,
                    perm_shk_count_slider,
                    tran_shk_std_slider,
                    tran_shk_count_slider,
                    unemp_prb_slider,
                    inc_unemp_slider,
                ],
            ),
        ],
    )
    income_accordion.set_title(0, "Income Process")
    income_accordion.selected_index = None  # Start collapsed

    dense_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    dense_axtra_min_slider,
                    dense_axtra_max_slider,
                    dense_axtra_count_slider,
                    dense_nest_fac_slider,
                ],
            ),
        ],
    )
    dense_accordion.set_title(0, "Dense Grid (Truth)")
    dense_accordion.selected_index = None  # Start collapsed

    sparse_accordion = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    sparse_axtra_min_slider,
                    sparse_axtra_max_slider,
                    sparse_axtra_count_slider,
                    sparse_nest_fac_slider,
                ],
            ),
        ],
    )
    sparse_accordion.set_title(0, "Sparse Grid (EGM/MoM)")
    sparse_accordion.selected_index = None  # Start collapsed

    # Make accordions mutually exclusive (only one open at a time)
    accordions = [
        econ_accordion,
        income_accordion,
        dense_accordion,
        sparse_accordion,
    ]

    def make_exclusive_handler(current_accordion, other_accordions):
        def on_accordion_change(change) -> None:
            if change["name"] == "selected_index" and change["new"] is not None:
                # Close all other accordions
                for accordion in other_accordions:
                    if accordion != current_accordion:
                        accordion.selected_index = None

        return on_accordion_change

    # Attach exclusive behavior to each accordion
    for i, accordion in enumerate(accordions):
        other_accordions = [acc for j, acc in enumerate(accordions) if j != i]
        accordion.observe(
            make_exclusive_handler(accordion, other_accordions),
            names="selected_index",
        )

    # Left panel content with scrolling for parameters
    left_panel = widgets.VBox(
        [
            narrative_html,
            econ_accordion,
            income_accordion,
            dense_accordion,
            sparse_accordion,
        ],
        layout=get_dashboard_left_panel_layout(),
    )

    # Right panel: 2x2 grid of figures
    top_row = widgets.HBox(
        [fig5_output, fig6_output],
        layout=get_dashboard_row_layout(),
    )

    bottom_row = widgets.HBox(
        [fig7_output, fig9_output],
        layout=get_dashboard_row_layout(),
    )

    right_panel = widgets.VBox(
        [top_row, bottom_row],
        layout=get_dashboard_right_panel_layout(),
    )

    # Main dashboard layout
    dashboard = widgets.HBox(
        [left_panel, right_panel],
        layout=get_dashboard_container_layout(),
    )

    # Create interactive updating
    interactive_update = widgets.interactive(
        update_all_figures,
        cycles=cycles_slider,
        crra=crra_slider,
        discfac=discfac_slider,
        livprb=livprb_slider,
        rfree=rfree_slider,
        boro_cnst_enable=boro_cnst_enable_checkbox,
        boro_cnst_value=boro_cnst_value_slider,
        perm_gro_fac=perm_gro_fac_slider,
        perm_shk_std=perm_shk_std_slider,
        perm_shk_count=perm_shk_count_slider,
        tran_shk_std=tran_shk_std_slider,
        tran_shk_count=tran_shk_count_slider,
        unemp_prb=unemp_prb_slider,
        inc_unemp=inc_unemp_slider,
        dense_axtra_min=dense_axtra_min_slider,
        dense_axtra_max=dense_axtra_max_slider,
        dense_axtra_count=dense_axtra_count_slider,
        dense_nest_fac=dense_nest_fac_slider,
        sparse_axtra_min=sparse_axtra_min_slider,
        sparse_axtra_max=sparse_axtra_max_slider,
        sparse_axtra_count=sparse_axtra_count_slider,
        sparse_nest_fac=sparse_nest_fac_slider,
        cubic_bool=cubic_bool_checkbox,
    )

    # Don't display the interactive widget controls (we have our own)
    interactive_update.children = ()

    # Trigger initial update with notebook defaults
    update_all_figures(
        cycles=1,
        crra=2.0,
        discfac=0.96,
        livprb=1.0,
        rfree=1.02,
        boro_cnst_enable=False,
        boro_cnst_value=0.0,
        perm_gro_fac=1.0,
        perm_shk_std=0.0,
        perm_shk_count=1,
        tran_shk_std=1.0,
        tran_shk_count=7,
        unemp_prb=0.0,
        inc_unemp=0.0,
        dense_axtra_min=0.001,
        dense_axtra_max=40,  # Match notebook
        dense_axtra_count=500,  # Match notebook
        dense_nest_fac=3,  # Match notebook
        sparse_axtra_min=0.001,
        sparse_axtra_max=4,
        sparse_axtra_count=5,
        sparse_nest_fac=-1,
        cubic_bool=True,  # Default to cubic interpolation
    )

    return dashboard


# Create and display the dashboard
dashboard = create_dashboard()
display(dashboard)
# Ensure the CSS is applied after the dashboard to override notebook defaults
with contextlib.suppress(Exception):
    display(widgets.HTML(DASHBOARD_CSS))
```

```python tags=["hide-input"]
# (moved CSS application just after dashboard creation above)
```
