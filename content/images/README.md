# Images for "The Method of Moderation"

This directory contains the figures referenced in the paper sources under `content/paper/` (`moderation_extended.md`, `moderation_letters.md`, and `moderation_with_appendix.md`).

## Current Images

The following image files are currently in this directory:

1. **ExtrapProblemPlot.pdf**
   - Shows predicted precautionary saving becoming negative for large market
     resources
   - Demonstrates the problem with linear extrapolation in endogenous gridpoints
     method
   - Referenced as Figure {ref}`fig:ExtrapProblem` in the paper

2. **IntExpFOCInvPesReaOptNeedHiPlot.pdf**
   - Graph showing consumption functions for pessimist, realist, and optimist
     consumers
   - Illustrates the key concept:
     $\underline{c}_{t-1} < \tilde{c}_{t-1} < \bar{c}_{t-1}$
   - Referenced as Figure {ref}`fig:IntExpFOCInvPesReaOptNeedHi` in the paper

3. **ExtrapProblemSolvedPlot.pdf**
   - Shows accurate extrapolation using the Method of Moderation
   - Demonstrates the improved performance compared to traditional methods
   - Referenced as Figure {ref}`fig:ExtrapProblemSolved` in the paper

4. **IntExpFOCInvPesReaOptNeed45Plot.pdf**
   - Graph demonstrating the implementation of a tighter upper bound constraint
   - Shows the refined method discussed in Section 5.1
   - Referenced as Figure {ref}`fig:IntExpFOCInvPesReaOptNeed45` in the paper

5. **SolutionErrorPlot.pdf**
   - Absolute consumption error across the whole domain, on a logarithmic
     axis, for sparse-grid EGM and MoM against the high-precision truth
   - The visual companion to the accuracy table: error dips at gridpoints,
     peaks between them, and diverges for EGM beyond the top gridpoint
   - Referenced as Figure {ref}`fig:SolutionError` in the paper

6. **StochasticBoundsPlot.pdf**
   - Consumption bounds under deterministic versus stochastic (i.i.d.) returns
   - Shows both optimist and pessimist bounds tightening under a
     mean-preserving spread of the gross return (for CRRA > 1)
   - Referenced as Figure {ref}`fig:StochasticBounds` in the paper

7. **LogitFunctionPlot.pdf**
   - The object the method actually interpolates: the logit moderation ratio
     $\logitModRte(\logmNrmEx)$ for the five-gridpoint solution, smooth and
     close to linear at both ends of the grid
   - Referenced as Figure {ref}`fig:LogitFunction` in the paper

8. **GridExtentPlot.pdf**
   - Beyond-grid extrapolation error of the converged infinite-horizon rule
     against the grid top in units of human wealth, double vs triple
     exponential spacing (about a third apart), with the paper's five-point
     grid and the twice-human-wealth reference marked
   - Referenced as Figure {ref}`fig:GridExtent` in the paper
   - NOTE: unlike the other figures, this one regenerates from
     `code/explorations/grid-extent/` (`run_grid_extent.py` then
     `make_exhibit.py`), not from `make_paper_figures.py`

## Reproducing These Figures

All figures are regenerated from the solver and the shared plotting layer by

```bash
uv run python code/make_paper_figures.py
```

which writes each figure as both `.pdf` (the build source) and `.png`. Figures
carry no in-image title: the LaTeX caption is the figure title. Edit
`code/plotting.py` (drawing) or `code/make_paper_figures.py` (calibration and
output names), then rerun the command.

## Image Specifications

- **Format**: PDF (current), will be converted to PNG/WebP by MyST during build
- **Resolution**: High resolution for publication quality
- **Size**: Figures are displayed at 80% width in the paper
- **Alt text**: Each image has descriptive alt text for accessibility

## MyST Figure Features

These images use MyST's figure directive with:

- Semantic labels for cross-referencing (e.g., `fig:ExtrapProblem`)
- Alt text for accessibility
- Consistent styling (center-aligned, 80% width)
- Proper captions explaining the mathematical content

## Build Process

During the MyST build process, these PDF images are automatically converted to:

- PNG format for web display
- WebP format for optimized web delivery
- Other formats as needed for different export targets

The conversion process requires ImageMagick with PDF support enabled.
