# Images for "The Method of Moderation"

This directory contains the figures referenced in the document `moderation.md`.

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
