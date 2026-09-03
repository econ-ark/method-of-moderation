# Grid design and beyond-grid extrapolation accuracy

Exploration supporting fig:GridExtent in the extended paper and the reply to
Chris's 2026-08-14 grid proposal (JSON results are gitignored and
regenerable; the JSON carries a params stamp that `make_exhibit.py` checks
before plotting). Each clause of the proposal is an arm of the experiment,
and each gets a measured verdict.

Setting: deterministic MoM, paper calibration, infinite horizon (h = 50),
linear interpolation (the paper's baseline), all solves convergence-checked
at tolerance 1e-10, scored with max absolute consumption error against a
dense truth (aXtraMax 2000 with 600 points) on the beyond-grid window from
just past the last real gridpoint out to m = 1000. Measurement notes from
the 2026-08-14 review pass: the beyond-grid window now starts at the last
REAL knot (the interpolant's synthetic extrapolation knot previously shifted
the window start by a factor exp(0.5)), it is built in excess-resources
space (the infinite-horizon borrowing limit is -hNrmPes = -6.77, so the
five-point arm's grid top sits at negative m), and solves terminate on a
measured convergence criterion rather than a blind cycle cap. These fixes
plus the linear baseline changed the headline numbers relative to the first
run of this exploration.

## Clause-by-clause verdicts

**"0,1,2,3,4,5 is pretty compelling but doesn't honestly work very well."**
Supported for production use: the linear five-point grid's beyond-grid max
error is 2.1e-2 (1.3e-2 relative). The paper already frames that grid as an
illustration with a 30-80 point production scope; this measurement
quantifies the gap.

**"Double or triple exponential gridpoints."** Mostly either: the two
spacings coincide at extents up to human wealth (within 0.3%) and part
company modestly past it, with double exponential about a third more
accurate beyond the grid at a = 2h and 4h (1.8e-3 vs 2.7e-3; 8.8e-4 vs
1.3e-3). Nothing large rides on the choice, but the "or" is not exactly
free at production extents. Extent, not spacing, still carries the bulk of
the improvement.

**"Up to a ratio of a to h ... at least a ratio of 2 times human wealth."**
Supported as a floor: a = 2h improves beyond-grid accuracy roughly an order
of magnitude over the five-point grid (11.6x with double spacing, 7.8x with
triple), and the gains do NOT saturate there - a further factor of about
two arrives by a = 4h. The criterion is a good operating point, not an
optimum.

**"We want it to be robust - that's part of the success of EGM."** The
robustness split the paper already makes: the theoretical bounds hold on
EVERY arm by construction, including the five-point grid - extrapolated
consumption never leaves the optimist-pessimist bracket at any extent, which
is the failure mode linear EGM extrapolation actually has. Grid extent buys
accuracy on top of that unconditional floor; it is not what keeps the
procedure well-behaved.

| grid | a/h | max abs beyond grid | max rel |
|---|---|---|---|
| linear 5-point (paper) | 0.08 | 2.1e-2 | 1.3e-2 |
| 30-point, nest 3, same extent | 0.08 | 2.9e-2 | 8.9e-3 |
| 30-point, nest 2 / nest 3 | 0.5 | 1.1e-2 / 1.1e-2 | 2.2e-3 |
| 30-point, nest 2 / nest 3 | 1 | 3.7e-3 / 3.6e-3 | 8.5e-4 / 1.3e-3 |
| 30-point, nest 2 / nest 3 | 2 | 1.8e-3 / 2.7e-3 | 4.2e-4 / 6.2e-4 |
| 30-point, nest 2 / nest 3 | 4 | 8.8e-4 / 1.3e-3 | 1.2e-4 / 1.7e-4 |

## Design notes

- Horizon: infinite (cycles = 0), where the extent question lives. At T-1
  the paper's h is about 1, so the existing sparse grid already tops out
  past 4h and the whole curve compresses into its flat right end; any
  graduated caption must state the horizon.
- Count-vs-extent confound is controlled by a dedicated arm: raising the
  count from 5 to 30 at the five-point grid's own extent leaves beyond-grid
  error essentially unchanged (2.9e-2 vs 2.1e-2, no improvement) - the
  extent gains are not density gains in disguise.
- Files: `run_grid_extent.py` regenerates `grid_extent.json` (one truth
  solve plus ten arms, about a minute now that solves stop at measured
  convergence); `make_exhibit.py` writes `GridExtentPlot.pdf/.png` directly
  to `content/images/` (single source of truth; no copy is kept here) and
  refuses a JSON whose params stamp does not match the current script.
