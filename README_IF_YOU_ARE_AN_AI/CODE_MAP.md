# Code Map: Repository Structure

This document explains what each file in the repository does.

## Core Code (`code/`)

### `code/moderation.py`
**Purpose**: Core implementation of the Method of Moderation

**Key classes**:
- `IndShockEGMConsumerType`: Standard endogenous gridpoints method (baseline)
- `IndShockMoMConsumerType`: Method of Moderation implementation

**Key functions**:
- `endogenous_grid_method()`: Standard EGM solver
- `method_of_moderation()`: MoM solver
- `prepare_to_solve()`: Calculate shared economic parameters
- `make_behavioral_bounds()`: Create optimist/pessimist solutions
- `solve_egm_step()`: Core EGM calculation
- `construct_value_functions()`: Build value functions

**Mathematical utilities**:
- `log_mnrm_ex()`: μ = log(m - m_min)
- `moderate()`: ω = (c - c_pes)/(c_opt - c_pes)
- `logit_moderate()`: χ = log(ω/(1-ω))
- `expit_moderate()`: ω = 1/(1 + exp(-χ))

### `code/plotting.py`
**Purpose**: Visualization functions for all figures

**Key functions**:
- `plot_moderation_ratio()`: Figure showing ω(μ)
- `plot_logit_function()`: Figure showing χ(μ)
- `plot_consumption_bounds()`: Optimist/pessimist/realist comparison
- `plot_precautionary_gaps()`: Approximation quality
- `plot_mpc_bounds()`: MPC bounded by theory
- `plot_value_functions()`: Value function approximations
- `plot_cusp_point()`: Cusp point visualization

### `code/style.py`
**Purpose**: Visual styling constants and configuration

**Contents**:
- Color definitions for economic concepts
- Line styles and widths
- Font sizes
- Matplotlib theme configuration

### `code/method-of-moderation.ipynb`
**Purpose**: Illustrative Jupyter notebook demonstrating the method

**Structure**:
- 32 cells (15 code, 16 markdown)
- Generates 13 figures
- Progresses through: Model → Problem → Solution → Extensions

### `code/test_moderation.py`
**Purpose**: Test suite (11 tests)

**Tests**:
- `test_all_consumer_types_solve`: Solutions exist
- `test_consumption_values`: Consumption accuracy
- `test_mpc_accuracy`: MPC bounds
- `test_solution_attributes`: Required attributes
- `test_consumption_bounds`: c_pes < c < c_opt
- `test_value_function`: Value function properties
- `test_moderation_ratio_formula`: ω calculation
- `test_cusp_point_formula`: Cusp point
- `test_mpc_bounds_everywhere`: MPC always valid
- `test_hermite_slope_formulas`: Derivative accuracy
- `test_stochastic_mpc_formula`: Stochastic extension

### `code/wordcount.py`
**Purpose**: Word count for journal submission

---

## Paper Content (`content/`)

### `content/paper/moderation_letters.md`
**Purpose**: Economics Letters submission format (main paper)

### `content/paper/moderation.md`
**Purpose**: Full paper with complete derivations

### `content/paper/appendix_letters.md`
**Purpose**: Appendix with proofs and extensions

### `content/paper/abstract.md`
**Purpose**: Paper abstract (included in other files)

### `content/references.bib`
**Purpose**: BibTeX bibliography

### `content/macros.yml`
**Purpose**: LaTeX macro definitions for MyST

### `content/images/`
**Purpose**: Static figures for the paper
- `ExtrapProblemPlot.{png,pdf}`: Extrapolation problem illustration
- `ExtrapProblemSolvedPlot.{png,pdf}`: MoM solution
- `IntExpFOCInvPesReaOptNeed*.{png,pdf}`: Consumption function comparisons

### `content/exports/`
**Purpose**: Generated PDF outputs
- `moderation_letters.pdf`: Letters version
- `moderation_with_appendix.pdf`: Full paper + appendix
- `appendix_letters.pdf`: Appendix only

---

## Reproduction Infrastructure

### `reproduce.sh`
**Purpose**: Full reproduction script

**Steps**:
1. Install dependencies (`uv sync`)
2. Run tests (`pytest`)
3. Build paper and PDFs (`myst build --all --pdf`)
4. Execute notebook (`jupyter nbconvert --execute`)
5. Verify outputs

### `reproduce_min.sh`
**Purpose**: Quick validation (<5 minutes)

**Steps**:
1. Install dependencies
2. Run tests
3. Build HTML only

### `README_IF_YOU_ARE_AN_AI/benchmarks/`
**Purpose**: Benchmarking infrastructure

**Files**:
- `benchmark.sh`: Wrapper script
- `capture_system_info.py`: System info capture
- `schema.json`: Benchmark format schema

---

## Configuration Files

### `pyproject.toml`
**Purpose**: Python project configuration

**Key sections**:
- `[project]`: Name, version, dependencies
- `[dependency-groups]`: Test and dev dependencies
- `[tool.ruff]`: Linting configuration
- `[tool.uv.sources]`: Use HARK from git

### `myst.yml`
**Purpose**: MyST documentation configuration

**Key sections**:
- `project`: Metadata, authors, affiliations
- `site`: Build configuration

### `CITATION.cff`
**Purpose**: Citation metadata (CFF format)

### `.pre-commit-config.yaml`
**Purpose**: Pre-commit hooks for code quality

### `.github/workflows/ci.yml`
**Purpose**: GitHub Actions CI configuration

---

## Binder Configuration (`binder/`)

### `binder/postBuild`
**Purpose**: Setup script for MyBinder.org

### `binder/environment.yml`
**Purpose**: Conda environment for Binder

### `binder/apt.txt`
**Purpose**: System packages for Binder

---

## Docker/DevContainer

### `Dockerfile`
**Purpose**: Docker image definition

### `.devcontainer/devcontainer.json`
**Purpose**: VS Code DevContainer configuration

### `reproduce/docker/setup.sh`
**Purpose**: Shared setup script for Docker/DevContainer
