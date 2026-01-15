# Binder Configuration

This directory contains configuration files for [MyBinder.org](https://mybinder.org),
which allows users to launch interactive Jupyter notebooks in the cloud.

## Single Source of Truth

The binder setup draws from the same source as Docker and DevContainer:

| Component | Source |
|-----------|--------|
| Python dependencies | `pyproject.toml` (installed via `uv`) |
| System dependencies | Same as `Dockerfile` (Node.js 18, curl, git, make) |
| Environment setup | Mirrors `reproduce/docker/setup.sh` |

## Files

| File | Purpose |
|------|---------|
| `environment.yml` | Minimal conda env (Python 3.12 + pip) |
| `apt.txt` | System packages (curl, git, make) |
| `postBuild` | Installs Node.js 18, uv, and project dependencies |
| `README.md` | This file |

## How It Works

1. **Binder reads `environment.yml`** → Creates conda environment with Python 3.12
2. **Binder reads `apt.txt`** → Installs system packages via apt-get
3. **Binder runs `postBuild`** → Installs Node.js 18, uv, and all Python dependencies

The `postBuild` script:
- Installs Node.js 18 from nodesource (required by MyST)
- Installs `uv` package manager
- Runs `uv pip install --system -e .` to install from `pyproject.toml`
- Configures `PYTHONPATH` for the `code/` directory
- Warms up matplotlib and numba caches

## Launching on MyBinder

Click the binder badge in the main README, or use this URL:

```
https://mybinder.org/v2/gh/econ-ark/method-of-moderation/main
```

## Testing Locally

To test the binder configuration locally:

```bash
# Option 1: Use Docker (recommended)
docker build -t method-of-moderation .
docker run -it --rm -p 8888:8888 method-of-moderation jupyter lab --ip=0.0.0.0 --no-browser

# Option 2: Use the setup script directly
bash reproduce/docker/setup.sh
source .venv-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)/bin/activate
jupyter lab
```

## Relationship to Other Environments

```
                    ┌─────────────────────┐
                    │   pyproject.toml    │  ← Single Source of Truth
                    │   (dependencies)    │
                    └─────────┬───────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Dockerfile  │   │  devcontainer   │   │     Binder      │
│               │   │     .json       │   │   postBuild     │
└───────┬───────┘   └────────┬────────┘   └────────┬────────┘
        │                    │                     │
        ▼                    ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ reproduce/    │   │ reproduce/      │   │ uv pip install  │
│ docker/       │   │ docker/         │   │ --system -e .   │
│ setup.sh      │   │ setup.sh        │   │                 │
└───────────────┘   └─────────────────┘   └─────────────────┘
```

All three environments install the same dependencies from `pyproject.toml`,
ensuring consistency across local development, CI, and cloud notebooks.
