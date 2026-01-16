# Binder Configuration

This directory contains configuration files for [MyBinder.org](https://mybinder.org),
which allows users to launch interactive Jupyter notebooks in the cloud.

## Single Source of Truth

The binder setup draws from the same source as Docker and DevContainer:

| Component           | Source                                             |
| ------------------- | -------------------------------------------------- |
| Python dependencies | `pyproject.toml` + `uv.lock` (installed via `uv`)  |
| System dependencies | Same as `Dockerfile` (Node.js 18, curl, git, make) |
| Environment setup   | Mirrors `reproduce/docker/setup.sh`                |

## Files

| File              | Purpose                                             |
| ----------------- | --------------------------------------------------- |
| `environment.yml` | Conda env (Python 3.12, pip, Node.js 18)            |
| `apt.txt`         | System packages (curl, git, make)                   |
| `postBuild`       | Installs uv and project dependencies                |
| `README.md`       | This file                                           |

## How It Works

1. **Binder reads `environment.yml`** → Creates conda environment with Python 3.12 and Node.js 18
2. **Binder reads `apt.txt`** → Installs system packages via apt-get
3. **Binder runs `postBuild`** → Installs uv and all Python dependencies

The `postBuild` script:

- Installs `uv` package manager
- Exports locked dependencies via `uv export --frozen` for reproducibility
- Installs dependencies into the system Python environment
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
                ┌─────────────────────────┐
                │  pyproject.toml         │  ← Single Source of Truth
                │  uv.lock                │    (dependencies + lock)
                └───────────┬─────────────┘
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
│ reproduce/    │   │ reproduce/      │   │ uv export       │
│ docker/       │   │ docker/         │   │ --frozen        │
│ setup.sh      │   │ setup.sh        │   │ + uv pip install│
└───────────────┘   └─────────────────┘   └─────────────────┘
```

All three environments install the same locked dependencies from `pyproject.toml`
and `uv.lock`, ensuring reproducibility across local development, CI, and cloud
notebooks.
