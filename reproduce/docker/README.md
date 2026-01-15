# Docker and DevContainer Setup

This directory contains the shared setup script for both Docker and DevContainer
environments. The goal is to maintain a **Single Source of Truth (SST)** for
environment configuration.

## Architecture

```
method-of-moderation/
├── Dockerfile                    # Standalone Docker build
├── .devcontainer/
│   └── devcontainer.json         # VS Code/Cursor DevContainer config
└── reproduce/
    └── docker/
        ├── setup.sh              # SST: Shared environment setup
        └── README.md             # This file
```

## Platform-Specific Virtual Environments

The setup creates **architecture-specific virtual environments** to enable
seamless switching between different platforms without rebuilding venvs:

| Platform | Architecture | Venv Directory |
|----------|--------------|----------------|
| macOS | Apple Silicon | `.venv-darwin-arm64` |
| macOS | Intel | `.venv-darwin-x86_64` |
| Linux | ARM64 | `.venv-linux-aarch64` |
| Linux | x86_64 | `.venv-linux-x86_64` |

This prevents architecture mismatches when switching between local development
(macOS) and DevContainer (Linux).

## Quick Start

### Option 1: DevContainer (Recommended for Development)

1. Open this repository in VS Code or Cursor
2. When prompted, click "Reopen in Container"
3. Wait for the container to build and setup to complete
4. You're ready to develop!

### Option 2: Docker (Standalone)

```bash
# Build the image
docker build -t method-of-moderation .

# Run interactively
docker run -it --rm method-of-moderation

# Run with local files mounted (for development)
docker run -it --rm -v "$(pwd):/workspace" method-of-moderation

# Run Jupyter Lab
docker run -it --rm -p 8888:8888 method-of-moderation \
    jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

### Option 3: Local Setup (No Docker)

```bash
# Run the setup script directly
bash reproduce/docker/setup.sh

# The script auto-detects your platform and creates the appropriate venv
# For example, on macOS Apple Silicon: .venv-darwin-arm64

# Activate the environment (example for macOS ARM)
source .venv-darwin-arm64/bin/activate
export PYTHONPATH="$(pwd)/code:$PYTHONPATH"
```

## Environment Details

The setup script (`setup.sh`) performs these steps:

1. **Detects platform and architecture** - Creates appropriate venv name
2. **Installs `uv`** - The fast Python package manager
3. **Creates virtual environment** - Using `uv sync --all-groups`
4. **Configures shell** - Auto-activates venv in bash/zsh

### Exposed Ports

| Port | Service |
|------|---------|
| 8888 | Jupyter Lab |
| 8866 | Voilà Dashboard |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONPATH` | `/workspace/code` | Module import path |
| `PYTHONUNBUFFERED` | `1` | Real-time output |

## Customization

To modify the environment setup:

1. Edit `reproduce/docker/setup.sh`
2. Changes automatically apply to both Docker and DevContainer builds

## Troubleshooting

### "Module not found" errors

Ensure `PYTHONPATH` includes the `code/` directory:

```bash
export PYTHONPATH="$(pwd)/code:$PYTHONPATH"
```

### Finding your venv

The venv is named based on your platform and architecture:

```bash
# Check which venvs exist
ls -la .venv-*

# Activate the correct one for your platform
source .venv-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)/bin/activate
```

### Permission issues in Docker

The container runs as user `vscode` (UID 1000). If you have permission issues
with mounted volumes, ensure your local files are owned by UID 1000 or run:

```bash
docker run -it --rm -u "$(id -u):$(id -g)" -v "$(pwd):/workspace" method-of-moderation
```

### Rebuilding from scratch

```bash
# Docker
docker build --no-cache -t method-of-moderation .

# DevContainer
# Use "Dev Containers: Rebuild Container" from command palette

# Local: remove venv and re-run setup
rm -rf .venv-*
bash reproduce/docker/setup.sh
```
