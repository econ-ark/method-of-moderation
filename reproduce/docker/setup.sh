#!/bin/bash
# Single Source of Truth for Method of Moderation environment setup
# Used by both Dockerfile and devcontainer.json
#
# Usage:
#   bash setup.sh              # Run with default settings
#   WORKSPACE_DIR=/path bash setup.sh  # Override workspace directory

set -e

echo "🚀 Setting up Method of Moderation development environment..."

# Detect workspace directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

echo "📁 Workspace: $WORKSPACE_DIR"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Failed to install uv"
    exit 1
fi

echo "✓ uv version: $(uv --version)"

# Create/sync virtual environment with all dependency groups
cd "$WORKSPACE_DIR"
echo "🐍 Setting up Python environment..."
uv sync --all-groups

# Verify virtual environment
if [ ! -f "$WORKSPACE_DIR/.venv/bin/activate" ]; then
    echo "❌ Virtual environment not created"
    exit 1
fi

# Configure shell auto-activation for interactive sessions
VENV_ACTIVATE="$WORKSPACE_DIR/.venv/bin/activate"
PYTHONPATH_EXPORT="export PYTHONPATH=\"$WORKSPACE_DIR/code:\$PYTHONPATH\""

for RC_FILE in ~/.bashrc ~/.zshrc; do
    if [ -f "$RC_FILE" ] || [ "$RC_FILE" = ~/.bashrc ]; then
        # Check if already configured
        if ! grep -q "method-of-moderation" "$RC_FILE" 2>/dev/null; then
            {
                echo ""
                echo "# Method of Moderation environment"
                echo "if [ -f \"$VENV_ACTIVATE\" ]; then source \"$VENV_ACTIVATE\"; fi"
                echo "$PYTHONPATH_EXPORT"
            } >> "$RC_FILE"
            echo "✓ Updated $RC_FILE"
        fi
    fi
done

echo ""
echo "✅ Environment setup complete!"
echo "   Python: $(uv run python --version)"
echo "   Workspace: $WORKSPACE_DIR"
echo ""
echo "To activate manually: source $VENV_ACTIVATE"
