#!/bin/bash
# Minimal reproduction script for Method of Moderation REMARK
# This script performs a quick validation (<5 minutes)

set -e  # Exit immediately if any command fails

# ============================================================================
# Platform Detection for Platform-Specific Virtual Environment
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect platform and architecture
detect_platform_venv() {
    local platform=""
    local arch=""
    
    # Detect platform
    case "$(uname -s)" in
        Darwin)
            platform="darwin"
            # macOS: Check actual hardware, not Rosetta-reported arch
            if sysctl -n hw.optional.arm64 2>/dev/null | grep -q 1; then
                arch="arm64"
            else
                arch="x86_64"
            fi
            ;;
        Linux)
            platform="linux"
            arch="$(uname -m)"
            # Normalize Linux ARM architecture name
            case "$arch" in
                aarch64) arch="aarch64" ;;
                arm64) arch="aarch64" ;;
                x86_64) arch="x86_64" ;;
            esac
            ;;
        *)
            # Fallback for unknown platforms
            echo "$SCRIPT_DIR/.venv"
            return
            ;;
    esac
    
    echo "$SCRIPT_DIR/.venv-$platform-$arch"
}

# Set platform-specific venv path
VENV_PATH=$(detect_platform_venv)
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

echo "=========================================="
echo "Method of Moderation - Quick Validation"
echo "=========================================="
echo ""
echo "Platform: $(uname -s) ($(uname -m))"
echo "Venv: $(basename "$VENV_PATH")"
echo ""

# Install dependencies
echo "Step 1/3: Installing dependencies..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Run tests
echo "Step 2/3: Running test suite..."
uv run pytest code/test_moderation.py -v --maxfail=3
echo "✓ Tests passed"
echo ""

# Build HTML documentation only (skip PDF and notebook execution)
echo "Step 3/3: Building HTML documentation..."
uv run myst build --html
echo "✓ HTML documentation built"
echo ""

echo "=========================================="
echo "Quick validation complete!"
echo "=========================================="
echo ""
echo "To view results:"
echo "  - Open _build/html/index.html in a browser"
echo ""
echo "For full reproduction including PDFs and executed notebooks, run:"
echo "  ./reproduce.sh"
echo ""
