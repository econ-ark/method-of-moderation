#!/bin/bash
# Full reproduction script for Method of Moderation REMARK
# This script reproduces all results: tests, paper, and computational notebooks

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
echo "Method of Moderation - Full Reproduction"
echo "=========================================="
echo ""
echo "Platform: $(uname -s) ($(uname -m))"
echo "Venv: $(basename "$VENV_PATH")"
echo ""

# Install dependencies
echo "Step 1/5: Installing dependencies..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Run tests to verify installation and code correctness
echo "Step 2/5: Running test suite..."
uv run pytest code/test_moderation.py -v
echo "✓ All tests passed"
echo ""

# Build the paper (HTML and PDF)
echo "Step 3/5: Building paper and PDFs..."
uv run myst build --all --pdf
echo "✓ HTML documentation and PDFs built"
echo ""

# Execute computational notebook
echo "Step 4/5: Executing computational notebook..."
uv run jupyter nbconvert --to notebook --execute --inplace code/method-of-moderation.ipynb
echo "✓ Notebook executed successfully"
echo ""

# Verify outputs
echo "Step 5/5: Verifying outputs..."
if [ -f "_build/html/index.html" ]; then
    echo "✓ HTML documentation: _build/html/index.html"
fi
if [ -f "content/exports/moderation_letters.pdf" ]; then
    echo "✓ Paper PDF: content/exports/moderation_letters.pdf"
fi
if [ -f "content/exports/moderation_with_appendix.pdf" ]; then
    echo "✓ Paper+Appendix PDF: content/exports/moderation_with_appendix.pdf"
fi
if [ -f "code/method-of-moderation.ipynb" ]; then
    echo "✓ Executed notebook: code/method-of-moderation.ipynb"
fi
echo ""

# Word count check
echo "Word count for Economics Letters submission:"
uv run python code/wordcount.py
echo ""

echo "=========================================="
echo "Reproduction complete!"
echo "=========================================="
echo ""
echo "To view results:"
echo "  - Open _build/html/index.html in a browser"
echo "  - Open code/notebook.ipynb in Jupyter"
echo "  - PDFs are in content/exports/"
echo ""
