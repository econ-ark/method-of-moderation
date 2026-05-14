#!/bin/bash
# Full reproduction script for Method of Moderation REMARK
# This script reproduces all results: tests, paper, and computational notebooks

set -euo pipefail  # -e: exit on error; -u: error on unset variable; pipefail: detect failures in pipelines

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
# Hard-fail if the core scientific stack didn't actually land in the venv.
uv run python -c "import HARK, numpy, scipy, matplotlib" || {
    echo "❌ Core packages failed to import after 'uv sync'." >&2
    echo "   The reproduction cannot proceed; check uv.lock and Python version." >&2
    exit 1
}
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
VERIFY_FAILED=0
check_output() {
    local file="$1"
    local label="$2"
    if [ -f "$file" ]; then
        echo "✓ ${label}: ${file}"
    else
        echo "✗ MISSING ${label}: ${file}" >&2
        VERIFY_FAILED=1
    fi
}
check_output "_build/html/index.html" "HTML documentation"
check_output "content/exports/moderation_letters.pdf" "Paper PDF"
check_output "content/exports/moderation_with_appendix.pdf" "Paper+Appendix PDF"

# The notebook file is tracked in git, so its mere presence proves nothing.
# Confirm that at least one code cell actually has an execution_count, which is
# the cheapest sign that nbconvert --execute did its job.
if uv run python -c "
import json, sys
nb = json.load(open('code/method-of-moderation.ipynb'))
executed = any(
    c.get('execution_count') for c in nb['cells'] if c.get('cell_type') == 'code'
)
sys.exit(0 if executed else 1)
"; then
    echo "✓ Executed notebook: code/method-of-moderation.ipynb (has populated execution_count)"
else
    echo "✗ Notebook code/method-of-moderation.ipynb has no executed cells." >&2
    VERIFY_FAILED=1
fi

if [ "$VERIFY_FAILED" -ne 0 ]; then
    echo "" >&2
    echo "❌ Verification failed: one or more outputs are missing or unexecuted." >&2
    exit 1
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
echo "  - Open code/method-of-moderation.ipynb in Jupyter"
echo "  - PDFs are in content/exports/"
echo ""
