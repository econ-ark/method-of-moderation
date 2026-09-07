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
echo "Step 1/4: Installing dependencies..."
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
echo "Step 2/4: Running test suite..."
uv run pytest code/ -v
echo "✓ All tests passed"
echo ""

# Build the paper and execute the notebooks. `--execute` runs every notebook in
# the myst.yml TOC, so the notebooks are built and run in one pass. It must go
# through `uv run`: mystmd spawns a bare `python` for the Jupyter kernel, which
# is not on PATH outside the project environment.
echo "Step 3/4: Building paper and PDFs, executing notebooks..."
uv run myst build --all --pdf --execute
echo "✓ HTML documentation, PDFs and executed notebooks built"
echo ""

# Verify outputs
echo "Step 4/4: Verifying outputs..."
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
check_output "content/exports/moderation_extended.pdf" "Paper PDF (JEDC version)"

# The notebook sources are tracked in git, so their presence proves nothing.
# mystmd writes executed results into the built AST rather than back into the
# source, so check there: an `outputs` node with no children is a cell that was
# parsed but never run, which is what a missing kernel silently produces.
for nb in method-of-moderation method-of-moderation-symbolic; do
    if uv run python -c "
import json, sys
doc = json.load(open('_build/site/content/${nb}.json'))
rendered = 0
def walk(node):
    global rendered
    if isinstance(node, dict):
        if node.get('type') == 'outputs':
            rendered += len(node.get('children') or [])
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(doc)
sys.exit(0 if rendered else 1)
" 2>/dev/null; then
        echo "✓ Executed notebook: code/${nb}.md (built outputs are populated)"
    else
        echo "✗ Notebook code/${nb}.md produced no outputs." >&2
        VERIFY_FAILED=1
    fi
done

if [ "$VERIFY_FAILED" -ne 0 ]; then
    echo "" >&2
    echo "❌ Verification failed: one or more outputs are missing or unexecuted." >&2
    exit 1
fi
echo ""

# Word count. The submission target is JEDC, which has no word limit; the
# Economics Letters version is retained for reference only, so this is
# reported for information rather than checked against a limit.
echo "Word count (Economics Letters reference version):"
uv run python code/wordcount.py
echo ""

echo "=========================================="
echo "Reproduction complete!"
echo "=========================================="
echo ""
echo "To view results:"
echo "  - Open _build/html/index.html in a browser"
echo "  - Open code/method-of-moderation.md (MyST notebook) in JupyterLab or an editor"
echo "  - PDFs are in content/exports/"
echo ""
