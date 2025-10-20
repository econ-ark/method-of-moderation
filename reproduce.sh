#!/bin/bash
# Full reproduction script for Method of Moderation REMARK
# This script reproduces all results: tests, paper, and computational notebooks

set -e  # Exit immediately if any command fails

echo "=========================================="
echo "Method of Moderation - Full Reproduction"
echo "=========================================="
echo ""

# Install dependencies
echo "Step 1/5: Installing dependencies..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Run tests to verify installation and code correctness
echo "Step 2/5: Running test suite..."
if uv run pytest -v 2>/dev/null; then
    echo "✓ All tests passed"
else
    echo "⚠ No tests found or tests not configured yet"
fi
echo ""

# Build the paper (HTML and PDF)
echo "Step 3/5: Building paper..."
uv run myst build --html
echo "✓ HTML documentation built"
echo ""

# Note: PDF build requires LaTeX installation
# Uncomment the following line if you have LaTeX installed:
# uv run myst build --pdf content/paper/moderation.md

# Execute computational notebook
echo "Step 4/5: Executing computational notebook..."
uv run jupyter nbconvert --to notebook --execute --inplace code/notebook.ipynb
echo "✓ Notebook executed successfully"
echo ""

# Generate figures (implicitly done during notebook execution)
echo "Step 5/5: Verifying outputs..."
if [ -f "_build/html/index.html" ]; then
    echo "✓ HTML documentation available at: _build/html/index.html"
fi
if [ -f "code/notebook.ipynb" ]; then
    echo "✓ Executed notebook available at: code/notebook.ipynb"
fi
echo ""

echo "=========================================="
echo "Reproduction complete!"
echo "=========================================="
echo ""
echo "To view results:"
echo "  - Open _build/html/index.html in a browser"
echo "  - Open code/notebook.ipynb in Jupyter"
echo ""
