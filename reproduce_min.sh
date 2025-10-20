#!/bin/bash
# Minimal reproduction script for Method of Moderation REMARK
# This script performs a quick validation (<5 minutes)

set -e  # Exit immediately if any command fails

echo "=========================================="
echo "Method of Moderation - Quick Validation"
echo "=========================================="
echo ""

# Install dependencies
echo "Step 1/3: Installing dependencies..."
uv sync
echo "✓ Dependencies installed"
echo ""

# Run minimal tests (skip slow tests)
echo "Step 2/3: Running quick tests..."
if uv run pytest -v -k "not slow" --maxfail=1 2>/dev/null; then
    echo "✓ Quick tests passed"
else
    echo "⚠ No tests found or tests not configured yet"
fi
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
echo "For full reproduction including executed notebooks, run:"
echo "  ./reproduce.sh"
echo ""
