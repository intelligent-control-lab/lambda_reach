#!/bin/bash
# Installation script for the safety_value package

set -e

echo "Installing safety_value package..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install in development mode
pip install -e "$SCRIPT_DIR"

echo ""
echo "safety_value package installed successfully."
echo ""
echo "Verify installation with:"
echo "  python -c 'from safety_value.safety_analysis_algos.model import build_mlp; print(\"Success!\")'"
