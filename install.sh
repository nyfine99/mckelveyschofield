#!/bin/bash

echo "========================================"
echo "McKelvey-Schofield Project Setup"
echo "========================================"
echo

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "Creating output directory..."
mkdir -p output

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "To run a demo, use:"
echo "  python -m scripts.euclidean_electorate"
echo
echo "For more examples, see QUICK_START.md"
echo

# Make the script executable
chmod +x install.sh 