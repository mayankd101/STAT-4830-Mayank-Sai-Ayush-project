#!/bin/bash
# Setup script for ViT training pipeline
# Run from project root: ./setup_env.sh

set -e
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating and installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install ipykernel

echo "Registering as Jupyter kernel..."
python -m ipykernel install --user --name=stat4830-vit --display-name="STAT4830 ViT (Python 3)"

echo ""
echo "Done! In your notebook, select kernel: 'STAT4830 ViT (Python 3)'"
