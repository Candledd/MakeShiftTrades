#!/usr/bin/env bash
# setup.sh — bootstrap the virtual environment and install dependencies

set -e

PYTHON=${PYTHON:-python3}

echo "==> Creating virtual environment in ./venv"
$PYTHON -m venv venv

echo "==> Installing dependencies into venv"
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo ""
echo "Done! To activate the environment in your shell, run:"
echo "  source venv/bin/activate   # macOS / Linux"
echo "  venv\\Scripts\\activate      # Windows"
echo ""
echo "Then start the bot with:"
echo "  python main.py"
