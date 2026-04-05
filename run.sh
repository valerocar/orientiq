#!/usr/bin/env bash
set -e

ENV_NAME="orientai"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH."
    exit 1
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}' from environment.yml..."
    conda env create -f "$(dirname "$0")/environment.yml"
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install package in editable mode (ensures latest code is picked up)
pip install -e "$(dirname "$0")[dev]" --quiet

# Run the app
cd "$(dirname "$0")"
uvicorn webapp.server:app --reload
