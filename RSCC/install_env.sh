#!/bin/bash

set -e

echo "=== Step 1: Creating conda environment ==="
conda env create -f ./environment_qwenvl_ft.yaml

echo "=== Step 2: Activating environment ==="
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qwenvl_ft

echo "=== Step 3: Installing flash-attn ==="
pip install flash-attn==2.7.4.post1

echo "=== Installation completed successfully! ==="
echo "To activate the environment, run: conda activate qwenvl_ft"
