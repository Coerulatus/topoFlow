#!/bin/bash

# --- Mamba Initialization ---
# This line loads the mamba functions into the current shell
# This is necessary for 'mamba activate' to work inside the script.
eval "$(mamba shell hook --shell bash)" 

# 1. Create and activate the Mamba environment
mamba env create -f mamba_environment.yml -n topoflow
mamba activate topoflow # This should now work correctly
python -m pip install --upgrade pip uv

# 2. Define PyTorch/CUDA variables
TORCH="2.8.0"
CUDA="cu128"
PYTORCH_INDEX="https://download.pytorch.org/whl/${CUDA}"
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"

# 3. Upgrade and install core PyTorch stack using UV (inside the active Mamba env)
uv pip install --upgrade pip uv
uv pip install torch=="${TORCH}" torchvision torchaudio --extra-index-url "${PYTORCH_INDEX}"

# 4. Install the rest of your project's dependencies
uv pip install -r uv_requirements.txt

# 5. Install the project in editable mode (last step)
uv pip install -e '.[all]'

# 6. Final setup steps
pre-commit install