#!/bin/bash
set -e
# --- Mamba Initialization ---
# This line loads the mamba functions into the current shell
eval "$(mamba shell hook --shell bash)" 

# 1. Update the Mamba environment
mamba env update -f mamba_environment.yml -n topoflow
mamba activate topoflow

# 2. Update dependencies using UV
uv pip install -r uv_requirements.txt

# 3. Install the project in editable mode
uv pip install -e '.[all]'

echo "Environment successfully updated!"
