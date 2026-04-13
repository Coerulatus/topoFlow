# TopoFlow  

## Installation Guide

This guide details how to set up the execution environment and install the necessary dependencies for the `topoFlow` project.

### Prerequisites

Before starting the installation, you must have the following tools installed and configured:

1.  **Git:** For cloning the repository.
2.  **Mamba/Conda:** The project uses **Mamba** for environment management. Follow the installation instructions for [Miniforge](https://github.com/conda-forge/miniforge) or your preferred Conda distribution.
3.  **NVIDIA CUDA Drivers:** Ensure you have the correct NVIDIA drivers installed for your GPU.

### Setup

#### 1\. Clone the Repository

Clone the project repository and navigate into the directory:

```bash
git clone https://github.com/Coerulatus/topoFlow.git
cd topoFlow
```

#### 2\. Configure CUDA and PyTorch Versions

The installation process is managed by the `env_setup.sh` script, which uses a hybrid approach (`mamba` for core dependencies, `uv` for speed).

**Action Required:** Before running the script, review and update the `TORCH` and `CUDA` variables at the top of the `env_setup.sh` file to ensure they match your system's hardware and drivers.

  * `TORCH`: Specifies the target PyTorch version (e.g., `2.8.0`).
  * `CUDA`: Specifies the CUDA version matching your PyTorch build (e.g., `cu128`).

#### 3\. Execute the Setup Script

Run the main setup script. This script will create the isolated Mamba environment, install core dependencies, and use `uv` to rapidly install the remaining Python packages, including the PyTorch and PyTorch Geometric stack.

```bash
bash env_setup.sh
```

-----

## Running the Project

### MLFlow Tracking

To view and manage experiment results using MLFlow, run the UI backend and access the provided address:

```bash
mlflow ui --backend-store-uri ./mlruns
```

The terminal will display the local tracking server address (e.g., `Listening at: http://127.0.0.1:5000`). Use this link to view your experiments in a web browser.

-----

## Troubleshooting

### PyTorch Geometric GPU Errors

If you encounter errors related to missing GPU implementations (e.g., for `torch_geometric.nn.radius` or `torch_geometric.nn.graph_radius`), it indicates a potential failure during the PyTorch Geometric auxiliary package compilation.

You can manually resolve this by running the following command **after activating your environment**, ensuring the correct torch and CUDA versions are used:

```bash
# This command forces the installation from the dedicated PyG wheel index
pip install torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
```

### Use GNINA

Download gnina executable
```
wget https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2
```

Create gnina_path file
```
nano topoflow/utils/gnina_path.py
```
which should contain: GNINA_EXECUTABLE = "/path/to/your/gnina.1.3.2"