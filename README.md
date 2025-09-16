## Installation guide

Create conda environment:
```
conda env create -f environment.yml
```
Activate it:
```
conda activate topoflow
```

Install pre-commit hook:
```
pre-commit install
```

Install package in editable mode:
```
pip install -e .
```

### If there are problems with torch_geometric.nn.radius or torch_geometric.nn.graph_radius not having GPU versions
```
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

### To remember
Run MLFLow. Then look for something like `Listening at: http://127.0.0.1:5000` and CTRL-click the address.
```
mlflow ui --backend-store-uri ./mlruns
```