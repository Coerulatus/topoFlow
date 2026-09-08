#!/bin/bash

# Define common variables
DATA_DIR="data/CrossDocked/"
BATCH_SIZE=4
MLFLOW_URI="./mlruns"
EXPERIMENT_NAME="TopoFlow_Validation_ToEvaluate"

# List of models to evaluate
MODELS=(
    "checkpoints/to_evaluate/topoflow-epoch=10.ckpt"
    "checkpoints/to_evaluate/topoflow-epoch=20.ckpt"
    "checkpoints/to_evaluate/topoflow-epoch=424.ckpt"
)

for CKPT in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$CKPT" .ckpt)
    echo "Validating model: $MODEL_NAME"
    
    python scripts/validate_flow_model.py \
        --checkpoint_path "$CKPT" \
        --data_dir "$DATA_DIR" \
        --batch_size "$BATCH_SIZE" \
        --mlflow_tracking_uri "$MLFLOW_URI" \
        --experiment_name "$EXPERIMENT_NAME" \
        --run_name "val_$MODEL_NAME"
done
