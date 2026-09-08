#!/bin/bash

# Define common variables
PDB_DIR="data/CrossDocked/processed/val_small"
MOL_PER_POCKET=1
BATCH_SIZE=16
N_STEPS=80

# List of models to evaluate
MODELS=(
    "checkpoints/to_evaluate/topoflow-epoch=10.ckpt"
    "checkpoints/to_evaluate/topoflow-epoch=20.ckpt"
    "checkpoints/to_evaluate/topoflow-epoch=424.ckpt"
)

for CKPT in "${MODELS[@]}"; do
    # Extract model name without extension for the output directory
    MODEL_NAME=$(basename "$CKPT" .ckpt)
    OUTPUT_PATH="checkpoints/to_evaluate/$MODEL_NAME"
    
    echo "Processing model: $MODEL_NAME"
    echo "Output path: $OUTPUT_PATH"
    
    python scripts/sample_for_pockets.py \
        --checkpoint_path "$CKPT" \
        --pdb_dir "$PDB_DIR" \
        --output_path "$OUTPUT_PATH" \
        --mol_per_pocket "$MOL_PER_POCKET" \
        --batch_size "$BATCH_SIZE" \
        --n_steps "$N_STEPS"
done
