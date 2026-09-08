#!/bin/bash

# Define common variables
DATA_DIR="data/CrossDocked/"
OUTPUT_BASE_DIR="checkpoints/to_evaluate/"
TEST_DISTRIBUTION_PATH="data/CrossDocked/processed/val_distribution.npy"
NUM_WORKERS=8 # Adjust based on your CPU/GPU
N_GROUPS=5

echo "Starting bootstrap evaluation for folders in $OUTPUT_BASE_DIR"
echo "Reference distribution: $TEST_DISTRIBUTION_PATH"

python scripts/bootstrap_eval.py \
    --data_dir "$DATA_DIR" \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --test_distribution_path "$TEST_DISTRIBUTION_PATH" \
    --num_workers "$NUM_WORKERS" \
    --n_groups "$N_GROUPS"
