#!/bin/bash
set -e

python scripts/train_flow_model.py \
    --vector_field_model ast_gat \
    --hidden_dim 128 \
    --epochs 1200 \
    --batch_size 8 \
    --samples_per_epoch 512 \
    --start_lr 0.0005 \
    --max_lr 0.0006 \
    --min_lr 0.0001 \
    --w_loss_last_step 0.02 \
    --max_atom_time 0.9 \
    --velocity_scale 0.25 \
    --mlflow_experiment_name TopoFlow_AST_GAT \
    --run_name ast_gat \
    "$@"