#!/bin/bash

# Make sure debugpy is installed: pip install debugpy

python -m debugpy --listen 5678 --wait-for-client scripts/train_flow_model.py \
  --vector_field_model ast_gat \
  --hidden_dim 32 \
  --epochs 1200 \
  --batch_size 2 \
  --start_lr 0.0005 \
  --max_lr 0.0006 \
  --min_lr 0.0001 \
  --w_loss_last_step 0.02 \
  --max_atom_time 0.9 \
  --velocity_scale 0.25 \
  --mlflow_experiment_name debug \
  --run_name ast_gat
