#!/bin/sh
export PYTHONPATH="/home/marco/Documents/phd/topoFlow/proven_model"
export CUDA_VISIBLE_DEVICES=""

python -m flowr.train \
    --gpus 0 \
    --d_model 32 \
    --d_edge 16 \
    --n_coord_sets 4 \
    --pocket_d_model 32 \
    --pocket_n_layers 1 \
    --arch ast \
    --pocket_noise fix \
    --epochs 1 \
    --val_check_epochs 1 \
    --self_condition \
    --max_atoms_pocket 600 \
    --dataset "crossdocked" \
    --exp_name "debug_run" \
    --data_path "/home/marco/Documents/phd/topoFlow/data/CrossDocked/processed_flowr" \
    --save_dir "/home/marco/Documents/phd/topoFlow/experiments/proven_model_logs" \
    --use_ema \
    --ema_decay 0.999 \
    --lr 5.0e-4 \
    --lr_schedule exponential \
    --lr_gamma 0.998 \
    --use_lig_pocket_rbf \
    --remove_hs \
    --optimal_transport equivariant \
    --categorical_strategy uniform-sample \
    --batch_cost 1 \
    --val_batch_cost 1
