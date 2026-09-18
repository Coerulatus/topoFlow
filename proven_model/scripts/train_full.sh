#!/bin/sh
export PYTHONPATH="$(pwd)"

python -m flowr.train \
    --gpus 1 \
    --d_model 128 \
    --d_edge 32 \
    --n_coord_sets 4 \
    --pocket_d_model 128 \
    --pocket_n_layers 3 \
    --arch ast \
    --pocket_noise fix \
    --epochs 50 \
    --val_check_epochs 10 \
    --self_condition \
    --max_atoms_pocket 600 \
    --dataset "crossdocked" \
    --exp_name "full_training_ast" \
    --data_path "../data/CrossDocked/processed_flowr" \
    --save_dir "../experiments/proven_model_logs" \
    --use_ema \
    --ema_decay 0.999 \
    --lr 5.0e-4 \
    --lr_schedule exponential \
    --lr_gamma 0.998 \
    --use_lig_pocket_rbf \
    --remove_hs \
    # --optimal_transport equivariant \
    --categorical_strategy uniform-sample \
    --batch_cost 2 \
    --val_batch_cost 2 \
    --use_bucket_sampler \
    --acc_batches 16
