#!/bin/bash
set -e

# Train all TopoFlow models reported in the paper sequentially.
# Any extra CLI flags passed to this script (e.g. --sample_every 0) are forwarded to each run.

echo "============================================================"
echo "1/4 Training Transformer (experiments/transformer.bash)"
echo "============================================================"
bash experiments/transformer.bash "$@"

echo "============================================================"
echo "2/4 Training GAT (experiments/gat.bash)"
echo "============================================================"
bash experiments/gat.bash "$@"

echo "============================================================"
echo "3/4 Training AST (experiments/ast.bash)"
echo "============================================================"
bash experiments/ast.bash "$@"

echo "============================================================"
echo "4/4 Training Ensemble GAT+AST (experiments/ast_gat.bash)"
echo "============================================================"
bash experiments/ast_gat.bash "$@"
