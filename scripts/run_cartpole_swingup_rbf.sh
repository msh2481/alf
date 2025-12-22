#!/bin/bash

NUM_COPIES=1
LR="0.05"
WD="1e-4"
GAMMA="2.0"
N_COMPONENTS=1000
PRIOR_SCALE="0.1"
NAME="$(date +%Y%m%d_%H%M%S)"

for arg in "$@"; do
    eval "$arg"
done

ROOT_DIR="/tmp/cartpole_swingup_rbf/${NAME}"

echo "Running RBF: num_copies=$NUM_COPIES lr=$LR wd=$WD gamma=$GAMMA n_components=$N_COMPONENTS prior_scale=$PRIOR_SCALE"
echo "root_dir=$ROOT_DIR"

python -m alf.bin.train \
    --conf=experiments/cartpole_swingup_rbf.py \
    --root_dir="$ROOT_DIR" \
    --conf_param="ConcurrentAlgorithm.num_copies=$NUM_COPIES" \
    --conf_param="create_environment.num_parallel_environments=$NUM_COPIES" \
    --conf_param="_CONFIG._USER.lr=$LR" \
    --conf_param="_CONFIG._USER.wd=$WD" \
    --conf_param="_CONFIG._USER.gamma=$GAMMA" \
    --conf_param="_CONFIG._USER.n_components=$N_COMPONENTS" \
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE"
