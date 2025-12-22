#!/bin/bash

NUM_COPIES=1
LR="3e-4"
WD="0"
NAME="$(date +%Y%m%d_%H%M%S)"

for arg in "$@"; do
    eval "$arg"
done

ROOT_DIR="/tmp/cartpole_swingup/${NAME}"

echo "Running: num_copies=$NUM_COPIES lr=$LR wd=$WD root_dir=$ROOT_DIR"

python -m alf.bin.train \
    --conf=experiments/cartpole_swingup.py \
    --root_dir="$ROOT_DIR" \
    --conf_param="ConcurrentAlgorithm.num_copies=$NUM_COPIES" \
    --conf_param="create_environment.num_parallel_environments=$NUM_COPIES" \
    --conf_param="ConcurrentAlgorithm.optimizer=alf.optimizers.Adam(lr=$LR, weight_decay=$WD, name='main')"
