#!/bin/bash

NUM_COPIES=1
LR="3e-6"
WD="0.9"
PRIOR_SCALE="0.0"
UTD=1
NAME="$(date +%Y%m%d_%H%M%S)"

for arg in "$@"; do
    eval "$arg"
done

ROOT_DIR="/tmp/long_horizon/${NAME}"

echo "Running: num_copies=$NUM_COPIES lr=$LR wd=$WD prior_scale=$PRIOR_SCALE utd=$UTD root_dir=$ROOT_DIR"

python -m alf.bin.train \
    --conf=experiments/long_horizon.py \
    --root_dir="$ROOT_DIR" \
    --conf_param="ConcurrentAlgorithm.num_copies=$NUM_COPIES" \
    --conf_param="create_environment.num_parallel_environments=$NUM_COPIES" \
    --conf_param="_CONFIG._USER.lr=$LR" \
    --conf_param="_CONFIG._USER.wd=$WD" \
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
    --conf_param="_CONFIG._USER.utd=$UTD"
