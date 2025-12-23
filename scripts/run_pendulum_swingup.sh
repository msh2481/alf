#!/bin/bash

NUM_COPIES=1
LR="3e-3"
WD="0"
PRIOR_SCALE="0.1"
FRAME_SKIP=8
UTD=1
NAME="$(date +%Y%m%d_%H%M%S)"

for arg in "$@"; do
    eval "$arg"
done

ROOT_DIR="/tmp/pendulum_swingup/${NAME}"

echo "Running: num_copies=$NUM_COPIES lr=$LR wd=$WD prior_scale=$PRIOR_SCALE frame_skip=$FRAME_SKIP utd=$UTD root_dir=$ROOT_DIR"

python -m alf.bin.train \
    --conf=experiments/pendulum_swingup.py \
    --root_dir="$ROOT_DIR" \
    --conf_param="ConcurrentAlgorithm.num_copies=$NUM_COPIES" \
    --conf_param="create_environment.num_parallel_environments=$NUM_COPIES" \
    --conf_param="_CONFIG._USER.lr=$LR" \
    --conf_param="_CONFIG._USER.wd=$WD" \
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
    --conf_param="_CONFIG._USER.frame_skip=$FRAME_SKIP" \
    --conf_param="_CONFIG._USER.utd=$UTD"
