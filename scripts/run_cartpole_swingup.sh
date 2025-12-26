#!/bin/bash

LR="1e-3"
WD="0.01"
PRIOR_SCALE="0.1"
LN="True"
UTD=1
NUM_AGENTS=1
TAU="0.1"
ASYNC="True"
NAME="$(date +%Y%m%d_%H%M%S)"

for arg in "$@"; do
    eval "$arg"
done

ROOT_DIR="/tmp/cartpole_swingup/${NAME}"

echo "Running: num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE ln=$LN utd=$UTD tau=$TAU async=$ASYNC root_dir=$ROOT_DIR"

python -m alf.bin.train \
    --conf=experiments/cartpole_swingup.py \
    --root_dir="$ROOT_DIR" \
    --conf_param="_CONFIG._USER.lr=$LR" \
    --conf_param="_CONFIG._USER.wd=$WD" \
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
    --conf_param="_CONFIG._USER.ln=$LN" \
    --conf_param="_CONFIG._USER.utd=$UTD" \
    --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS" \
    --conf_param="_CONFIG._USER.tau=$TAU" \
    --conf_param="_CONFIG._USER.async=$ASYNC"
