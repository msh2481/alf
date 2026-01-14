#!/bin/bash

LR="3e-3"
WD="0.01"
PRIOR_SCALE="1e-9"
LN="True"
UTD=1
NUM_AGENTS=1
TAU="0.01"
ASYNC="False"
NAME="$(date +%Y%m%d_%H%M%S)"
SEEDS="8"

for arg in "$@"; do
    eval "$arg"
done

BASE_DIR="/tmp/rotator/${NAME}"

if [ -n "$SEEDS" ] && [ "$SEEDS" -gt 0 ] 2>/dev/null; then
    echo "Running multi-seed batch: num_seeds=$SEEDS num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE ln=$LN utd=$UTD tau=$TAU async=$ASYNC base_dir=$BASE_DIR"
    
    for SEED in $(seq 0 $((SEEDS - 1))); do
        ROOT_DIR="${BASE_DIR}/${SEED}"
        LOG_FILE="${ROOT_DIR}/logs.log"
        mkdir -p "$ROOT_DIR"
        echo "Launching seed $SEED in background: root_dir=$ROOT_DIR log=$LOG_FILE"
        
        (
            python -m alf.bin.train \
                --conf=experiments/rotator_conf.py \
                --root_dir="$ROOT_DIR" \
                --conf_param="_CONFIG._USER.lr=$LR" \
                --conf_param="_CONFIG._USER.wd=$WD" \
                --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
                --conf_param="_CONFIG._USER.ln=$LN" \
                --conf_param="_CONFIG._USER.utd=$UTD" \
                --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS" \
                --conf_param="_CONFIG._USER.tau=$TAU" \
                --conf_param="_CONFIG._USER.async=$ASYNC" \
                --conf_param="TrainerConfig.random_seed=$SEED" \
                2>&1 | tee "$LOG_FILE"
        ) &
    done
    
    echo "Waiting for all $SEEDS seed runs to complete..."
    wait
    echo "All seed runs completed!"
else
    ROOT_DIR="$BASE_DIR"
    echo "Running single run: num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE ln=$LN utd=$UTD tau=$TAU async=$ASYNC root_dir=$ROOT_DIR"
    
    python -m alf.bin.train \
        --conf=experiments/rotator_conf.py \
        --root_dir="$ROOT_DIR" \
        --conf_param="_CONFIG._USER.lr=$LR" \
        --conf_param="_CONFIG._USER.wd=$WD" \
        --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
        --conf_param="_CONFIG._USER.ln=$LN" \
        --conf_param="_CONFIG._USER.utd=$UTD" \
        --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS" \
        --conf_param="_CONFIG._USER.tau=$TAU" \
        --conf_param="_CONFIG._USER.async=$ASYNC"
fi
