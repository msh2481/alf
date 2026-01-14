#!/bin/bash

NUM_COPIES=1
LR="1e-3"
WD="0.1"
PRIOR_SCALE="0.0"
UTD=2
NAME="$(date +%Y%m%d_%H%M%S)"
SEEDS="5"

for arg in "$@"; do
    eval "$arg"
done

BASE_DIR="/tmp/long_horizon/${NAME}"

if [ -n "$SEEDS" ] && [ "$SEEDS" -gt 0 ] 2>/dev/null; then
    echo "Running multi-seed batch: num_seeds=$SEEDS num_copies=$NUM_COPIES lr=$LR wd=$WD prior_scale=$PRIOR_SCALE utd=$UTD base_dir=$BASE_DIR"

    for SEED in $(seq 0 $((SEEDS - 1))); do
        ROOT_DIR="${BASE_DIR}/${SEED}"
        LOG_FILE="${ROOT_DIR}/logs.log"
        mkdir -p "$ROOT_DIR"
        echo "Launching seed $SEED in background: root_dir=$ROOT_DIR log=$LOG_FILE"

        (
            python -m alf.bin.train \
                --conf=experiments/long_horizon.py \
                --root_dir="$ROOT_DIR" \
                --conf_param="ConcurrentAlgorithm.num_copies=$NUM_COPIES" \
                --conf_param="create_environment.num_parallel_environments=$NUM_COPIES" \
                --conf_param="_CONFIG._USER.lr=$LR" \
                --conf_param="_CONFIG._USER.wd=$WD" \
                --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
                --conf_param="_CONFIG._USER.utd=$UTD" \
                --conf_param="TrainerConfig.random_seed=$SEED" \
                2>&1 | tee "$LOG_FILE"
        ) &
    done

    echo "Waiting for all $SEEDS seed runs to complete..."
    wait
    echo "All seed runs completed!"
else
    ROOT_DIR="$BASE_DIR"
    echo "Running single run: num_copies=$NUM_COPIES lr=$LR wd=$WD prior_scale=$PRIOR_SCALE utd=$UTD root_dir=$ROOT_DIR"

    python -m alf.bin.train \
        --conf=experiments/long_horizon.py \
        --root_dir="$ROOT_DIR" \
        --conf_param="ConcurrentAlgorithm.num_copies=$NUM_COPIES" \
        --conf_param="create_environment.num_parallel_environments=$NUM_COPIES" \
        --conf_param="_CONFIG._USER.lr=$LR" \
        --conf_param="_CONFIG._USER.wd=$WD" \
        --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE" \
        --conf_param="_CONFIG._USER.utd=$UTD"
fi
