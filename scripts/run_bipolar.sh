#!/bin/bash

CONF="experiments/bipolar_conf.py"
SEEDS="4"
NUM_AGENTS=4
ASYNC="False"
NAME="$(date +%Y%m%d_%H%M%S)"
EXTRA_ARGS=""

for arg in "$@"; do
    eval "$arg"
done

BASE_DIR="/tmp/bipolar/${NAME}"

if [ -n "$SEEDS" ] && [ "$SEEDS" -gt 0 ] 2>/dev/null; then
    echo "Running multi-seed batch: num_seeds=$SEEDS num_agents=$NUM_AGENTS async=$ASYNC conf=$CONF base_dir=$BASE_DIR"

    for SEED in $(seq 0 $((SEEDS - 1))); do
        ROOT_DIR="${BASE_DIR}/${SEED}"
        LOG_FILE="${ROOT_DIR}/logs.log"
        mkdir -p "$ROOT_DIR"
        echo "Launching seed $SEED in background: root_dir=$ROOT_DIR log=$LOG_FILE"

        (
            python -m alf.bin.train \
                --conf="$CONF" \
                --root_dir="$ROOT_DIR" \
                --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS" \
                --conf_param="_CONFIG._USER.async=$ASYNC" \
                --conf_param="TrainerConfig.random_seed=$SEED" \
                $EXTRA_ARGS \
                2>&1 | tee "$LOG_FILE"
        ) &
    done

    echo "Waiting for all $SEEDS seed runs to complete..."
    wait
    echo "All seed runs completed!"
else
    ROOT_DIR="$BASE_DIR"
    echo "Running single run: num_agents=$NUM_AGENTS async=$ASYNC conf=$CONF root_dir=$ROOT_DIR"

    python -m alf.bin.train \
        --conf="$CONF" \
        --root_dir="$ROOT_DIR" \
        --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS" \
        --conf_param="_CONFIG._USER.async=$ASYNC" \
        $EXTRA_ARGS
fi
