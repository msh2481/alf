#!/bin/bash

# Unified launcher for `experiments/dmc_conf.py`.
#
# Usage examples:
#   bash scripts/run_dmc.sh ENV="cartpole:swingup_sparse" SEEDS=8
#   bash scripts/run_dmc.sh ENV="Rotator" NUM_AGENTS=4 SEEDS=4

LR="1e-3"
WD="1e-5"
GAMMA="0.99"
PRIOR_SCALE="0.01"
LN="True"
UTD=1
NUM_AGENTS=1
TAU="0.01"
ASYNC="False"
ENV="cartpole:swingup_sparse"
NAME="$(date +%Y%m%d_%H%M%S)"
SEEDS="1"
BASE_DIR=""

for arg in "$@"; do
    eval "$arg"
done

SAFE_ENV="${ENV//[:\/]/_}"
if [ -z "$BASE_DIR" ]; then
    BASE_DIR="/tmp/dmc/${SAFE_ENV}/${NAME}"
fi

COMMON_ARGS=(
    --conf=experiments/dmc_conf.py
    --conf_param="_CONFIG._USER.lr=$LR"
    --conf_param="_CONFIG._USER.wd=$WD"
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE"
    --conf_param="_CONFIG._USER.ln=$LN"
    --conf_param="_CONFIG._USER.utd=$UTD"
    --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS"
    --conf_param="_CONFIG._USER.tau=$TAU"
    --conf_param="_CONFIG._USER.async=$ASYNC"
    --conf_param="_CONFIG._USER.env='$ENV'"
    --conf_param="_CONFIG._USER.gamma=$GAMMA"
)

if [ -n "$SEEDS" ] && [ "$SEEDS" -gt 0 ] 2>/dev/null; then
    echo "Running multi-seed batch: num_seeds=$SEEDS num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE ln=$LN utd=$UTD tau=$TAU async=$ASYNC env=$ENV gamma=$GAMMA video_record_interval=$VIDEO_RECORD_INTERVAL base_dir=$BASE_DIR"

    for SEED in $(seq 0 $((SEEDS - 1))); do
        ROOT_DIR="${BASE_DIR}/${SEED}"
        LOG_FILE="${ROOT_DIR}/logs.log"
        mkdir -p "$ROOT_DIR"
        echo "Launching seed $SEED in background: root_dir=$ROOT_DIR log=$LOG_FILE"

        (
            python -m alf.bin.train \
                "${COMMON_ARGS[@]}" \
                --root_dir="$ROOT_DIR" \
                --conf_param="TrainerConfig.random_seed=$SEED" \
                2>&1 | tee "$LOG_FILE"
        ) &
    done

    echo "Waiting for all $SEEDS seed runs to complete..."
    wait
    echo "All seed runs completed!"
else
    ROOT_DIR="$BASE_DIR"
    echo "Running single run: num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE ln=$LN utd=$UTD tau=$TAU async=$ASYNC env=$ENV gamma=$GAMMA video_record_interval=$VIDEO_RECORD_INTERVAL root_dir=$ROOT_DIR"

    python -m alf.bin.train \
        "${COMMON_ARGS[@]}" \
        --root_dir="$ROOT_DIR"
fi
