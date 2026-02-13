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
ALPHA="0"
LN="True"
UTD=1
RESET_PERIOD=1e9
NUM_AGENTS=1
SCALE_BATCH="True"
SCALE_UNROLL="False"
TAU="0.01"
N_CRITICS=1
LENGTH=2
LAMBDA=0
ASYNC="True"
ENV="Rotator"
SHARE_ACTOR="False"
SHARE_CRITIC="False"
NAME="$(date +%Y%m%d_%H%M%S)"
SEEDS=""
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
    --conf_param="_CONFIG._USER.alpha=$ALPHA"
    --conf_param="_CONFIG._USER.ln=$LN"
    --conf_param="_CONFIG._USER.utd=$UTD"
    --conf_param="_CONFIG._USER.reset_period=$RESET_PERIOD"
    --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS"
    --conf_param="_CONFIG._USER.scale_batch=$SCALE_BATCH"
    --conf_param="_CONFIG._USER.scale_unroll=$SCALE_UNROLL"
    --conf_param="_CONFIG._USER.tau=$TAU"
    --conf_param="_CONFIG._USER.n_critics=$N_CRITICS"
    --conf_param="_CONFIG._USER.length=$LENGTH"
    --conf_param="_CONFIG._USER.lambda=$LAMBDA"
    --conf_param="_CONFIG._USER.async=$ASYNC"
    --conf_param="_CONFIG._USER.env='$ENV'"
    --conf_param="_CONFIG._USER.gamma=$GAMMA"
    --conf_param="_CONFIG._USER.share_actor=$SHARE_ACTOR"
    --conf_param="_CONFIG._USER.share_critic=$SHARE_CRITIC"
)

if [ -n "$SEEDS" ] && [ "$SEEDS" -gt 0 ] 2>/dev/null; then
    echo "Running multi-seed batch: num_seeds=$SEEDS num_agents=$NUM_AGENTS scale_batch=$SCALE_BATCH scale_unroll=$SCALE_UNROLL lr=$LR wd=$WD prior_scale=$PRIOR_SCALE alpha=$ALPHA ln=$LN utd=$UTD tau=$TAU n_critics=$N_CRITICS length=$LENGTH lambda=$LAMBDA async=$ASYNC env=$ENV gamma=$GAMMA share_actor=$SHARE_ACTOR share_critic=$SHARE_CRITIC video_record_interval=$VIDEO_RECORD_INTERVAL base_dir=$BASE_DIR"

    for SEED in $(seq 0 $((SEEDS - 1))); do
        ROOT_DIR="${BASE_DIR}/${SEED}"
        LOG_FILE="${ROOT_DIR}/logs.log"
        mkdir -p "$ROOT_DIR"
        echo "Launching seed $SEED in background: root_dir=$ROOT_DIR log=$LOG_FILE"

        (
            if [[ "$ENV" == Rotator* ]]; then
                export ALF_ROTATOR_LOG_DIR="logs/${NAME}/${SEED}"
            fi
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
    echo "Running single run: num_agents=$NUM_AGENTS scale_batch=$SCALE_BATCH scale_unroll=$SCALE_UNROLL lr=$LR wd=$WD prior_scale=$PRIOR_SCALE alpha=$ALPHA ln=$LN utd=$UTD tau=$TAU n_critics=$N_CRITICS length=$LENGTH lambda=$LAMBDA async=$ASYNC env=$ENV gamma=$GAMMA share_actor=$SHARE_ACTOR share_critic=$SHARE_CRITIC video_record_interval=$VIDEO_RECORD_INTERVAL root_dir=$ROOT_DIR"

    if [[ "$ENV" == Rotator* ]]; then
        export ALF_ROTATOR_LOG_DIR="logs/${NAME}/0"
    fi
    python -m alf.bin.train \
        "${COMMON_ARGS[@]}" \
        --root_dir="$ROOT_DIR"
fi

# scripts/run_dmc.sh SEEDS="8" NUM_AGENTS=4 ENV="swimmer:swimmer6" NAME="test4-noentropy-std1em2"
