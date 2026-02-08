#!/bin/bash

# Unified launcher for `experiments/bipolar_conf.py`.
#
# Usage examples:
#   bash scripts/run_bipolar.sh ENV="BipolarChain-medium-dense-onehot-continuous-v0" SEEDS=4
#   bash scripts/run_bipolar.sh NUM_AGENTS=4 ASYNC="False" UTD=8

CONF="experiments/bipolar_conf.py"
ENV="BipolarChain-medium-dense-onehot-discrete-v0"
LR="0.05"
WD="1e-4"
GAMMA="0.9"
PRIOR_SCALE="0.1"
ALPHA="5e-4"
TAU="0.05"
UTD=8
NUM_AGENTS=4
ASYNC="True"
ENTROPY_REWARD="False"
N_COMPONENTS=4000
NAME="$(date +%Y%m%d_%H%M%S)"
SEEDS=""
BASE_DIR=""
EXTRA_ARGS=""

for arg in "$@"; do
    eval "$arg"
done

SAFE_ENV="${ENV//[:\/]/_}"
if [ -z "$BASE_DIR" ]; then
    BASE_DIR="/tmp/bipolar/${SAFE_ENV}/${NAME}"
fi

COMMON_ARGS=(
    --conf="$CONF"
    --conf_param="_CONFIG._USER.env='$ENV'"
    --conf_param="_CONFIG._USER.lr=$LR"
    --conf_param="_CONFIG._USER.wd=$WD"
    --conf_param="_CONFIG._USER.gamma=$GAMMA"
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE"
    --conf_param="_CONFIG._USER.alpha=$ALPHA"
    --conf_param="_CONFIG._USER.tau=$TAU"
    --conf_param="_CONFIG._USER.utd=$UTD"
    --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS"
    --conf_param="_CONFIG._USER.async=$ASYNC"
    --conf_param="_CONFIG._USER.entropy_reward=$ENTROPY_REWARD"
    --conf_param="_CONFIG._USER.n_components=$N_COMPONENTS"
)

if [ -n "$SEEDS" ] && [ "$SEEDS" -gt 0 ] 2>/dev/null; then
    echo "Running multi-seed batch: num_seeds=$SEEDS env=$ENV num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE alpha=$ALPHA tau=$TAU gamma=$GAMMA utd=$UTD async=$ASYNC entropy_reward=$ENTROPY_REWARD n_components=$N_COMPONENTS conf=$CONF base_dir=$BASE_DIR"

    for SEED in $(seq 0 $((SEEDS - 1))); do
        ROOT_DIR="${BASE_DIR}/${SEED}"
        LOG_FILE="${ROOT_DIR}/logs.log"
        mkdir -p "$ROOT_DIR"
        echo "Launching seed $SEED in background: root_dir=$ROOT_DIR log=$LOG_FILE"

        (
            export ALF_BIPOLAR_LOG_DIR="logs/${NAME}/${SEED}"
            python -m alf.bin.train \
                "${COMMON_ARGS[@]}" \
                --root_dir="$ROOT_DIR" \
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
    echo "Running single run: env=$ENV num_agents=$NUM_AGENTS lr=$LR wd=$WD prior_scale=$PRIOR_SCALE alpha=$ALPHA tau=$TAU gamma=$GAMMA utd=$UTD async=$ASYNC entropy_reward=$ENTROPY_REWARD n_components=$N_COMPONENTS seed_version=$SEED_VERSION conf=$CONF root_dir=$ROOT_DIR"

    export ALF_BIPOLAR_LOG_DIR="logs/${NAME}/0"
    python -m alf.bin.train \
        "${COMMON_ARGS[@]}" \
        --root_dir="$ROOT_DIR" \
        $EXTRA_ARGS
fi
