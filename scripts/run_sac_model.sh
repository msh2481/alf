#!/bin/bash

# Launcher for sac_model_conf.py — compares SacAlgorithm, SacVAlgorithm, SacGradAlgorithm.
#
# Usage examples:
#   bash scripts/run_sac_model.sh ALGO="sac" ENV="cartpole:swingup_sparse" SEEDS=8
#   bash scripts/run_sac_model.sh ALGO="sac_v" ENV="Rotator" SEEDS=4
#   bash scripts/run_sac_model.sh ALGO="sac_grad" MODEL_LOSS_WEIGHT=0.5 SEEDS=8

CONF="experiments/sac_model_conf.py"
LR="1e-3"
WD="1e-5"
GAMMA="0.99"
PRIOR_SCALE="0.01"
ALPHA="0"
LN="True"
UTD=1
RESET_PERIOD=1e9
NUM_AGENTS=1
NUM_ENVS=32
SCALE_BATCH="True"
UNROLL_LENGTH="0.25"
SHUFFLE="False"
TAU="0.01"
N_CRITICS=1
ASYNC="True"
ENV="Rotator"
SHARE_ACTOR="False"
SHARE_CRITIC="False"
SHARED_CRITIC_MODE="first"
OWN_ROLLOUT_FRACTION="-1"
NUM_LAYERS=2
ALGO="sac"
MODEL_LOSS_WEIGHT="1.0"
GRAD_SYNC_WEIGHT="1.0"
DYNAMICS_HIDDEN="(256,256)"
REWARD_HIDDEN="(256,256)"
NAME="$(date +%Y%m%d_%H%M%S)"
SEEDS=""
BASE_DIR=""

for arg in "$@"; do
    key="${arg%%=*}"
    val="${arg#*=}"
    printf -v "$key" '%s' "$val"
done

SAFE_ENV="${ENV//[:\/]/_}"
if [ -z "$BASE_DIR" ]; then
    BASE_DIR="/tmp/dmc/${SAFE_ENV}/${NAME}"
fi

SEED_START=0
SEED_END=0
SEED_COUNT=0
if [ -n "$SEEDS" ]; then
    if [[ "$SEEDS" =~ ^[0-9]+$ ]]; then
        SEED_START=0
        SEED_END="$SEEDS"
    elif [[ "$SEEDS" =~ ^[0-9]+,[0-9]+$ ]]; then
        IFS=',' read -r SEED_START SEED_END <<< "$SEEDS"
    else
        echo "Invalid SEEDS='$SEEDS'. Use N or L,R (R excluded)." >&2
        exit 1
    fi

    if [ "$SEED_END" -lt "$SEED_START" ]; then
        echo "Invalid SEEDS='$SEEDS': require R >= L in L,R." >&2
        exit 1
    fi
    SEED_COUNT=$((SEED_END - SEED_START))
fi

COMMON_ARGS=(
    --conf="$CONF"
    --conf_param="_CONFIG._USER.lr=$LR"
    --conf_param="_CONFIG._USER.wd=$WD"
    --conf_param="_CONFIG._USER.prior_scale=$PRIOR_SCALE"
    --conf_param="_CONFIG._USER.alpha=$ALPHA"
    --conf_param="_CONFIG._USER.ln=$LN"
    --conf_param="_CONFIG._USER.utd=$UTD"
    --conf_param="_CONFIG._USER.reset_period=$RESET_PERIOD"
    --conf_param="_CONFIG._USER.num_agents=$NUM_AGENTS"
    --conf_param="_CONFIG._USER.num_envs=$NUM_ENVS"
    --conf_param="_CONFIG._USER.scale_batch=$SCALE_BATCH"
    --conf_param="_CONFIG._USER.unroll_length=$UNROLL_LENGTH"
    --conf_param="_CONFIG._USER.shuffle=$SHUFFLE"
    --conf_param="_CONFIG._USER.tau=$TAU"
    --conf_param="_CONFIG._USER.n_critics=$N_CRITICS"
    --conf_param="_CONFIG._USER.async=$ASYNC"
    --conf_param="_CONFIG._USER.env='$ENV'"
    --conf_param="_CONFIG._USER.gamma=$GAMMA"
    --conf_param="_CONFIG._USER.share_actor=$SHARE_ACTOR"
    --conf_param="_CONFIG._USER.share_critic=$SHARE_CRITIC"
    --conf_param="_CONFIG._USER.shared_critic_mode='$SHARED_CRITIC_MODE'"
    --conf_param="_CONFIG._USER.own_rollout_fraction=$OWN_ROLLOUT_FRACTION"
    --conf_param="_CONFIG._USER.num_layers=$NUM_LAYERS"
    --conf_param="_CONFIG._USER.algo='$ALGO'"
    --conf_param="_CONFIG._USER.model_loss_weight=$MODEL_LOSS_WEIGHT"
    --conf_param="_CONFIG._USER.grad_sync_weight=$GRAD_SYNC_WEIGHT"
    --conf_param="_CONFIG._USER.dynamics_hidden=$DYNAMICS_HIDDEN"
    --conf_param="_CONFIG._USER.reward_hidden=$REWARD_HIDDEN"
)

if [ "$SEED_COUNT" -gt 0 ]; then
    echo "Running multi-seed batch: conf=$CONF algo=$ALGO seeds=$SEED_START,$SEED_END count=$SEED_COUNT env=$ENV model_loss_weight=$MODEL_LOSS_WEIGHT grad_sync_weight=$GRAD_SYNC_WEIGHT base_dir=$BASE_DIR"

    for SEED in $(seq "$SEED_START" $((SEED_END - 1))); do
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

    echo "Waiting for all $SEED_COUNT seed runs to complete..."
    wait
    echo "All seed runs completed!"
else
    ROOT_DIR="$BASE_DIR"
    echo "Running single run: conf=$CONF algo=$ALGO env=$ENV model_loss_weight=$MODEL_LOSS_WEIGHT grad_sync_weight=$GRAD_SYNC_WEIGHT root_dir=$ROOT_DIR"

    if [[ "$ENV" == Rotator* ]]; then
        export ALF_ROTATOR_LOG_DIR="logs/${NAME}/0"
    fi
    python -m alf.bin.train \
        "${COMMON_ARGS[@]}" \
        --root_dir="$ROOT_DIR"
fi
