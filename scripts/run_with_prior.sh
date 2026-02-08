#!/bin/bash

bash scripts/run_bipolar.sh \
    CONF="experiments/bipolar_conf.py" \
    ENV="BipolarChain-medium-sparse-onehot-discrete-v0" \
    LR="0.05" \
    WD="1e-5" \
    GAMMA="0.95" \
    PRIOR_SCALE="0.01" \
    ALPHA="0" \
    TAU="0.05" \
    UTD=4 \
    NUM_AGENTS=1 \
    ASYNC="False" \
    ENTROPY_REWARD="False" \
    N_COMPONENTS=500 \
    SEEDS="" \
    BASE_DIR="" \
    NAME="with_prior" \
    EXTRA_ARGS=""
