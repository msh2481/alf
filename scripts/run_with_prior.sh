#!/bin/bash

bash scripts/run_bipolar.sh \
    CONF="experiments/bipolar_conf.py" \
    ENV="BipolarChain-medium-sparse-onehot-discrete-v0" \
    LR="0.1" \
    WD="1e-5" \
    GAMMA="0.95" \
    PRIOR_SCALE="2.0" \
    ALPHA="0" \
    TAU="0.05" \
    UTD=8 \
    RESET_PERIOD=24 \
    NUM_AGENTS=4 \
    ASYNC="True" \
    ENTROPY_REWARD="False" \
    N_COMPONENTS=500 \
    SEEDS="" \
    BASE_DIR="" \
    NAME="with_prior" \
    EXTRA_ARGS=""
