#!/usr/bin/env bash
python "./trajectories.py" --alpha 0.9 --snapshots 40 --k 2000 --thin 10 --view pca "$@"
