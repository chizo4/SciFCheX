#!/bin/bash
#
# AUTHOR:
# Filip J. Cierkosz
#
# INFO:
# The following script performs training for RATIONALE SELECTION
# module using the SCIFACT dataset. It uses BioBERT-base as the
# default transformer of choice. The called script is adjusted
# from the work of Wadden et al. (2020).
#
# NOTE:
# This script takes around 0.5 hour to compute on NVIDIA L4/A100 GPU.
# Hence, there is no point attempting to run it on CPU. GPU is
# a must for this task.
#
# USAGE:
# bash script/train-rationale-selection.sh [model]
#
# OPTIONS:
# [model] -> "biobert_large", "biobert_base", "roberta_large"

####################

# HELPER FUNCTIONS

# Handle activating different conda environments.
activate_env() {
    ENV_PATH="/usr/local/envs/$1/bin"
    export PATH="$ENV_PATH:$PATH"
    echo; echo "Activated '$1' environment."
    which python; echo
}

####################

# SET UP ARGS / VARIABLES / TRANSFORMERS

# Fetch CLI args and set variables.
model=$1

####################

# RUN TRAINING

# STEP 1: Assert running the training.
echo; echo "RUNNING RATIONALE SELECTION TRAINING 'SCIFACT' DATASET."; echo

# STEP 2: Ensure to switch to the target conda env.
echo; echo "Activating 'scifchex' environment..."
activate_env scifchex; echo

# STEP 3: Run the RS training script.
python pipeline/training/rs_transformer_scifact.py \
       --model $model
