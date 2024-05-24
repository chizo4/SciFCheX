#!/bin/bash
#
# AUTHOR:
# Filip J. Cierkosz
#
# INFO:
# The following script performs training for ABSTRACT RETRIEVAL
# moudle in "BIOBERT" setting, using the SCIFACT dataset. It uses
# BioBERT-base as the transformer of choice.
#
# NOTE:
# This script takes around 0.5 hour to compute on NVIDIA L4/A100 GPU.
# Hence, there is no point attempting to run it on CPU. GPU is
# a must for this task.
#
# USAGE:
# bash script/train-abstract-retrieval.sh [model]
#
# OPTIONS:
# [model] -> "scibert", "biobert_base", "biobert_large", "bert"

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
echo; echo "RUNNING ABSTRACT RETRIEVAL TRAINING ON 'SCIFACT' DATASET."; echo

# STEP 2: Ensure to switch to the target conda env.
echo; echo "Activating 'scifchex' environment..."
activate_env scifchex; echo

# STEP 3: Run the AR training script.
python pipeline/training/ar_transformer_scifact.py \
       --model $model
