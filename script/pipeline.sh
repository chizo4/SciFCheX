#!/bin/bash
#
# AUTHOR:
# Filip J. Cierkosz
#
# INFO:
# The following script peforms full-pipeline predictions, starting from
# abstract retrieval, and then moving to rationale selection, and label
# prediciton stages. Finally, it evaluates the pipeline results, provided
# the dataset is either "train", or "dev".
#
# NOTE:
# Abstract Retrieval is run within this script by calling another script
# named: abstract-retrieval.sh.
#
# USAGE:
# bash script/pipeline.sh [retrieval] [model] [dataset]
#
# OPTIONS:
# [retrieval] -> "biobert", "scibert", "bert", "bge_m3", "tfidf", "oracle"
# [model]     -> "scifchex", "baseline", "verisci"
# [dataset]   -> "dev", "test"

####################

# HELPER FUNCTIONS

# Handle activating different conda environments.
activate_env() {
    ENV_PATH="/usr/local/envs/$1/bin"
    export PATH="$ENV_PATH:$PATH"
    echo; echo "Activated '$1' environment."
    which python; echo
}

# Check if models are loaded into the target directory.
if [ -d "model/" ]; then
    echo; echo "Models are loaded. Proceeding..."
else
    echo; echo "Models are NOT loaded. You must either load from Drive or run:"
    echo "> bash script/train-{module}.sh"
    echo "where {module} either targets AR, RS, or LP training..."; echo
fi

####################

# SET UP ARGS / VARIABLES / TRANSFORMERS

# Fetch CLI args and set variables.
retrieval=$1
model=$2
dataset=$3
rationale_threshold=0.5

# Handle loading models, either: VERISCI, BASELINE, SCIFCHEX.
# VERISCI - RS: RoBERTa-large (SciFact); LP: RoBERTa-large (FEVER+SciFact).
if [ $model == "verisci" ]
then
    echo; echo "Loading MODELS for VERISCI..."
    rationale_transformer="model/rationale_roberta_large_scifact/"
    label_transformer="model/label_roberta_large_fever_scifact/"
# BASELINE - RS: RoBERTa-large (SciFact); LP: RoBERTa-large (SciFact).
elif [ $model == "baseline" ]
then
    echo; echo "Loading MODELS for BASELINE..."
    rationale_transformer="model/rationale_roberta_large_scifact/"
    label_transformer="model/label_roberta_large_scifact/"
# SCIFCHEX - RS: BioBERT-large (SciFact); LP: BioBERT-large (SciFact).
else
    echo; echo "Loading MODELS for SCIFCHEX..."
    rationale_transformer="model/rationale_scibert_scifact/"
    label_transformer="model/label_roberta_large_fever_scifact/"
fi

####################

# RUN PIPELINE

# STEP 1: Assert running the model of choice on the selected dataset.
echo; echo "RUNNING '${model}' PIPELINE ON '${dataset}' DATASET."; echo

# STEP 2: Run the Abstract Retrieval stage (via its script).
bash script/abstract-retrieval.sh ${retrieval} ${dataset} "true"

# STEP 3: Ensure to switch to the target conda env.
echo; echo "Activating 'scifchex' environment..."
activate_env scifchex; echo

# STEP 4: Run the Rationale Selection stage.
echo; echo "Running RATIONALE SELECTION."
python3 pipeline/inference/rationale_selection/rs_transformer.py \
    --dataset data/claims_${dataset}.jsonl \
    --model ${rationale_transformer}

# STEP 5: Run the Label Prediction stage.
echo; echo "Running LABEL PREDICTION."
python3 pipeline/inference/label_prediction/lp_transformer.py \
    --dataset data/claims_${dataset}.jsonl \
    --model ${label_transformer}

# STEP 6: Merge rationale and label predictions.
echo; echo "MERGING PREDICTIONS from RS/LP..."; echo
python3 pipeline/inference/merge_predictions.py

# STEP 7: Evaluate merged pipeline predictions.
if  [ $dataset == "test" ]
then
    echo "Test dataset. Skip EVALUATION (no labels provided)."
else
    echo "Evaluating on fold $dataset"
    python3 pipeline/eval/pipeline_evaluator.py \
        --dataset $dataset \
        --model $model \
        --retrieval $retrieval
fi
