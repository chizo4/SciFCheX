#!/bin/bash
#
# AUTHOR:
# Filip J. Cierkosz
#
# INFO:
# The following script peforms experiments for RECALL@K
# for Abstract Retrieval methods. It is very similar to
# script/abstract-retrieval.sh. In nutshell, this script
# includes K args for each retrieval method and executes
# different metrics in AbstractRetrievalEvaluator.
#
# NOTE:
# BERT-based classifier models are excluded from these
# experiments, since they classify docs, and so cannot be
# forced to return K docs.
#
# USAGE:
# bash script/abstract-retrieval-recall.sh [retrieval] [dataset]
#
# OPTIONS:
# [retrieval] -> "bge_m3", "bm25", "tfidf", "oracle"
# [dataset]   -> "dev", "train"

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

# SET UP ARGS / VARIABLES

# Fetch CLI args and set variables.
retrieval=$1
dataset=$2

####################

# RUN RECALL@K EXPERIMENT FOR ABSTRACT RETRIEVAL

# STEP 0: Activate the target environment. BGE-M3 requires different env.
if [ $retrieval == "bge_m3" ]
then
    echo; echo "Activating 'reranker' environment..."
    activate_env reranker; echo
else
    echo; echo "Activating 'scifchex' environment..."
    activate_env scifchex; echo
fi

# STEP 1: Assert running on the dataset of choice.
echo; echo "RUNNING ABSTRACT RETRIEVAL RECALL@K FOR '${dataset}' DATASET."; echo

# STEP 2: Set up (or reset) prediction/ directory to store results.
rm -rf prediction
mkdir -p prediction


echo; echo "ABSTRACT RETRIEVAL RECALL@K in '${retrieval}' setting."
# STEP 3: Perform ABSTRACT RETRIEVAL.
echo; echo "ABSTRACT RETRIEVAL in '${retrieval}' setting."
# Option 1: Perform the HYBRID AR (BM25 + BGE M3 reranker).
if [ $retrieval == "bge_m3" ]
then
    python3 pipeline/inference/abstract_retrieval/bge_m3_retriever.py \
            --dataset data/claims_${dataset}.jsonl \
            --k 20
# Option 2: Perform the sparse BM25 AR.
elif [ $retrieval == "bm25" ]
then
    python3 pipeline/inference/abstract_retrieval/bm25_retriever.py \
            --dataset data/claims_${dataset}.jsonl \
            --k 20
# Option 3: Perform the sparse TF-IDF AR.
elif [ $retrieval == "tfidf" ]
then
    python3 pipeline/inference/abstract_retrieval/tfidf_retriever.py \
            --dataset data/claims_${dataset}.jsonl \
            --k 20
# Option 4: Oracle setting provides the gold abstracts (i.e., no retrieval).
else
    python3 pipeline/inference/abstract_retrieval/oracle_retriever.py \
        --dataset data/claims_${dataset}.jsonl \
        --k 20
fi

# STEP 4: If performed BGE-M3 AR, then make sure to switch back to scifchex.
if [ $retrieval == "bge_m3" ]
then
    echo; echo "Activating 'scifchex' environment..."
    activate_env scifchex; echo
fi

# STEP 5: EVALUATION R@K.
echo "RUNNING ABSTRACT RETRIEVAL RECALL@K FOR '$dataset' DATASET."; echo
python3 pipeline/eval/abstract_retrieval_evaluator.py \
    --retrieval $retrieval \
    --dataset $dataset \
    --recall "yes"
