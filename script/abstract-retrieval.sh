#!/bin/bash
#
# AUTHOR:
# Filip J. Cierkosz
#
# INFO:
# The following script peforms Abstract Retrieval only.
# It also evaluates its results provided the dataset is:
# "train", or "dev".
#
# USAGE:
# bash script/abstract-retrieval.sh [retrieval] [dataset] [run_eval]
#
# OPTIONS:
# [retrieval] -> "biobert", "scibert", "bert", "bge_m3", "bm25", "tfidf", "oracle"
# [dataset]   -> "dev", "test", "train"
# [run_eval]  -> "true", "false"

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
run_eval=$3

# If BERT-based retrieval, then load the respective model from path.
if [ $retrieval == "biobert" ]
then
    echo; echo "Loading fine-tuned BioBERT-base for AR..."
    abstract_transformer="model/abstract_biobert_base_scifact/"
elif [ $retrieval == "scibert" ]
then
    echo; echo "Loading fine-tuned SciBERT for AR..."
    abstract_transformer="model/abstract_scibert_scifact/"
elif [ $retrieval == "bert" ]
then
    echo; echo "Loading fine-tuned BERT for AR..."
    abstract_transformer="model/abstract_bert_scifact/"
fi

####################

# RUN ABSTRACT RETRIEVAL

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
echo; echo "RUNNING ABSTRACT RETRIEVAL ON '${dataset}' DATASET."; echo

# STEP 2: Set up (or reset) prediction/ directory to store results.
rm -rf prediction
mkdir -p prediction

# STEP 3: Perform ABSTRACT RETRIEVAL.
echo; echo "ABSTRACT RETRIEVAL in '${retrieval}' setting."
# Option 1: Perform the HYBRID AR (BM25 + BIOBERT classifier).
if [ $retrieval == "biobert" ]
then
    python3 pipeline/inference/abstract_retrieval/biobert_retriever.py \
            --dataset data/claims_${dataset}.jsonl \
            --model ${abstract_transformer}
# Option 2: Perform the HYBRID AR (BM25 + SCIBERT classifier).
elif [ $retrieval == "scibert" ]
then
    python3 pipeline/inference/abstract_retrieval/scibert_retriever.py \
            --dataset data/claims_${dataset}.jsonl \
            --model ${abstract_transformer}
# Option 3: Perform the HYBRID AR (BM25 + BERT classifier).
elif [ $retrieval == "bert" ]
then
    python3 pipeline/inference/abstract_retrieval/bert_retriever.py \
            --dataset data/claims_${dataset}.jsonl \
            --model ${abstract_transformer}
# Option 4: Perform the HYBRID AR (BM25 + BGE M3 reranker).
elif [ $retrieval == "bge_m3" ]
then
    python3 pipeline/inference/abstract_retrieval/bge_m3_retriever.py \
            --dataset data/claims_${dataset}.jsonl
# Option 5: Perform the sparse BM25 AR.
elif [ $retrieval == "bm25" ]
then
    python3 pipeline/inference/abstract_retrieval/bm25_retriever.py \
            --dataset data/claims_${dataset}.jsonl
# Option 6: Perform the sparse TF-IDF AR.
elif [ $retrieval == "tfidf" ]
then
    python3 pipeline/inference/abstract_retrieval/tfidf_retriever.py \
            --dataset data/claims_${dataset}.jsonl
# Option 7: Oracle setting provides the gold abstracts (i.e., no retrieval).
else
    python3 pipeline/inference/abstract_retrieval/oracle_retriever.py \
        --dataset data/claims_${dataset}.jsonl
fi

# STEP 4: If performed BGE-M3 AR, then make sure to switch back to scifchex.
if [ $retrieval == "bge_m3" ]
then
    echo; echo "Activating 'scifchex' environment..."
    activate_env scifchex; echo
fi

# STEP 5: Optional. EVALUATION step to enable AR-only evaluation.
if [ $run_eval == "true" ] && [ $dataset != "test" ]
then
    echo "RUNNING ABSTRACT RETRIEVAL EVALUATION FOR '$dataset' DATASET."; echo
    python3 pipeline/eval/abstract_retrieval_evaluator.py \
        --retrieval $retrieval \
        --dataset $dataset
else
    echo "Skipping ABSTRACT RETRIEVAL evaluation..."; echo
fi
