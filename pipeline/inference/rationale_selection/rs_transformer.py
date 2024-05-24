'''
--------------------------------------------------------------
FILE:
    pipeline/inference/rationale_selection/transformer.py

NOTE:
    This module is author's re-implementation of work by Wadden
    et al. (2020), following a similar methodology, following
    a cleaner organization of resources with docs.

INFO:
    Perform rationale selection predictions using the passed
    model of choice (BERT-like). The sentences are extracted
    from the predictions passed from the earlier abstract
    retrieval stage.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

import argparse
import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Specify required files.
CORPUS_FILE = 'data/corpus.jsonl'
AR_RESULTS_FILE = 'prediction/abstract_retrieval.jsonl'
OUTPUT_FILE = 'prediction/rationale_selection.jsonl'

def set_args() -> argparse.Namespace:
    '''
    Handle parsing CLI args associated with the script.

        Returns:
        -------------------------
        args : argparse.Namespace
            Object contains pre-processed CLI args.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        required=False
    )
    args = parser.parse_args()
    return args

def predict_rationale_selection() -> list:
    '''
    Perform rationale selection predictions utilizing the passed BERT model.

        Returns:
        -------------------------
        results : list
            Predicted results by the model.
    '''
    results = []
    with torch.no_grad():
        for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
            assert data['id'] == retrieval['claim_id']
            claim = data['claim']
            evidence_scores = {}
            # Extract sentences from predicted abstracts.
            for doc_id in retrieval['doc_ids']:
                doc = corpus[doc_id]
                sentences = doc['abstract']
                # Feed the model with extracted sentences and claim.
                encoded_dict = tokenizer.batch_encode_plus(
                    zip(sentences, [claim] * len(sentences)),
                    pad_to_max_length=True,
                    return_tensors='pt'
                )
                # Transfer to GPU.
                encoded_dict = {
                    key: tensor.to(device)
                    for key, tensor in encoded_dict.items()
                }
                sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1].detach().cpu().numpy()
                evidence_scores[doc_id] = sentence_scores
            results.append({
                'claim_id': retrieval['claim_id'],
                'evidence_scores': evidence_scores
            })
    return results

def write_results_output(output_path: str, results: list, thresh=0.5) -> None:
    '''
    Write RS results into its output file.

        Parameters:
        -------------------------
        output_path : str
            Path to write the RS prediction results.
        results : list
            List of predictions for RS stage.
        thresh : float, default=0.5
            Threshold for decision rule in RS predictions.
    '''
    output = jsonlines.open(output_path, 'w')
    for res in results:
        evidence = {
            doc_id: (sentence_scores >= thresh).nonzero()[0].tolist()
            for doc_id, sentence_scores in res['evidence_scores'].items()
        }
        output.write({
            'claim_id': res['claim_id'],
            'evidence': evidence
        })

# Run rationale selection predictions.
if __name__ == '__main__':
    # Load CLI args...
    args = set_args()
    # Load corpus, dataset and AR predictions.
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(CORPUS_FILE)}
    dataset = jsonlines.open(args.dataset)
    abstract_retrieval = jsonlines.open(AR_RESULTS_FILE)
    # Set up the device. Ideally, GPU for faster computation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: "{device}"')
    # Load the model specified in args through Transformers utils.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    # Run predictions on the model loaded above.
    results = predict_rationale_selection()
    # Detect threshold arg and write out the results.
    if args.threshold:
        write_results_output(
            output_path=OUTPUT_FILE,
            results=results,
            thresh=args.threshold
        )
    else:
        write_results_output(output_path=OUTPUT_FILE)
