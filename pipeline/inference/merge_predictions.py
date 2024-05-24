'''
--------------------------------------------------------------
FILE:
    pipeline/inference/merge_predictions.py

NOTE:
    This module was applied from the work by Wadden et al.
    (2020) used in order to merge predictions for RS/LP
    when running the scientific fact-checking pipeline. The
    code was optimized/simplified/re-organized, and better
    documented by Filip J. Cierkosz.

ADJUSTED BY:
    Filip J. Cierkosz

SOURCE:
    https://github.com/allenai/scifact

VERSION:
    02/2024
--------------------------------------------------------------
'''

import json

# Specify input/output files.
RATIONALE_FILE = 'prediction/rationale_selection.jsonl'
LABEL_FILE = 'prediction/label_prediction.jsonl'
RESULT_FILE = 'prediction/merged_predictions.jsonl'

# Handle no-info label.
NEI_LABEL = 'NOT_ENOUGH_INFO'

def merge_one(rationale, label) -> dict:
    '''
    Merge a single RATIONALE / LABEL pair. Discard NEI predictions.
    '''
    evidence = rationale['evidence']
    labels = label['labels']
    claim_id = rationale['claim_id']
    # Check that the documents match.
    if evidence.keys() != labels.keys():
        raise ValueError(f"Evidence docs for rationales and labels don't match for claim {claim_id}.")
    docs = sorted(evidence.keys())
    final_predictions = {}
    for this_doc in docs:
        this_evidence = evidence[this_doc]
        this_label = labels[this_doc]['label']
        if this_label != NEI_LABEL:
            final_predictions[this_doc] = {
                'sentences': this_evidence,
                'label': this_label
            }
    res = {
        'id': claim_id,
        'evidence': final_predictions
    }
    return res

def merge() -> None:
    '''
    Merge rationales with predicted labels.
    '''
    rationales = [json.loads(line) for line in open(RATIONALE_FILE)]
    labels = [json.loads(line) for line in open(LABEL_FILE)]
    # Verify the ordering.
    rationale_ids = [rat['claim_id'] for rat in rationales]
    label_ids = [lab['claim_id'] for lab in labels]
    if rationale_ids != label_ids:
        raise ValueError("Claim ID's for label and rationale file do NOT match!")
    res = [
        merge_one(rationale, label)
        for rationale, label in zip(rationales, labels)
    ]
    # Output the merged results.
    with open(RESULT_FILE, 'w') as f:
        for entry in res:
            print(json.dumps(entry), file=f)

# Merge predictions...
if __name__ == '__main__':
    merge()
