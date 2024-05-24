'''
--------------------------------------------------------------
FILE:
    pipeline/inference/label_prediction/transformer.py

NOTE:
    This module is author's re-implementation of work by Wadden
    et al. (2020), following a similar methodology, following
    a cleaner organization of resources with docs.

INFO:
    Perform label predictions using the passed model of choice
    (BERT-like). The labels are predicted using the rationale
    sentences extracted from candidate abstracts by RS transformer.

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
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

# Specify required files.
CORPUS_FILE = 'data/corpus.jsonl'
RS_RESULTS_FILE = 'prediction/rationale_selection.jsonl'
OUTPUT_FILE = 'prediction/label_prediction.jsonl'

# Labels for predictions.
NUM_LABELS = 3
LABELS = ['CONTRADICT', 'NOT_ENOUGH_INFO', 'SUPPORT']

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
    args = parser.parse_args()
    return args

def encode(sentences: str, claim: str) -> dict:
    '''
    Encode sentence and claim to get them ready to be
    passed into the model for predictions.
    '''
    sc_input = list(zip(sentences, claim))
    encoded_dict = tokenizer.batch_encode_plus(
        sc_input,
        pad_to_max_length=True,
        return_tensors='pt'
    )
    # Truncate longer inputs, if exceeding the allowed token size.
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            sc_input,
            max_length=512,
            pad_to_max_length=True,
            truncation_strategy='only_first',
            return_tensors='pt'
        )
    encoded_dict = {
        key: tensor.to(device)
        for key, tensor in encoded_dict.items()
    }
    return encoded_dict

def predict_labels() -> None:
    '''
    Perform label predictions utilizing the passed BERT model.
    Subequently, write the results into the output file.
    '''
    output = jsonlines.open(OUTPUT_FILE, 'w')
    with torch.no_grad():
        for data, selection in tqdm(list(zip(dataset, rationale_selection))):
            assert data['id'] == selection['claim_id']
            claim = data['claim']
            results = {}
            for doc_id, indices in selection['evidence'].items():
                # Handle NEI label.
                if not indices:
                    results[doc_id] = {
                        'label': 'NOT_ENOUGH_INFO',
                        'confidence': 1
                    }
                else:
                    # Otherwise, encode evidence and claim and pass to the model.
                    evidence = ' '.join(
                        [corpus[int(doc_id)]['abstract'][i] for i in indices]
                    )
                    encoded_dict = encode([evidence], [claim])
                    label_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[0]
                    label_index = label_scores.argmax().item()
                    label_confidence = label_scores[label_index].item()
                    results[doc_id] = {
                        'label': LABELS[label_index],
                        'confidence': round(label_confidence, 4)
                    }
            # Write the results...
            output.write({
                'claim_id': data['id'],
                'labels': results
            })

# Run LABEL predictions.
if __name__ == '__main__':
    # Load CLI args...
    args = set_args()
    # Load corpus, dataset and rationale selections.
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(CORPUS_FILE)}
    dataset = jsonlines.open(args.dataset)
    rationale_selection = jsonlines.open(RS_RESULTS_FILE)
    # Set up the device. Ideally, GPU for faster computation.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: "{device}"')
    # Load the model specified in args through Transformers utils. Set labels num.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=NUM_LABELS
    )
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).eval().to(device)
    # Run label predictions and output the results.
    predict_labels()
