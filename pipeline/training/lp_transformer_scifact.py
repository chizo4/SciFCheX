'''
--------------------------------------------------------------
FILE:
    pipeline/training/lp_transformer_scifact.py

NOTE:
    This module was applied from the work by Wadden et al.
    (2020) used in order to train a Transformer model (RoBERTa-large)
    for the task of LABEL PREDICTION. The code was optimized / simplified
    / re-organized, and better documented by Filip J. Cierkosz. Still
    keeping the core functionality coming from the source listed
    below. Also, the adjustments by FJC included testing models that
    were not experimented by Wadden et al. (2020), e.g., BioBERT-base.

TRANSFORMER MODEL:
    SCIFCHEX Optimal: BioBERT-large - defined as:
    'dmis-lab/biobert-large-cased-v1.1'.
    Other tested transformers: RoBERTa-large, BioBERT-base.

ADJUSTED BY:
    Filip J. Cierkosz

SOURCE:
    https://github.com/allenai/scifact

VERSION:
    05/2024
--------------------------------------------------------------
'''

import jsonlines
import os
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from typing import List

# Internal package import.
from tools.transformer import map_transformer

# Specify data files from SCIFACT dataset.
SCIFACT_CORPUS = 'data/corpus.jsonl'
SCIFACT_TRAIN_SET = 'data/claims_train.jsonl'
SCIFACT_DEV_SET = 'data/claims_dev.jsonl'

# Training hyperparameters - as in VERISCI (Wadden et al., 2020).
EPOCHS = 20
BATCH_SIZE_GPU = 8
BATCH_SIZE_ACCUMULATED = 256
LEARNING_RATE_BASE = 1e-5
LEARNING_RATE_LINEAR = 1e-4

class SciFactLabelPredictionDataset(Dataset):
    '''
    SciFactLabelPredictionDataset - load the SCIFACT dataset for LP.
    '''

    def __init__(self: 'SciFactLabelPredictionDataset', corpus: str, claims: str) -> None:
        self.samples = []
        # Load corpus.
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        # Set label mappings to integers.
        label_encodings = {
            'CONTRADICT': 0,
            'NOT_ENOUGH_INFO': 1,
            'SUPPORT': 2
        }
        # Build the sample set.
        for claim in jsonlines.open(claims):
            if claim['evidence']:
                for doc_id, evidence_sets in claim['evidence'].items():
                    doc = corpus[int(doc_id)]
                    # Add individual evidence set as samples.
                    for evidence_set in evidence_sets:
                        rationale = [
                            doc['abstract'][i].strip()
                            for i in evidence_set['sentences']
                        ]
                        self.samples.append({
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale),
                            'label': label_encodings[evidence_set['label']]
                        })
                    # Add all evidence sets as POSITIVE samples.
                    rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                    rationale_sentences = [
                        doc['abstract'][i].strip()
                        for i in sorted(list(rationale_idx))
                    ]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(rationale_sentences),
                        'label': label_encodings[evidence_sets[0]['label']]
                    })
                    # Add NEGATIVE samples.
                    non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
                    non_rationale_idx = random.sample(
                        non_rationale_idx,
                        k=min(random.randint(1, 2), len(non_rationale_idx))
                    )
                    non_rationale_sentences = [
                        doc['abstract'][i].strip()
                        for i in sorted(list(non_rationale_idx))
                    ]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
            else:
                # If no evidence for the claim, then add NEGATIVE samples.
                for doc_id in claim['cited_doc_ids']:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
                    non_rationale_sentences = [
                        doc['abstract'][i].strip()
                        for i in non_rationale_idx
                    ]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })

    def __len__(self: 'SciFactLabelPredictionDataset') -> int:
        '''
        Get the number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self: 'SciFactLabelPredictionDataset', idx: int) -> dict:
        '''
        Get a dataset item by index.
        '''
        return self.samples[idx]

def encode(claims: List[str], rationale: List[str]) -> dict:
    '''
    Encode claim and sentence for model.
    '''
    encoded_dict = tokenizer.batch_encode_plus(
        zip(rationale, claims),
        pad_to_max_length=True,
        return_tensors='pt'
    )
    # Handle samples exceeding the limit of 512 tokens. Truncate them.
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            zip(rationale, claims),
            max_length=512,
            truncation_strategy='only_first',
            pad_to_max_length=True,
            return_tensors='pt'
        )
    encoded_dict = {
        key: tensor.to(device)
        for key, tensor in encoded_dict.items()
    }
    return encoded_dict

def evaluate(model, dataset: 'SciFactLabelPredictionDataset') -> tuple:
    '''
    Evaluate the model during training. Compare ideal target results
    from dataset with the predicted model outputs.
    '''
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE_GPU):
            encoded_dict = encode(batch['claim'], batch['rationale'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
    }

def train_label_prediction() -> None:
    '''
    Run LABEL PREDICTION training.
    '''
    for e in range(EPOCHS):
        model.train()
        t = tqdm(DataLoader(train_set, batch_size=BATCH_SIZE_GPU, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode(batch['claim'], batch['rationale'])
            loss, _ = model(**encoded_dict, labels=batch['label'].long().to(device))
            loss.backward()
            if (i + 1) % (BATCH_SIZE_ACCUMULATED // BATCH_SIZE_GPU) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        # Evaluate the model scores on train set for the current EPOCH.
        train_score = evaluate(model, train_set)
        print(f'TRAIN SCORE FOR EPOCH {e}:')
        print(train_score)
        dev_score = evaluate(model, dev_set)
        print(f'DEV SCORE FOR EPOCH {e}:')
        print(dev_score)
        # Save the model...
        save_path = os.path.join(model_path, f'epoch-{e}-f1-{int(dev_score["macro_f1"] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

# Run the training for LP.
if __name__ == '__main__':
    # Set up (GPU) device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: "{device}"')
    # Determine the transformer of choice (from CLI), and its save path after fine-tuning.
    transformer_checkpoint, transformer_name = map_transformer()
    model_path = f'model/label_{transformer_name}_scifact'
    if transformer_checkpoint and transformer_name:
        print(f'Training LP transformer from checkpoint: {transformer_checkpoint}...\n')
        # Load train set (for training) and dev set (validation set).
        train_set = SciFactLabelPredictionDataset(
            corpus=SCIFACT_CORPUS,
            claims=SCIFACT_TRAIN_SET
        )
        dev_set = SciFactLabelPredictionDataset(
            corpus=SCIFACT_CORPUS,
            claims=SCIFACT_DEV_SET
        )
        # Load the transformer, etc. Set its config as classifier with 3 labels.
        tokenizer = AutoTokenizer.from_pretrained(transformer_checkpoint)
        config = AutoConfig.from_pretrained(transformer_checkpoint, num_labels=3)
        model = AutoModelForSequenceClassification.from_pretrained(transformer_checkpoint, config=config).to(device)
        # Set the optimizer depending on the BERT variation.
        if transformer_checkpoint == 'roberta-large':
            # For RoBERTa-large:
            optimizer = torch.optim.Adam([
                {'params': model.roberta.parameters(), 'lr': LEARNING_RATE_BASE},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_LINEAR}
            ])
        else:
            # For BioBERT-base, etc.:
            optimizer = torch.optim.Adam([
                {'params': model.bert.parameters(), 'lr': LEARNING_RATE_BASE},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_LINEAR}
            ])
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
        # Train...
        train_label_prediction()
    else:
        raise ValueError('Wrong Transformer name provided. Aborting training...\n')
