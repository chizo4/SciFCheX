'''
--------------------------------------------------------------
FILE:
    pipeline/training/rs_transformer_scifact.py

NOTE:
    This module was applied from the work by Wadden et al.
    (2020) used in order to train a Transformer model (BioBERT-large)
    for the task of RATIONALE SELECTION. The code was optimized / simplified
    / re-organized, and better documented by Filip J. Cierkosz. Still
    keeping the core functionality coming from the source listed
    below. Also, the adjustments by FJC included testing models that
    were not experimented by Wadden et al. (2020), e.g., BioBERT-large.
    This module was applied from the work by Wadden et al.

TRANSFORMER MODEL:
    SCIFCHEX Optimal: SciBERT.
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
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
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
LEARNING_RATE_LINEAR = 1e-3

class SciFactRationaleSelectionDataset(Dataset):
    '''
    SciFactRationaleSelectionDataset - load the SCIFACT dataset for RS.
    '''

    def __init__(self: 'SciFactRationaleSelectionDataset', corpus: str, claims: str) -> None:
        '''
        Initialize the dataset.

            Parameters:
            -------------------------
            corpus : str
                Path to access the dataset corpus file.
            claims : str
                Path to access the dataset fold file (e.g., train).
        '''
        self.samples = []
        # Load corpus.
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        # Append each sample claim from the fold of choice..
        for claim in jsonlines.open(claims):
            for doc_id, evidence in claim['evidence'].items():
                doc = corpus[int(doc_id)]
                evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': claim['claim'],
                        'sentence': sentence,
                        'evidence': i in evidence_sentence_idx
                    })

    def __len__(self: 'SciFactRationaleSelectionDataset') -> int:
        '''
        Get the number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self: 'SciFactRationaleSelectionDataset', i: int) -> dict:
        '''
        Get a dataset item by index.
        '''
        return self.samples[i]

def encode(claims: List[str], sentences: List[str]) -> dict:
    '''
    Encode claim and sentence for model.
    '''
    encoded_dict = tokenizer.batch_encode_plus(
        zip(sentences, claims),
        pad_to_max_length=True,
        return_tensors='pt'
    )
    # Handle samples exceeding the limit of 512 tokens. Truncate them.
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            zip(sentences, claims),
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

def evaluate(model, dataset: 'SciFactRationaleSelectionDataset') -> tuple:
    '''
    Evaluate the model during training. Compare ideal target results
    from dataset with the predicted model outputs.
    '''
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE_GPU):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
    scores = (
        f1_score(targets, outputs, zero_division=0),
        precision_score(targets, outputs, zero_division=0),
        recall_score(targets, outputs, zero_division=0)
    )
    return scores

def train_rationale_selection() -> None:
    '''
    Run RATIONALE SELECTION training.
    '''
    for e in range(EPOCHS):
        model.train()
        t = tqdm(DataLoader(train_set, batch_size=BATCH_SIZE_GPU, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode(batch['claim'], batch['sentence'])
            loss, _ = model(**encoded_dict, labels=batch['evidence'].long().to(device))
            loss.backward()
            if (i + 1) % (BATCH_SIZE_ACCUMULATED // BATCH_SIZE_GPU) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        # Evaluate the model scores on train set for the current EPOCH.
        (train_f1, train_p, train_r) = evaluate(model, train_set)
        print(f'Epoch {e}, TRAINSET F1: {train_f1}, precision: {train_p}, recall: {train_r}')
        # Evaluate the model scores on dev set for the current EPOCH.
        (dev_f1, dev_p, dev_r) = evaluate(model, dev_set)
        print(f'Epoch {e}, DEVSET F1: {dev_f1}, precision: {dev_p}, recall: {dev_r}')
        # Save the model...
        save_model_path = os.path.join(model_path, f'epoch-{e}-f1-{int(dev_f1 * 1e4)}')
        os.makedirs(save_model_path)
        tokenizer.save_pretrained(save_model_path)
        model.save_pretrained(save_model_path)

# Run the training for RS.
if __name__ == '__main__':
    # Set up (GPU) device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: "{device}"')
    # Determine the transformer of choice (from CLI), and its save path after fine-tuning.
    transformer_checkpoint, transformer_name = map_transformer()
    model_path = f'model/rationale_{transformer_name}_scifact'
    if transformer_checkpoint and transformer_name:
        print(f'Training RS transformer from checkpoint: {transformer_checkpoint}...\n')
        # Load train set (for training) and dev set (validation set).
        train_set = SciFactRationaleSelectionDataset(corpus=SCIFACT_CORPUS, claims=SCIFACT_TRAIN_SET)
        dev_set = SciFactRationaleSelectionDataset(corpus=SCIFACT_CORPUS, claims=SCIFACT_DEV_SET)
        # Load the transformer, etc.
        tokenizer = AutoTokenizer.from_pretrained(transformer_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(transformer_checkpoint).to(device)
        # Set the optimizer depending on the BERT variation.
        if transformer_checkpoint == 'roberta-large':
            # For RoBERTa-large:
            optimizer = torch.optim.Adam([
                {'params': model.roberta.parameters(), 'lr': LEARNING_RATE_BASE},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_LINEAR}
            ])
        else:
            # For BERT variants (e.g., BioBERT-large):
            optimizer = torch.optim.Adam([
                {'params': model.bert.parameters(), 'lr': LEARNING_RATE_BASE},
                {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_LINEAR}
            ])
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
        # Train...
        train_rationale_selection()
    else:
        raise ValueError('Wrong Transformer name provided. Aborting training...\n')
