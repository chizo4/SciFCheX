'''
--------------------------------------------------------------
FILE:
    pipeline/training/ar_transformer_scifact.py

INFO:
    This module implements essential tools to fine-tune BERT-like
    transformers on SciFact dataset for the task of ABSTRACT RETRIEVAL
    in e.g. "BIOBERT" (HYBRID) setting.
    Unlike RS/LP training files, this module was implemented from
    scratch by Filip J. Cierkosz, since the work from Wadden et al.
    (2020) did not include any training approaches for AR modules.
    This is innovative part of this research, that included testing
    different BERT variations in combination with initial BM25 ranking.

TRANSFORMER MODEL:
    SCIFCHEX Optimal (standalone): BioBERT-base.
    SCIFCHEX Optimal (pipeline): SciBERT.
    Other tested: SciBERT, BERT.

NOTE:
    See line 50 if wanted to experiment with different BERT models.
    The current setting is set for most optimal model: BioBERT-base.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    05/2024
--------------------------------------------------------------
'''

import jsonlines
import numpy as np
import os
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
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

# Training hyperparameters.
EPOCHS = 10
BATCH_SIZE_GPU = 8
BATCH_SIZE_ACCUMULATED = 256
LEARNING_RATE_BASE = 1e-5
LEARNING_RATE_LINEAR = 1e-3

class SciFactAbstractRetrievalDataset(Dataset):
    '''
    SciFactAbstractRetrievalDataset - class to handle loading and pre-processing
                                      the SCIFACT dataset for AR training.
    '''

    def __init__(self: 'SciFactAbstractRetrievalDataset', dataset: str) -> None:
        '''
        Initialize the class to sample the dataset fold of choice.

            Parameters:
            -------------------------
            dataset : str
                The path to access the dataset fold of choice to be sampled.
        '''
        self.dataset = dataset
        # Load the contents of corpus.
        self.corpus = {doc['doc_id']: doc for doc in jsonlines.open(SCIFACT_CORPUS)}
        self.corpus_docs = list(jsonlines.open(SCIFACT_CORPUS))
        # BM25: Build tokenizer, tokenize corpus, and load the model.
        self.tokenizer_bm25 = CountVectorizer(stop_words='english').build_analyzer()
        self.corpus_tokens = [
            self.tokenizer_bm25(doc['title'] + ' ' + ' '.join(doc['abstract']))
            for doc in list(jsonlines.open(SCIFACT_CORPUS))
        ]
        self.model_bm25 = BM25Okapi(self.corpus_tokens)
        # Rank TOP 20 abstract docs for each claim from the dataset fold.
        self.top20_docs_for_claims = self.rank_bm25()
        # Build dataset samples, incl. positive/negative samples.
        self.samples = self.build_samples()

    def __len__(self: 'SciFactAbstractRetrievalDataset') -> int:
        '''
        Get the length of the sampled dataset.

            Returns:
            -------------------------
            int
                The number of samples in the fold.
        '''
        return len(self.samples)

    def __getitem__(self: 'SciFactAbstractRetrievalDataset', i: int) -> dict:
        '''
        Get a dataset fold item by index.

            Parameters:
            -------------------------
            i : int
                Index of the item.

            Returns:
            -------------------------
            dict
                The searched sample item.
        '''
        return self.samples[i]

    def rank_bm25(self: 'SciFactAbstractRetrievalDataset', K=20) -> list:
        '''
        Perform BM25 ranking of abstract docs for each sample from
        the dataset fold. In nutshell, it finds the TOP 20 docs for
        each claim from the dataset fold.

            Parameters:
            -------------------------
            K : int, default=20
                Number of top docs for BM25.

            Returns:
            -------------------------
            top_doc_ids_res : list
                Top docs listed for each sample from the dataset fold.
        '''
        top_doc_ids_res = []
        # Perform ranking of abstract docs for each of the claim sample.
        for data in jsonlines.open(self.dataset):
            # Set the current claim and tokenize for BM25.
            claim = data['claim']
            claim_tokens = self.tokenizer_bm25(claim)
            doc_scores = self.model_bm25.get_scores(claim_tokens)
            # Prepare K best docs found for the BM25 stage.
            top_indices_bm25 = np.argsort(doc_scores)[::-1][:K]
            # Assign the courpus IDs of the identified top docs.
            top_doc_ids = [self.corpus_docs[i]['doc_id'] for i in top_indices_bm25]
            # Update the results with new entry.
            top_doc_ids_res.append(top_doc_ids)
        return top_doc_ids_res

    def build_samples(self: 'SciFactAbstractRetrievalDataset') -> list:
        '''
        Build the sample set for the fold using ranked docs.
        Utilize them to perform positive/negative sampling.
        Use matching docs as positive samples, and the other
        unmatching ones as negatives.

            Returns:
            -------------------------
            samples : list
                The list of sampled dataset.
        '''
        samples = []
        for i, data in enumerate(jsonlines.open(self.dataset)):
            # Extract top 20 docs for the current claim (data item).
            data_top_docs = self.top20_docs_for_claims[i]
            # Set correctly identified abstract IDs.
            pos_abstract_ids = [
                int(doc_id) for doc_id in list(data['evidence'].keys())
                if int(doc_id) in data_top_docs
            ]
            cited_ids = [
                int(doc_id) for doc_id in data['cited_doc_ids']
                if doc_id in data_top_docs
            ]
            # Extract the set of uncorrect abstract IDs.
            neg_abstract_ids = set(cited_ids) - set(pos_abstract_ids)
            # Add POSITIVE samples.
            for doc_id in pos_abstract_ids:
                doc = self.corpus[doc_id]
                # For token size, include claim+title only.
                samples.append({
                    'claim': data['claim'],
                    'title': doc['title'],
                    'evidence': 1
                })
            # Add NEGATIVE samples for wrong identifications.
            if len(neg_abstract_ids) > 0:
                for doc_id in neg_abstract_ids:
                    doc = self.corpus[doc_id]
                    samples.append({
                        'claim': data['claim'],
                        'title': doc['title'],
                        'evidence': 0
                    })
            # Use the remaining BM25 retrievals, as also NEGATIVE samples.
            unmatched_docs = [
                doc_id for doc_id in data_top_docs
                if doc_id not in cited_ids
            ]
            for doc_id in unmatched_docs:
                doc = self.corpus[doc_id]
                samples.append({
                    'claim': data['claim'],
                    'title': doc['title'],
                    'evidence': 0
                })
        return samples

def encode(claims: List[str], titles: List[str]) -> dict:
    '''
    Encode the claim+title for the model.

        Returns:
        -------------------------
        encoded_dict : dict
            The encoded dictionary for the model.
    '''
    encoded_dict = tokenizer.batch_encode_plus(
        zip(titles, claims),
        pad_to_max_length=True,
        return_tensors='pt'
    )
    # Handle samples exceeding the limit of 512 tokens. Truncate them.
    if encoded_dict['input_ids'].size(1) > 512:
        encoded_dict = tokenizer.batch_encode_plus(
            zip(titles, claims),
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

def evaluate(model, dataset: 'SciFactAbstractRetrievalDataset') -> dict:
    '''
    Evaluate the model during training. Compare ideal target results
    from dataset with the predicted model outputs.

        Returns:
        -------------------------
        res : dict
            The dictionary that contains the current results for the
            model. They include: precision, recall and F1.
    '''
    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE_GPU):
            encoded_dict = encode(batch['claim'], batch['title'])
            logits = model(**encoded_dict)[0]
            targets.extend(batch['evidence'].long().tolist())
            outputs.extend(logits.argmax(dim=1).long().tolist())
    res = {
        'f1': f1_score(targets, outputs, zero_division=0),
        'precision': precision_score(targets, outputs, zero_division=0),
        'recall': recall_score(targets, outputs, zero_division=0)
    }
    return res

def train_abstract_retrieval() -> None:
    '''
    Run ABSTRACT RETRIEVAL training.
    '''
    for e in range(EPOCHS):
        model.train()
        t = tqdm(DataLoader(
            train_set,
            batch_size=BATCH_SIZE_GPU,
            shuffle=True
        ))
        for i, batch in enumerate(t):
            encoded_dict = encode(batch['claim'], batch['title'])
            loss, _ = model(
                **encoded_dict,
                labels=batch['evidence'].long().to(device)
            )
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
        save_path = os.path.join(model_path, f'epoch-{e}-f1-{int(dev_score["f1"] * 1e4)}')
        os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

# Run the training for AR.
if __name__ == '__main__':
    # Set up (GPU) device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: "{device}"')
    # Determine the transformer of choice, and its path to save after fine-tuning.
    transformer_checkpoint, transformer_name = map_transformer()
    model_path = f'model/abstract_{transformer_name}_scifact'
    if transformer_checkpoint and transformer_name:
        print(f'Training AR transformer from checkpoint: {transformer_checkpoint}...\n')
        # Load train set (for training) and dev set (validation set).
        train_set = SciFactAbstractRetrievalDataset(dataset=SCIFACT_TRAIN_SET)
        dev_set = SciFactAbstractRetrievalDataset(dataset=SCIFACT_DEV_SET)
        # Load the transformer, optimizer and scheduler. Set its config as classifier with 3 labels.
        tokenizer = AutoTokenizer.from_pretrained(transformer_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(transformer_checkpoint).to(device)
        # For BERT variants (e.g., BioBERT-large). AR does not allow other types.
        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': LEARNING_RATE_BASE},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_LINEAR}
        ])
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
        # Train...
        train_abstract_retrieval()
    else:
        raise ValueError('Wrong Transformer name provided. Aborting AR training...\n')
