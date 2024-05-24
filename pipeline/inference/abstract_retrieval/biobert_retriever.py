'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/biobert_retriever.py

NOTE:
    Prior to running this script you must have either trained
    the BioBERT-base model through:
    > bash script/train-abstract-retrieval.sh
    **OR**
    Used a previously fine-tuned model for this task, which would
    mean that you previously run the script listed above.

INFO:
    The script implements the BioBERTRetriever class that
    performs Abstract Retrieval in "BIOBERT" (i.e. HYBRID) setting.
    It first uses the BM25 algorithm to find 20 matching docs, that
    are further classified by BioBERT-base classifier to determine
    final picks. This AR approach is used for the optimal version
    of SCIFCHEX pipeline. It also uses SciFactDataset class to
    pre-process the dataset, finding top K docs for each sample
    prior to classification. This script was also earlier adjusted
    to conduct similar experiments with BioBERT-large for AR.
    However, after observing better performance with base model
    the decision has been made to exclude large from this setup
    (to avoid any confusion). The results for large are still reported
    in results/abstract_retrieval directory.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    05/2024
--------------------------------------------------------------
'''

import jsonlines
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from tools.retriever import Retriever
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BioBERTRetriever(Retriever):
    '''
    -------------------------
    BioBERTRetriever - Class for Abstract Retrieval in "BIOBERT" (HYBRID) setting.
                       It utilizes a fine-tuned BioBERT-base on SciFact.
                       Parent class: Retriever.
    -------------------------
    '''

    # Hyperparameters for model eval.
    BATCH_SIZE_GPU = 20

    def __init__(self: 'BioBERTRetriever') -> None:
        '''
        Initialize the class for AR in "BIOBERT" setting.
        '''
        # BM25: Build tokenizer, tokenize corpus, and load the model.
        super().__init__(retrieval_strategy='BIOBERT')
        self.tokenizer_bm25 = CountVectorizer(stop_words='english').build_analyzer()
        self.corpus_tokens = [
            self.tokenizer_bm25(doc['title'] + ' ' + ' '.join(doc['abstract']))
            for doc in self.corpus
        ]
        self.model_bm25 = BM25Okapi(self.corpus_tokens)
        # Rank TOP K abstract docs for each claim from the dataset fold.
        self.top_k_docs_for_claims = self.rank_bm25()
        # Instantiate the SciFact dataset instance. Required for torch.DataLoader.
        self.scifact_dataset = SciFactDataset(
            dataset=self.args.dataset,
            corpus=self.CORPUS_PATH,
            top_k_docs_for_claims=self.top_k_docs_for_claims
        )
        # Set up the device. Run the model on GPU (if available).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\nUsing device: "{self.device}"\n')
        # Load the BIOBERT-base transformer fine-tuned on SCIFACT.
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.biobert_model = AutoModelForSequenceClassification.from_pretrained(self.args.model).to(self.device).eval()

    def rank_bm25(self: 'BioBERTRetriever', K=20) -> list:
        '''
        Perform BM25 ranking of abstract docs for each sample from
        the dataset fold. In nutshell, it finds the TOP K=20 docs for
        each claim from the dataset fold.

            Parameters:
            -------------------------
            K : int, default=20
                Number of top docs for BM25.

            Returns:
            -------------------------
            top_doc_ids_res : list
                Top K=20 docs listed for each sample from the dataset fold.
        '''
        top_doc_ids_res = []
        # Perform ranking of abstract docs for each of the claim sample.
        for data in self.dataset:
            # Set the current claim and tokenize for BM25.
            claim = data['claim']
            claim_tokens = self.tokenizer_bm25(claim)
            doc_scores = self.model_bm25.get_scores(claim_tokens)
            # Prepare K=20 best docs found for the BM25 stage.
            top_indices_bm25 = np.argsort(doc_scores)[::-1][:K]
            # Assign the corpus IDs of the identified top docs.
            top_doc_ids = [self.corpus[i]['doc_id'] for i in top_indices_bm25]
            # Update the results with new entry.
            top_doc_ids_res.append(top_doc_ids)
        return top_doc_ids_res

    def encode(self: 'BioBERTRetriever', claims: list, titles: list) -> dict:
        '''
        Encode the claim+title for the model predictions.

            Parameters:
            -------------------------
            claims : list
                Batch for claims.
            titles : list
                Batch for titles.

            Returns:
            -------------------------
            encoded_dict : dict
                The encoded dictionary for the model.
        '''
        encoded_dict = self.biobert_tokenizer.batch_encode_plus(
            zip(titles, claims),
            pad_to_max_length=True,
            return_tensors='pt'
        )
        # Handle samples exceeding the limit of 512 tokens. Truncate them.
        if encoded_dict['input_ids'].size(1) > 512:
            encoded_dict = self.biobert_tokenizer.batch_encode_plus(
                zip(titles, claims),
                max_length=512,
                truncation_strategy='only_first',
                pad_to_max_length=True,
                return_tensors='pt'
            )
        encoded_dict = {
            key: tensor.to(self.device)
            for key, tensor in encoded_dict.items()
        }
        return encoded_dict

    def classify(self: 'BioBERTRetriever', K=3) -> None:
        '''
        Perform classification of BM25 doc candidates for each claim
        from the dataset fold. The classification is performed using
        the loaded BioBERT-base model that was fine-tuned on SciFact.

            Parameters:
            -------------------------
            K : int, default=3
                The maximum number of classified docs. It is the
                upper threshold, which means that IFF the output
                contains more than 3 scores of 1, then only the
                first 3 will be considered.
        '''
        classify_res = []
        self.biobert_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(self.scifact_dataset, batch_size=self.BATCH_SIZE_GPU)):
                encoded_dict = self.encode(batch['claim'], batch['title'])
                logits = self.biobert_model(**encoded_dict)[0]
                output = logits.argmax(dim=1).long().tolist()
                # Classification step - keep only the results with score 1.
                # If more than K results computed, then truncate at K-1 index.
                if 1 in output:
                    classified_indices = np.argwhere(np.array(output) == 1)
                    top_docs_biobert = [
                        self.top_k_docs_for_claims[i][ix[0]]
                        for ix in classified_indices
                    ]
                    # Allow at most K=3 results.
                    classify_res.append(top_docs_biobert[:K])
                else:
                    classify_res.append([])
        return classify_res

    def run(self: 'BioBERTRetriever') -> None:
        '''
        Run the "BIOBERT" retrieval by calling the classifier function
        and then writing the predictions into target output file.
        '''
        classify_res = self.classify()
        # Build the results by processing the classifier's results.
        for i, data in enumerate(self.dataset):
            self.retrieval_results[data['id']] = classify_res[i]
        self.write_results()
        print(f'Abstract Retrieval in "{self.retrieval_strategy}" setting: COMPLETE.\n')

######################################

class SciFactDataset(Dataset):
    '''
    SciFactDataset - class to handle loading and pre-processing the SCIFACT
                     dataset for the AR task. It works similar as the training
                     dataset with the caveat that it does not need to be sampled
                     with negatives, etc. Essential for usage with torch.DataLoader.
    '''

    def __init__(self: 'SciFactDataset', dataset: str, corpus: str, top_k_docs_for_claims: list) -> None:
        '''
        Initialize the class to set up the dataset for retrieval.

            Parameters:
            -------------------------
            dataset : str
                The path to access the SciFact dataset fold.
            corpus : str
                The path to access the SciFact dataset corpus.
            top_k_docs_for_claims : list
                Top K=20 docs listed for each sample from the dataset fold.
        '''
        # Load the contents of corpus.
        self.corpus = {
            doc['doc_id']: doc
            for doc in jsonlines.open(corpus)
        }
        # TOP K=20 abstract docs for each claim from the dataset fold.
        self.top_k_docs_for_claims = top_k_docs_for_claims
        # Build dataset samples, using BM25 docs.
        self.samples = self.build_samples(dataset)

    def __len__(self: 'SciFactDataset') -> int:
        '''
        Get the length of the sampled dataset.

            Returns:
            -------------------------
            int
                The number of samples in the fold.
        '''
        return len(self.samples)

    def __getitem__(self: 'SciFactDataset', i: int) -> dict:
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

    def build_samples(self: 'SciFactDataset', dataset: str) -> list:
        '''
        Build the sample set for the fold using ranked docs.
        Unlike in the training script, there is no need for
        negative sampling.

            Parameters:
            -------------------------
            dataset : str
                The path to access the SciFact dataset fold.

            Returns:
            -------------------------
            samples : list
                The list of sampled dataset.
        '''
        samples = []
        for i, data in enumerate(jsonlines.open(dataset)):
            # Extract top 20 docs for the current claim (data item).
            data_top_docs = self.top_k_docs_for_claims[i]
            # Sample using the top K=20 docs.
            for doc_id in data_top_docs:
                # Include per sample: ID, claim, title.
                samples.append({
                    'id': data['id'],
                    'claim': data['claim'],
                    'title': self.corpus[doc_id]['title']
                })
        return samples

######################################

# Run the training for AR with BioBERT-base.
if __name__ == '__main__':
    biobert_retriever = BioBERTRetriever()
    biobert_retriever.run()
