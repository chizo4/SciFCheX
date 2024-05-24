'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/bge_m3_retriever.py

INFO:
    The script implements the BGEM3Retriever class that
    performs Abstract Retrieval in "BGE-M3" (i.e. HYBRID) setting.
    It first uses the BM25 algorithm to find 20 matching docs, that
    are further reranked by a SOTA BGE M3 reranker to determine
    final picks (at most 3).

AUTHOR:
    Filip J. Cierkosz

VERSION:
    05/2024
--------------------------------------------------------------
'''

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from tools.retriever import Retriever
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BGEM3Retriever(Retriever):
    '''
    -------------------------
    BGEM3Retriever - Class for Abstract Retrieval in "HYBRID" setting.
                      Parent class: Retriever.
    -------------------------
    '''

    def __init__(self: 'BGEM3Retriever') -> None:
        '''
        Initialize the class for AR in "BGE-M3" setting.
        '''
        super().__init__(retrieval_strategy='BGE_M3')
        # BM25: Build tokenizer, tokenize corpus, and load the model.
        self.tokenizer_bm25 = CountVectorizer(stop_words='english').build_analyzer()
        self.corpus_tokens = [
            self.tokenizer_bm25(doc['title'] + ' ' + ' '.join(doc['abstract']))
            for doc in self.corpus
        ]
        self.model_bm25 = BM25Okapi(self.corpus_tokens)
        # BGE M3: Load the reranker. This version is lightweight with fast inference.
        self.tokenizer_bge = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.model_bge = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
        # Set up the device. Run the model on GPU (if available).
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\nUsing device "{self.device}"\n')
        self.model = self.model_bge.to(self.device)

    @classmethod
    def normalize_scores(cls, scores: torch.Tensor) -> torch.Tensor:
        '''
        Normalize raw model's scores into range: [0, 1].

            Parameters:
            -------------------------
            scores : torch.Tensor
                Raw reranker scores to be normalized.

            Returns:
            -------------------------
            normal_scores : torch.Tensor
                Normalized scores from reranking.
        '''
        min_scores = torch.min(scores)
        max_scores = torch.max(scores)
        normal_scores = (scores - min_scores) / (max_scores - min_scores)
        return normal_scores

    @classmethod
    def score_drop_off(cls, scores: torch.Tensor, threshold=0.05):
        '''
        Determine the cutoff index where scores drop significantly.

            Parameters:
            -------------------------
            scores : torch.Tensor
                Normalized scores from the reranker.
            threshold : float, default=0.05
                Percentage drop that signifies a significant drop-off.

            Returns:
            -------------------------
            int
                Index to cut off document inclusion.
        '''
        for i in range(1, len(scores)):
            if (scores[i-1] - scores[i]) / scores[i-1] > threshold:
                return i
        return len(scores)

    def rerank_neural(self: 'BGEM3Retriever', claim: str, docs: list, K_BGE=3) -> np.ndarray:
        '''
        Perform neural reranking with BGE M3 reranker for N docs
        retrieved by BM25 to find K most matching docs.

            Parameters:
            -------------------------
            claim : str
                The claim for which docs are to be reranked.
            docs : list
                List of candidate abstracts to be reranked in reference to claim.
            K_RERANK : int, default=3
                Maximum number of top candidates to be selected after reranking.

            Returns:
            -------------------------
            top_rerank_docs : np.ndarray
                Indices of (at most) top K documents after reranking.
        '''
        # Prepare (claim, doc) pairs for evaluation.
        pairs = [[claim, doc] for doc in docs]
        # Evaluate pairs to find the top docs.
        with torch.no_grad():
            inputs = self.tokenizer_bge(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            # Convert inputs to GPU.
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze()
        # Normalize scores to range 0-1.
        normal_scores = self.normalize_scores(scores)
        # Analyze score drop-offs to determine cutoff for top matches.
        cutoff_index = self.score_drop_off(normal_scores)
        # Re-assign K if it was forced in CLI args (recall@k experiments).
        if self.args.k:
            final_index = self.args.k
        # Otherwise, limit to either drop-off index or K, whichever is smaller.
        else:
            final_index = min(cutoff_index, K_BGE)
        top_rerank_docs = normal_scores.argsort(descending=True)[:final_index]
        return top_rerank_docs.cpu().numpy()

    def rank_hybrid(self: 'BGEM3Retriever', N_BM25=20) -> None:
        '''
        Perform HYBRID ranking of abstracts for each sample in the
        dataset. The pipeline consists of BM25 that first retrieves the top
        20 abstracts, that are further re-ranked by BGE RERANKER to decide
        on 3 best matching docs. Finally, write outputs into a target file.

            Parameters:
            -------------------------
            N_BM25 : int, default=20
                Number of top for BM25 that are passed for reranking.
        '''
        # Perform ranking of documents for each of the claim sample.
        it = 0
        for data in self.dataset:
            it += 1
            print(f'Reranking for claim: [{it}/{len(self.dataset)}].')
            # Set the current claim and tokenize for BM25.
            claim = data['claim']
            claim_tokens = self.tokenizer_bm25(claim)
            doc_scores = self.model_bm25.get_scores(claim_tokens)
            # Prepare N best docs found for the BM25 stage.
            top_indices_bm25 = np.argsort(doc_scores)[::-1][:N_BM25]
            # Merge doc titles and abstracts for each of top N candidates from BM25.
            top_docs_bm25 = [
                self.corpus[i]['title'] + ' '.join(self.corpus[i]['abstract'])
                for i in top_indices_bm25
            ]
            # Perform reranking with BGE M3 for N docs to find the top K docs.
            reranked_indices = self.rerank_neural(claim, top_docs_bm25)
            top_docs_rerank = [
                self.corpus[top_indices_bm25[i]]['doc_id']
                for i in reranked_indices.tolist()
            ]
            self.retrieval_results[data['id']] = top_docs_rerank

    def run(self: 'BGEM3Retriever') -> None:
        '''
        Run the "BGE-M3" retrieval by calling the ranking function
        and then writing the predictions into target file.
        '''
        self.rank_hybrid()
        self.write_results()
        print(f'Abstract Retrieval in "{self.retrieval_strategy}" setting: COMPLETE.\n')

# Perform BGE-M3 (HYBRID) abstract retrieval.
if __name__ == '__main__':
    bge_m3_retriever = BGEM3Retriever()
    bge_m3_retriever.run()
