'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/bm25_retriever.py

INFO:
    The script implements the BM25Retrieval class performing
    the BM25 Abstract Retrieval. It uses the BM25 algorithm
    to find 3 matching docs. This AR approach is used in
    combination with a neural reranker in hybrid retrievers
    for the most optimal version of SCIFCHEX.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    04/2024
--------------------------------------------------------------
'''

import numpy as np
from rank_bm25 import BM25Okapi
from tools.retriever import Retriever
from sklearn.feature_extraction.text import CountVectorizer

class BM25Retriever(Retriever):
    '''
    -------------------------
    BM25Retriever - Class for Abstract Retrieval in "BM25" setting.
                    Parent class: Retriever.
    -------------------------
    '''

    def __init__(self: 'BM25Retriever') -> None:
        '''
        Initialize the class for AR in "BM25" setting.
        '''
        super().__init__(retrieval_strategy='BM25')
        # Build tokenizer, tokenize corpus, and load BM25 model.
        self.tokenizer = CountVectorizer(stop_words='english').build_analyzer()
        self.corpus_tokens = [
            self.tokenizer(doc['title'] + ' ' + ' '.join(doc['abstract'])) for doc in self.corpus
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def rank_bm25(self: 'BM25Retriever', K=3) -> None:
        '''
        Perform the BM25 ranking algorithm to retrieve the top 3 abstracts.

            Parameters:
            -------------------------
            K : int, default=3
                Number of top docs retrieved for BM25.
        '''
        # Re-assign K if it was forced in CLI args.
        if self.args.k:
            K = self.args.k
        # Perform ranking of documents for each of the claim sample.
        for data in self.dataset:
            # Set the current claim and tokenize.
            claim = data['claim']
            claim_tokens = self.tokenizer(claim)
            doc_scores = self.bm25.get_scores(claim_tokens)
            # Prepare N best docs found for the BM25 stage.
            top_indices = np.argsort(doc_scores)[::-1][:K]
            final_docs = [self.corpus[idx]['doc_id'] for idx in top_indices]
            # Update the result dict with the current claim and its docs.
            self.retrieval_results[data['id']] = final_docs

    def run(self: 'BM25Retriever') -> None:
        '''
        Run the "BM25" retrieval by calling the ranking function
        and then writing the predictions into target file.
        '''
        self.rank_bm25()
        self.write_results()
        print(f'Abstract Retrieval in "{self.retrieval_strategy}" setting: COMPLETE.\n')

# Perform BM25 abstract retrieval.
if __name__ == '__main__':
    bm25 = BM25Retriever()
    bm25.run()
