'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/tfidf_retriever.py

INFO:
    The script implements the TfidfRetriever class for Abstract
    Retrieval in "TFIDF" setting. It is a new implementation of
    the original sparse retrieval method from VERISCI. Added in
    order to verify that this implementation leads to the same
    results, as it was reported in VERISCI.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

import numpy as np
from tools.retriever import Retriever
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfRetriever(Retriever):
    '''
    -------------------------
    TfidfRetriever - Class for Abstract Retrieval in "TFIDF" setting.
                     Parent class: Retriever.
    -------------------------
    '''

    def __init__(self: 'TfidfRetriever') -> None:
        '''
        Initialize the class for AR in "TFIDF" setting.
        '''
        super().__init__(retrieval_strategy='TFIDF')

    def rank_tfidf(self: 'TfidfRetriever', K=3, min_gram=1, max_gram=2) -> None:
        '''
        Perform TF-IDF ranking of abstracts for each sample in the
        dataset. Retrieve K abstracts for each sample claim.

            Parameters:
            -------------------------
            K : int, default=3
                Number of abstracts to be retrieved.
            min_gram : int, default=1
                Minimum size for N-gram.
            max_gram : int, default=2
                Maximum size for N-gram.
        '''
        # Re-assign K if it was forced in CLI args.
        if self.args.k:
            K = self.args.k
        # Set TF-IDF vectorizer and transform each abstract into a vector.
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(min_gram, max_gram)
        )
        doc_vecs = vectorizer.fit_transform(
            [doc['title'] + ' '.join(doc['abstract']) for doc in self.corpus]
        )
        # Perform ranking of documents for each of the claim sample.
        for data in self.dataset:
            # Set the current claim and convert into a vector.
            claim = data['claim']
            claim_vec = vectorizer.transform([claim]).todense()
            # Rank the docs for the claim.
            doc_scores = np.asarray(doc_vecs @ claim_vec.T).squeeze()
            doc_indices_rank = doc_scores.argsort()[::-1].tolist()
            doc_id_rank = [self.corpus[idx]['doc_id'] for idx in doc_indices_rank]
            # Update the result dict with the current claim and its docs.
            self.retrieval_results[data['id']] = doc_id_rank[:K]

    def run(self: 'TfidfRetriever') -> None:
        '''
        Run the "TFIDF" retrieval by calling ranking function
        and then writing the predictions into target file.
        '''
        self.rank_tfidf()
        self.write_results()
        print(f'Abstract Retrieval in "{self.retrieval_strategy}" setting: COMPLETE.\n')

# Perform TF-IDF abstract retrieval.
if __name__ == '__main__':
    tfidf = TfidfRetriever()
    tfidf.run()
