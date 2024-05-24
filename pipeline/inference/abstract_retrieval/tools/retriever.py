'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/retriever.py

NOTE:
    This is an abstract class ("abstract" in relation to OOP
    architecture). It cannot be instantiated as-is, but rather
    must be used as parent for any of the children class, e.g.,
    BGEM3Retriever, OracleRetriever, etc.

INFO:
    The script implements the Retriever class, an interface
    class for all abstract retrieval types (BM25, TFIDF, etc.),
    that implements all re-usable functionalities and tools.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

import argparse
import jsonlines

class Retriever:
    '''
    -------------------------
    Retriever - interface class for all retrieval types handling
                DRY tools, prediction file, CLI args, etc.
    -------------------------
    '''

    # Specify the path of the retrieval result file.
    AR_RESULT_PATH = 'prediction/abstract_retrieval.jsonl'
    # Specify the path to access the corpus of abstracts.
    CORPUS_PATH = 'data/corpus.jsonl'

    def __init__(self: 'Retriever', retrieval_strategy: str) -> None:
        '''
        Initialize the interface for abstract retrieval.

            Parameters:
            -------------------------
            retrieval_strategy : str
                Name of the AR strategy (e.g., "BM25").
        '''
        self.retrieval_strategy = retrieval_strategy
        # Store retrievals in dict format: {claim_id: [doc_ids]}
        self.retrieval_results = {}
        # Handle CLI args.
        self.args = self.set_args()
        # Load dataset and corpus files.
        self.dataset = list(jsonlines.open(self.args.dataset))
        self.corpus = list(jsonlines.open(self.CORPUS_PATH))

    def set_args(self: 'Retriever') -> argparse.Namespace:
        '''
        Handle parsing CLI args associated with the script.

            Returns:
            -------------------------
            args : argparse.Namespace
                Object contains pre-processed CLI args.
        '''
        parser = argparse.ArgumentParser()
        if self.retrieval_strategy == 'ORACLE':
            parser.add_argument(
                '--include-nei',
                action='store_true'
            )
        elif self.retrieval_strategy in ['BIOBERT', 'SCIBERT', 'BERT']:
            # Handle pre-trained/fine-tuned version of BERT-like
            # transformer in a child class.
            parser.add_argument(
                '--model',
                type=str,
                required=True
            )
        parser.add_argument(
            '--dataset',
            type=str,
            required=True
        )
        parser.add_argument(
            '--k',
            type=int,
            required=False
        )
        return parser.parse_args()

    def write_results(self: 'Retriever') -> None:
        '''
        Write prediction results into the result file.
        '''
        # Write outputs into the target file.
        res_file = jsonlines.open(self.AR_RESULT_PATH, 'w')
        # If "oracle" setting, then pass dataset retrievals.
        if self.retrieval_strategy == 'ORACLE':
            for data in self.dataset:
                doc_ids = list(map(int, data['evidence'].keys()))
                if not doc_ids and self.args.include_nei:
                    doc_ids = [data['cited_doc_ids'][0]]
                res_file.write({
                    'claim_id': data['id'],
                    'doc_ids': doc_ids
                })
        else:
            # Otherwise, consider the actual prediction results.
            for claim_id, doc_ids in self.retrieval_results.items():
                res_file.write({
                    'claim_id': claim_id,
                    'doc_ids': doc_ids
                })
