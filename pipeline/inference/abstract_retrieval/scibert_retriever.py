'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/scibert_retriever.py

NOTE:
    Prior to running this script you must have either trained
    the SciBERT model through:
    > bash script/train-abstract-retrieval.sh
    **OR**
    Used a previously fine-tuned model for this task, which would
    mean that you previously run the script listed above.

INFO:
    The script implements the SciBERTRetriever class that performs
    Abstract Retrieval in "SCIBERT" (i.e. HYBRID) setting. It first
    uses the BM25 algorithm to find 20 matching docs, that are
    further classified by SciBERT classifier to determine final picks.
    It inherits all methods from BioBERTRetriever, as it works
    exactly the same but loads a different model. This class was
    implemented for completeness/clean organization of resources.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    05/2024
--------------------------------------------------------------
'''

from biobert_retriever import BioBERTRetriever

class SciBERTRetriever(BioBERTRetriever):
    '''
    -------------------------
    SciBERTRetriever - Class for Abstract Retrieval in "SCIBERT" (HYBRID) setting.
                       It utilizes a fine-tuned SciBERT on SciFact.
                       Parent class: BioBERTRetriever.
    -------------------------
    '''

    def __init__(self: 'SciBERTRetriever') -> None:
        '''
        Initialize the class for AR in "SCIBERT" setting.
        '''
        super().__init__()
        # Over-write the retrieval strategy.
        self.retrieval_strategy = 'SCIBERT'

    def run(self: 'SciBERTRetriever') -> None:
        '''
        Run the "SCIBERT" retrieval by calling the classifier function
        and then writing the predictions into target output file.
        It is an over-written version of the parent class method for clarity.
        '''
        classify_res = self.classify()
        # Build the results by processing the classifier's results.
        for i, data in enumerate(self.dataset):
            self.retrieval_results[data['id']] = classify_res[i]
        self.write_results()
        print(f'Abstract Retrieval in "{self.retrieval_strategy}" setting: COMPLETE.\n')

# Run the training for AR with SciBERT-base.
if __name__ == '__main__':
    scibert_retriever = SciBERTRetriever()
    scibert_retriever.run()
