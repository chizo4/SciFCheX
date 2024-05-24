'''
--------------------------------------------------------------
FILE:
    pipeline/inference/abstract_retrieval/oracle_retriever.py

INFO:
    The script implements the OracleRetriever class for Abstract
    Retrieval in "oracle" setting, meaning there is no actual
    retrieval, but rather provides further stages of the
    pipeline with gold abstracts, as they are in the dataset.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

from tools.retriever import Retriever

class OracleRetriever(Retriever):
    '''
    -------------------------
    OracleRetriever - Class for Abstract Retrieval in "oracle" setting.
                      Parent class: Retriever.
    -------------------------
    '''

    def __init__(self: 'OracleRetriever') -> None:
        '''
        Initialize the class for AR in "oracle" setting.
        '''
        super().__init__(retrieval_strategy='ORACLE')

    def run(self: 'OracleRetriever') -> None:
        '''
        Perform the "oracle" setting by writing results.
        '''
        self.write_results()
        print(f'Abstract Retrieval in "{self.retrieval_strategy}" setting: COMPLETE.\n')

# Oracle simply provides gold abstracts.
if __name__ == '__main__':
    oracle = OracleRetriever()
    oracle.run()
