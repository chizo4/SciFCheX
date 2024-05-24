'''
--------------------------------------------------------------
FILE:
    pipeline/eval/tools/corpus.py

INFO:
    The script implements data-classes, and their respective tools,
    to handle loading corpus data into objects. Specifically, to
    load corpus and its documents.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

from dataclasses import dataclass
import json
from typing import List, Tuple

@dataclass
class Document:
    '''
    -------------------------
    Document - data class to represent a single abstract document
               from the corpus.
    -------------------------
    '''
    id: str
    title: str
    sentences: Tuple[str]

    def __repr__(self: 'Document') -> str:
        '''
        Document object representation in the string format.

            Returns:
            -------------------------
            doc_str : str
                Document object string representation.
        '''
        doc_str = f'{self.title.upper()}\n' + '\n'.join(['- ' + sen for sen in self.sentences])
        return doc_str

    def __lt__(self: 'Document', other: 'Document') -> bool:
        '''
        Handle less then operator for document comparison.
        '''
        return self.title.__lt__(other.title)

@dataclass
class Corpus:
    '''
    -------------------------
    Corpus - data class to represent a collection of Documents.
    -------------------------
    '''
    documents: List[Document]

    @classmethod
    def load_jsonl(cls, corpus_path='data/corpus.jsonl') -> List[Document]:
        '''
        Handle loading corpus documents from jsonl file into an object.

            Parameters:
            -------------------------
            corpus_path : str, default='data/corpus.jsonl'
                The corpus path to load from.

            Returns:
            -------------------------
            docs : List[Document]
                The full-list of documents load from the corpus.
        '''
        corpus = [json.loads(line) for line in open(corpus_path)]
        docs = []
        # Build a document for each corpus entry.
        for data in corpus:
            doc = Document(
                data['doc_id'],
                data['title'],
                data['abstract']
            )
            docs.append(doc)
        return cls(docs)

    def __repr__(self: 'Corpus') -> str:
        '''
        Corpus object representation in the string format.

            Returns:
            -------------------------
            str
                Corpus object string representation.
        '''
        return f'Corpus of {len(self.documents)} docs.'

    def __getitem__(self: 'Corpus', i: int) -> Document:
        '''
        Find doc in corpus by its index.

            Parameters:
            -------------------------
            i : int
                Document index.

            Returns:
            -------------------------
            Document
                The document found for a specified index.
        '''
        return self.documents[i]

    def get_doc_by_doc_id(self: 'Corpus', doc_id: int) -> Document:
        '''
        Find a document by the document ID.

            Parameters:
            -------------------------
            doc_id : int
                ID of the document to be found.

            Returns:
            -------------------------
            doc : Document
                The document found for ID.
        '''
        doc_res = [doc for doc in self.documents if doc.id == doc_id]
        # Make sure no more than one doc can be found for ID.
        assert len(doc_res) == 1
        return doc_res[0]
