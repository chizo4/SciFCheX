'''
--------------------------------------------------------------
FILE:
    pipeline/eval/tools/gold_dataset.py

NOTE:
    These data-related tools were author's re-implementation
    to handle data in evaluation, as interpreted from the
    Wadden et al. (2020) publication. They were inspired by
    Wadden et al. (2020) work and validated for consistency
    against their original implementations; by verifying the
    consistency of obtained results for full-pipeline.

INFO:
    The script implements key data-related classes, enabling to
    load data from e.g. a gold dataset into objects, so that the
    results can be processed more easily in evaluation steps.
    It also contains key data classes to represent claims.

ADJUSTED BY:
    Filip J. Cierkosz

SOURCE:
    https://github.com/allenai/scifact

VERSION:
    03/2024
--------------------------------------------------------------
'''

# External imports.
import copy
from dataclasses import dataclass
import json
from typing import Dict, List

# Imports for other tools from the package.
from .corpus import Corpus, Document
from .label import Label, map_label

class GoldDataset:
    '''
    GoldDataset - class that represents gold dataset, including its
                  claims, corpus and labels.
    '''

    def __init__(self: 'GoldDataset', dataset: str) -> None:
        '''
        Initialize the gold dataset object, loading the corpus and its claims.

            Parameters:
            -------------------------
            dataset : str
                Name of the dataset fold to load for.
        '''
        self.corpus = Corpus.load_jsonl()
        self.dataset = dataset
        self.dataset_path = f'data/claims_{self.dataset}.jsonl'
        self.claims = self.load_claims(self.dataset_path)

    def __repr__(self: 'GoldDataset') -> str:
        '''
        Gold dataset object representation in the string format.

            Returns:
            -------------------------
            str
                Gold Dataset string representation.
        '''
        return f'Gold Dataset for "{self.dataset}". Contains {len(self.claims)} claims.'

    def __getitem__(self: 'GoldDataset', i: int) -> object:
        '''
        Find a claim by index.

            Parameters:
            -------------------------
            i : int
                Claim index.

            Returns:
            -------------------------
            Claim
                The claim object found for index.
        '''
        return self.claims[i]

    def load_claims(self: 'GoldDataset', dataset_path: str) -> List:
        '''
        Handle loading claims from a dataset fold.

            Parameters:
            -------------------------
            dataset_path : str
                Path for the dataset fold to load from.

            Returns:
            claims : List[Claim]
                The loaded list of claims.
        '''
        dataset = [json.loads(line) for line in open(dataset_path)]
        claims = []
        # Build a claim for each dataset entry.
        for data in dataset:
            entry = copy.deepcopy(data)
            entry['release'] = self
            # Load cited docs for each claim.
            entry['cited_docs'] = [
                self.corpus.get_doc_by_doc_id(doc) for doc in entry['cited_doc_ids']
            ]
            assert len(entry['cited_docs']) == len(entry['cited_doc_ids'])
            del entry['cited_doc_ids']
            claims.append(Claim(**entry))
        # Sort the claims by ID and return
        claims = sorted(claims, key=lambda c: c.id)
        return claims

    def get_claim_by_id(self: 'GoldDataset', claim_id: str) -> object:
        '''
        Find a claim by its ID.

            Parameters:
            -------------------------
            claim_id : int
                ID of the claim to be found.

            Returns:
            -------------------------
            claim : Claim
                The claim found for the passed ID.
        '''
        claim_res = [claim for claim in self.claims if claim.id == claim_id]
        # Make sure no more than one doc can be found for ID.
        assert len(claim_res) == 1
        return claim_res[0]

@dataclass
class EvidenceAbstract:
    '''
    -------------------------
    EvidenceAbstract - data class to represent an evidence abstract.
    -------------------------
    '''
    id: int
    label: Label
    rationales: List[List[int]]

@dataclass(repr=False)
class Claim:
    '''
    -------------------------
    Claim - data class to represent a single claim from dataset fold.
    -------------------------
    '''
    id: int
    claim: str
    evidence: Dict[int, EvidenceAbstract]
    cited_docs: List[Document]
    release: GoldDataset

    def __repr__(self: 'Claim') -> str:
        '''
        Claim object representation in the string format.

            Returns:
            -------------------------
            str
                Gold Dataset string representation.
        '''
        return f'Claim (ID: {self.id}): {self.claim}'

    def __post_init__(self: 'Claim') -> None:
        '''
        Handle post-initialization operations for a claim to
        load evidence in an appropriate format.
        '''
        self.evidence = self.format_evidence(self.evidence)

    @staticmethod
    def format_evidence(evidence_dict: dict) -> dict:
        '''
        Helper function to handle storing labels on the "abstract level"
        rather than "rationale level".

        Source: https://github.com/allenai/scifact

            Parameters:
            -------------------------
            evidence_dict : dict
                The initial evidence dictionary.

            Returns:
            -------------------------
            adj_evidence_dict : dict
                The adjusted evidence ditionary.
        '''
        adj_evidence_dict = {}
        for doc_id, rationales in evidence_dict.items():
            doc_id = int(doc_id)
            labels = [rat['label'] for rat in rationales]
            if len(set(labels)) > 1:
                msg = ('In this SciFact release, each claim / abstract pair should only have one label.')
                raise Exception(msg)
            label = map_label(labels[0])
            rationale_sents = [rat['sentences'] for rat in rationales]
            abstract = EvidenceAbstract(doc_id, label, rationale_sents)
            adj_evidence_dict[doc_id] = abstract
        return adj_evidence_dict
