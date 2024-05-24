'''
--------------------------------------------------------------
FILE:
    pipeline/eval/tools/predicted_dataset.py

NOTE:
    This module was applied from the work by Wadden et
    al. (2020) used in combination with other eval.tools
    to perform pipeline evaluation.

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
from dataclasses import dataclass
import json
from typing import Dict, List

# Imports for other tools from the package
from .label import *
from .gold_dataset import Claim

class PredictedDataset:
    '''
    PredictedDataset - class that represents predicted dataset with the pointer
                       to the gold data.
    '''

    def __init__(self: 'PredictedDataset', gold, prediction_file) -> None:
        '''
        Takes a GoldDataset, as well as files with rationale and label
        predictions.
        '''
        self.gold = gold
        self.predictions = self._read_predictions(prediction_file)

    def __getitem__(self: 'PredictedDataset', i):
        return self.predictions[i]

    def __repr__(self: 'PredictedDataset'):
        msg = f"Predictions for {len(self.predictions)} claims."
        return msg

    def _read_predictions(self: 'PredictedDataset', prediction_file):
        res = []
        predictions = [json.loads(line) for line in open(prediction_file)]
        for pred in predictions:
            prediction = self._parse_prediction(pred)
            res.append(prediction)
        return res

    def _parse_prediction(self: 'PredictedDataset', pred_dict):
        claim_id = pred_dict["id"]
        predicted_evidence = pred_dict["evidence"]
        res = {}
        # Predictions should never be NEI; there should only be predictions for
        # the abstracts that contain evidence.
        for key, this_prediction in predicted_evidence.items():
            label = this_prediction["label"]
            evidence = this_prediction["sentences"]
            pred = PredictedAbstract(int(key),
                                     map_label(label, include_nei=False),
                                     evidence)
            res[int(key)] = pred
        gold_claim = self.gold.get_claim_by_id(claim_id)
        return ClaimPredictions(claim_id, res, gold_claim)

@dataclass
class PredictedAbstract:
    '''
    -------------------------
    PredictedAbstract - data class to represent a predicted abstract.
                        For predictions, consider only a single list
                        of rationale sentences instead of a list of
                        separate rationales.
    -------------------------
    '''
    abstract_id: int
    label: Label
    rationale: List

@dataclass
class ClaimPredictions:
    '''
    -------------------------
    ClaimPredictions - data class to represent a multiple claim predictions.
    -------------------------
    '''
    claim_id: int
    predictions: Dict[int, PredictedAbstract]
    gold: Claim=None

    def __repr__(self):
        msg = f"Predictions for {self.claim_id}: {self.gold.claim}"
        return msg
