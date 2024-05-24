'''
--------------------------------------------------------------
FILE:
    pipeline/eval/evaluator.py

NOTE:
    This is an abstract class ("abstract" in relation to OOP
    architecture). It cannot be instantiated as-is, but rather
    must be used as parent for any of the children class, e.g.,
    AbstractRetrievalEvaluator, or PipelineEvaluator.

INFO:
    The script implements the Evaluator class, an interface
    class for all evaluators, that implements all re-usable
    functionalities, e.g., writing results to files. On top
    of that, it implements various re-usable metrics, such
    as: precision, recall, F1.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    03/2024
--------------------------------------------------------------
'''

from collections import Counter
import os
from .gold_dataset import GoldDataset
from typing import Tuple

class Evaluator:
    '''
    -------------------------
    Evaluator - an interface class for specific evaluators.
    -------------------------
    '''

    # Set base path for evaluation results.
    EVAL_PATH = 'results/'

    def __init__(self: 'Evaluator', res_eval_file: str, dataset: str) -> None:
        '''
        Initialize the interface for evaluation.

            Parameters:
            -------------------------
            res_eval_file : str
                Name of the file for eval results.
            dataset : str
                Name of the dataset fold.
        '''
        self.res_eval_file_path = f'{self.EVAL_PATH}{res_eval_file}.txt'
        # Build a gold dataset object to perform more robust eval.
        self.gold_dataset = GoldDataset(dataset)

    @classmethod
    def divide_safe(cls, numerator: int, denominator: int) -> float:
        '''
        Perform save division, handling scenarios with a zero in
        denominator.

            Parameters:
            -------------------------
            numerator : int
                The division numerator.
            denominator : int
                The division denominator.

            Returns:
            -------------------------
            res : float
                The division result.
        '''
        return 0 if (denominator == 0) else (numerator / denominator)

    @classmethod
    def calc_f1(cls, counter: Counter, mode=None) -> Tuple[float]:
        '''
        Calculate F1 and their associated metrics of PRECISION and RECALL,
        which signify:
            PRECISION - the ratio of correctly retrieved relevant docs to
                        the total number of retrieved docs.
            RECALL - the ratio of correctly predicted items to the total
                     number of relevant items
            F1 - the harmonic mean of PRECISION and RECALL.

            Parameters:
            -------------------------
            counter : collections.Counter
                Counter object with calculations over some data fold. It must
                contain the following keys: "retrieved", "correct", "relevant".
            mode : str (optional)
                Specified mode for "correct" key. If passed, then the key is
                customized as required for, e.g., "selection" in pipeline eval.

            Returns:
            -------------------------
            f1_results : Tuple[float]
                Full results for F1 and its metrics, i.e. PRECISION, RECALL.
        '''
        correct_key = 'correct' if mode is None else f'correct_{mode}'
        # PRECISION = TRUE POSITIVES / (TRUE POSITIVES + FALSE POSITIVES)
        precision = cls.divide_safe(counter[correct_key], counter['retrieved'])
        # RECALL = TRUE POSITIVES / (TRUE POSITIVES + FALSE NEGATIVES)
        recall = cls.divide_safe(counter[correct_key], counter['relevant'])
        # F1 = (2 * PRECISION * RECALL) / (PRECISION + RECALL)
        f1 = cls.divide_safe((2 * precision * recall), (precision + recall))
        return (precision, recall, f1)

    def write_eval_results(self: 'Evaluator', data: str) -> None:
        '''
        Save evaluation results into a TXT file. If the file with name
        already exists, then replace it with a new version.

            Parameters:
            -------------------------
            data : str
                Evalulation results data for TXT.
        '''
        # Ensure the directory exists.
        directory = os.path.dirname(self.res_eval_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Write the data into the file.
        with open(self.res_eval_file_path, 'w') as file:
            file.write(data)
