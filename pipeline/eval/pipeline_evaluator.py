'''
--------------------------------------------------------------
FILE:
    pipeline/eval/pipeline_evaluator.py

NOTE:
    This eval tool were author's re-implementation to handle
    data in evaluation, as interpreted from the Wadden et al.
    (2020) publication. They were inspired by Wadden et al.
    (2020) work and validated for consistency against their
    original implementations; by verifying the consistency
    of obtained results for full-pipeline. Copied functions
    within the module are annotated with respective source.

INFO:
    The script performs PIPELINE evaluation comparing the predicted
    results with the correct dataset outputs. The evaluation metrics
    include precision, recall and F1 on both abstract- and sentence-
    -level, as specified in Wadden et al. (2020). This implementation
    has been verified against their work for correctness. Any scenario
    of evaluation leads to a creation of an eval file in the directory:
    results/pipeline.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    05/2024
--------------------------------------------------------------
'''

# External imports.
import argparse
from collections import Counter
import pandas as pd
import warnings

# Imports for other tools from the package.
from tools.evaluator import Evaluator
from tools.gold_dataset import GoldDataset
from tools.label import Label
from tools.predicted_dataset import PredictedDataset

class PipelineEvaluator(Evaluator):
    '''
    PipelineEvaluator - class implements full-pipeline evaluation, including:
                        recall, precision, F1, both on abstract-/sentence-levels.
                        Parent class: Evaluator.
    '''

    # Specify the path of the results file for pipeline.
    PIPE_RESULT_PATH = 'prediction/merged_predictions.jsonl'

    # Cap on how many abstract sentences can be returned.
    MAX_ABSTRACT_SENTS = 3

    def __init__(self: 'PipelineEvaluator') -> None:
        '''
        Initialize the class for pipeline evaluation.
        '''
        self.pipe_eval_results = None
        # Handle CLI args to assign eval file name.
        self.args = self.set_args()
        self.model = self.args.model.upper()
        self.retrieval = self.args.retrieval.upper()
        self.dataset = self.args.dataset.upper()
        self.res_eval_file = f'PIPE_EVAL_{self.model}_{self.retrieval}_{self.dataset}'
        # Call the init of the parent Evaluator class with the pre-procesed params.
        super().__init__(
            res_eval_file=f'pipeline/{self.res_eval_file}',
            dataset=self.args.dataset
        )
        # Load gold dataset and the predicted one.
        self.data = GoldDataset(dataset=self.args.dataset)
        self.predictions = PredictedDataset(
            gold=self.data,
            prediction_file=self.PIPE_RESULT_PATH
        )
        # Initialize the eval results string.
        self.results = f'PIPELINE RESULTS for "{self.model.upper()}" model '
        self.results += f'with "{self.retrieval.upper()}" retrieval '
        self.results += f'on "{self.dataset.upper()}" dataset.\n\n'

    @classmethod
    def set_args(cls) -> argparse.Namespace:
        '''
        Handle parsing CLI args associated with the script.

            Returns:
            -------------------------
            args : argparse.Namespace
                Object contains pre-processed CLI args.
        '''
        parser = argparse.ArgumentParser()
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
            '--retrieval',
            type=str,
            required=True
        )
        return parser.parse_args()

    @classmethod
    def count_rationale_sents(cls, predicted, gold) -> int:
        '''
        Count rationale sentences for sentence-level evaluation.

        SOURCE: https://github.com/allenai/scifact.
        '''
        n_correct = 0
        for i in predicted:
            gold_sets = [entry for entry in gold if i in entry]
            # Can't be in two rationales.
            assert len(gold_sets) < 2
            # If it's not in a gold set, no dice.
            if len(gold_sets) == 0:
                continue
            # If it's in a gold set, make sure the rest got retrieved.
            gold_set = gold_sets[0]
            if gold_set.issubset(predicted):
                n_correct += 1
        return n_correct

    @classmethod
    def count_correct(cls, doc_id, doc_pred, gold) -> tuple:
        '''
        Count both correct selections and labels.

        SOURCE: https://github.com/allenai/scifact.
        '''
        # If not an evidence doc, no good.
        if doc_id not in gold.evidence:
            return 0, 0
        # Count the number of rationale sentences we get credit for.
        gold_rationales = [set(x) for x in gold.evidence[doc_id].rationales]
        n_correct = cls.count_rationale_sents(set(doc_pred.rationale), gold_rationales)
        gold_label = gold.evidence[doc_id].label
        n_correct_selection = n_correct
        correct_label = int(doc_pred.label == gold_label)
        n_correct_label = correct_label * n_correct
        return n_correct_selection, n_correct_label

    @classmethod
    def contains_evidence(cls, predicted, gold) -> bool:
        '''
        Check for evidence in abstract-level evaluations.

        SOURCE: https://github.com/allenai/scifact.
        '''
        # If any of gold are contained in predicted, we're good.
        for gold_rat in gold:
            if gold_rat.issubset(predicted):
                return True
        # Otherwise, no evidence.
        return False

    @classmethod
    def is_correct(cls, doc_id, doc_pred, gold) -> tuple:
        '''
        Check for correctness on abstract-level eval.

        SOURCE: https://github.com/allenai/scifact.
        '''
        pred_rationales = doc_pred.rationale[:cls.MAX_ABSTRACT_SENTS]
        # If it's not an evidence document, we lose.
        if doc_id not in gold.evidence:
            return False, False
        # If the label's wrong, we lose.
        gold_label = gold.evidence[doc_id].label
        if doc_pred.label != gold_label:
            return False, False
        gold_rationales = [set(x) for x in gold.evidence[doc_id].rationales]
        good_rationalized = cls.contains_evidence(set(pred_rationales), gold_rationales)
        good_label_only = True
        return good_label_only, good_rationalized

    def check_rationale_lengths(self: 'PipelineEvaluator') -> None:
        '''
        Validate length of rationales, making sure they are not
        longer than expected. If so, push a warning message.

        SOURCE: https://github.com/allenai/scifact.
        '''
        bad_rats = []
        for pred in self.predictions:
            claim_id = pred.claim_id
            predictions = pred.predictions
            for doc_key, prediction in predictions.items():
                n_rationales = len(prediction.rationale)
                if n_rationales > self.MAX_ABSTRACT_SENTS:
                    to_append = {
                        'claim_id': claim_id,
                        'abstract': doc_key,
                        'n_rationales': n_rationales
                    }
                    bad_rats.append(to_append)
        if bad_rats:
            bad_rats = pd.DataFrame(bad_rats)
            msg = (
                f'\nRationales with more than {self.MAX_ABSTRACT_SENTS} sentences found.\n'
                f'The first 3 will be used for abstract-level evaluation\n\n'
                f'{bad_rats.__repr__()}'
            )
            warnings.warn(msg)

    @classmethod
    def update_counts_abstract(cls, pred, gold, counts_abstract) -> Counter:
        '''
        Update the counter object for abstracts.

        SOURCE: https://github.com/allenai/scifact.
        '''
        counts_abstract['relevant'] += len(gold.evidence)
        for doc_id, doc_pred in pred.predictions.items():
            # If it's NEI, doesn't count one way or the other.
            if doc_pred.label == Label.NEI:
                continue
            counts_abstract['retrieved'] += 1
            good_label_only, good_rationalized = cls.is_correct(
                doc_id, doc_pred, gold
            )
            if good_label_only:
                counts_abstract['correct_label_only'] += 1
            if good_rationalized:
                counts_abstract['correct_rationalized'] += 1
        return counts_abstract

    @classmethod
    def update_counts_sentence(cls, pred, gold, counts_sentence) -> Counter:
        '''
        Update the counter object for sentence.

        SOURCE: https://github.com/allenai/scifact.
        '''
        # Update the gold evidence sentences.
        for gold_doc in gold.evidence.values():
            counts_sentence['relevant'] += sum([len(x) for x in gold_doc.rationales])
        for doc_id, doc_pred in pred.predictions.items():
            # If it's NEI, skip it.
            if doc_pred.label == Label.NEI:
                continue
            counts_sentence['retrieved'] += len(doc_pred.rationale)
            n_correct_selection, n_correct_label = cls.count_correct(
                doc_id,
                doc_pred,
                gold
            )
            counts_sentence['correct_selection'] += n_correct_selection
            counts_sentence['correct_label'] += n_correct_label
        return counts_sentence

    def compute_metrics(self: 'PipelineEvaluator') -> pd.DataFrame:
        '''
        Compute pipeline metrics based on the dataset of predictions.

        SOURCE: https://github.com/allenai/scifact.
        '''
        counts_abstract = Counter()
        counts_sentence = Counter()
        self.check_rationale_lengths()
        # Iterate for all predictions to eval on both sentence- and abstract-level.
        for pred in self.predictions:
            gold = self.data.get_claim_by_id(pred.claim_id)
            counts_abstract = self.update_counts_abstract(pred, gold, counts_abstract)
            counts_sentence = self.update_counts_sentence(pred, gold, counts_sentence)
        # Compute F1 results (through parent method) and build results outputs.
        ss_res = self.calc_f1(counts_sentence, 'selection')
        sl_res = self.calc_f1(counts_sentence, 'label')
        alo_res = self.calc_f1(counts_abstract, 'label_only')
        ar_res = self.calc_f1(counts_abstract, 'rationalized')
        sentence_selection_res = {
            'Precision': ss_res[0],
            'Recall': ss_res[1],
            'F1': ss_res[2]
        }
        sentence_label_res = {
            'Precision': sl_res[0],
            'Recall': sl_res[1],
            'F1': sl_res[2]
        }
        abstract_label_only_res = {
            'Precision': alo_res[0],
            'Recall': alo_res[1],
            'F1': alo_res[2]
        }
        abstract_rationalized_res = {
            'Precision': ar_res[0],
            'Recall': ar_res[1],
            'F1': ar_res[2]
        }
        pipe_eval_results = pd.DataFrame({
            'sentence_selection': sentence_selection_res,
            'sentence_label': sentence_label_res,
            'abstract_label_only': abstract_label_only_res,
            'abstract_rationalized': abstract_rationalized_res
        })
        return pipe_eval_results

    def run(self: 'PipelineEvaluator') -> None:
        '''
        Main function to run the full pipeline evaluation cycle.
        Also, write the results into a TXT file (through: write_eval_results).
        '''
        eval_res = self.compute_metrics()
        self.results += eval_res.to_string()
        self.write_eval_results(data=self.results)
        print(self.results)

# Perform full-pipeline evaluation.
if __name__ == '__main__':
    pipe_eval = PipelineEvaluator()
    pipe_eval.run()
