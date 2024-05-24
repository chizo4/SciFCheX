'''
--------------------------------------------------------------
FILE:
    pipeline/eval/abstract_retrieval_evaluator.py

INFO:
    The script performs abstract-retrieval-only (AR) evaluation
    comparing retrieved results with the correct dataset abstracts.
    The standard evaluation metrics of the AbstractRetrievalEvaluation
    class include: recall, precision, F1 (all inherited from parent
    class), and hit-one, hit-all (implemented by the class). On top of
    that, the script contains a function enabling recall experiments
    for different K values for retriever of choice. Any scenario of
    evaluation leads to a creation of an eval file in the directory:
    results/abstract_retrieval.

AUTHOR:
    Filip J. Cierkosz

VERSION:
    04/2024
--------------------------------------------------------------
'''

import argparse
from collections import Counter
import jsonlines
from tools.evaluator import Evaluator

class AbstractRetrievalEvaluator(Evaluator):
    '''
    AbstractRetrievalEvaluator - class implements AR evaluation, including
                                 recall (@K), precision, F1, hits (one/all).
                                 Parent class: Evaluator.
    '''

    # Specify the path of the retrieval result file.
    AR_RESULT_PATH = 'prediction/abstract_retrieval.jsonl'

    def __init__(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Initialize the class for AR evaluation.
        '''
        # Handle CLI args to assign eval file name, dataset (with path), etc.
        self.args = self.set_args()
        self.retrieval = self.args.retrieval.upper()
        self.dataset = self.args.dataset.upper()
        # Handle eval mode for standard eval/recall@k experiments.
        self.init_recall_mode() if self.args.recall else self.init_standard_mode()
        # Call the init of the parent Evaluator class with the pre-procesed params.
        super().__init__(
            res_eval_file=f'abstract_retrieval/{self.res_eval_file}',
            dataset=self.args.dataset
        )
        # Load gold dataset docs into dict and the predicted results.
        self.gold_docs = {
            claim.id: list(claim.evidence.keys())
            for claim in self.gold_dataset.claims
        }
        self.predict_docs = list(jsonlines.open(self.AR_RESULT_PATH))
        # Counter object, required for all eval modes.
        self.counter = Counter()

    def init_standard_mode(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Handle initializing variables associated with standard evaluation metrics.
        '''
        self.res_eval_file = f'AR_EVAL_{self.retrieval}_{self.dataset}'
        # Assign standard eval metrics. They are accumulated over the data fold.
        self.eval_mode = 'standard'
        self.hit_one = 0
        self.hit_all = 0
        self.recall = 0
        self.precision = 0
        self.f1 = 0
        # Initialize the results string with basic info about AR method.
        self.results = f'RESULTS for: "{self.retrieval.upper()}" retrieval '
        self.results += f'on "{self.dataset.upper()}" dataset'
        self.results += ' with K=3.' if self.retrieval in ['BM25', 'TFIDF'] else '.'

    def init_recall_mode(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Handle initializing variables associated with RECALL@K metrics.
        '''
        self.res_eval_file = f'AR_RECALL@K_{self.retrieval}_{self.dataset}'
        # Assign non-standard eval metrics.
        self.eval_mode = 'recall@k'
        self.recall_at_1 = 0
        self.recall_at_3 = 0
        self.recall_at_5 = 0
        self.recall_at_10 = 0
        self.results = f'RECALL@K for "{self.retrieval.upper()}" retrieval '
        self.results += f'on "{self.dataset.upper()}" dataset.'

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
            '--dataset',
            type=str,
            required=True
        )
        parser.add_argument(
            '--recall',
            type=str,
            required=False
        )
        parser.add_argument(
            '--retrieval',
            type=str,
            required=True
        )
        return parser.parse_args()

    @classmethod
    def calc_hit_one(cls, predict_docs: set, true_docs: set) -> int:
        '''
        Calculate HIT-ONE, which: considers a retrieval successful
        if there is at least one relevant retrieval doc when compared
        with the golden (true) ones.

            Parameters:
            -------------------------
            predict_docs : set
                Set of the predicted docs.
            true_docs : set
                Set of all the true docs.

            Returns:
            -------------------------
            int
                1 - if hits one. 0 - otherwise.
        '''
        if predict_docs.intersection(true_docs) or not true_docs:
            return 1
        return 0

    @classmethod
    def calc_hit_all(cls, predict_docs: set, true_docs: set) -> int:
        '''
        Calculate HIT-ALL, which: requires all retrievals to be consistent
        with golden (true) retrieval docs to be successful.

            Parameters:
            -------------------------
            predict_docs : set
                Set of the predicted docs.
            true_docs : set
                Set of all the true docs.

            Returns:
            -------------------------
            int
                1 - if hits one. 0 - otherwise.
        '''
        if predict_docs.issuperset(true_docs):
            return 1
        return 0

    def eval(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Perform evaluation for AR using all standard metrics over the
        dataset. They include: precision, recall, F1, hit-one, hit-all.
        '''
        # Calculate standard eval metrics for predictions for each sample.
        for prediction in self.predict_docs:
            claim_id = prediction['claim_id']
            # Set the predicted docs and the true ones, as in the dataset.
            predict_doc_ids = prediction['doc_ids']
            true_doc_ids = set(self.gold_docs[claim_id])
            # Update hit metrics.
            self.hit_one += self.calc_hit_one(set(predict_doc_ids), true_doc_ids)
            self.hit_all += self.calc_hit_all(set(predict_doc_ids), true_doc_ids)
            # Update the counter object for further P, R, F1 calculations.
            self.counter['relevant'] += len(true_doc_ids)
            for prediction in predict_doc_ids:
                self.counter['retrieved'] += 1
                if prediction in true_doc_ids:
                    self.counter['correct'] += 1
        # Finally, calculate P, R, F1 cumulative metrics (using parent's method).
        (self.precision, self.recall, self.f1) = self.calc_f1(self.counter)
        # Normalize hits metrics, by division over dataset size.
        self.hit_one = self.hit_one / len(list(self.gold_docs.keys()))
        self.hit_all = self.hit_all / len(list(self.gold_docs.keys()))

    def calc_recall_at_k(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Additional experiment to compute the RECALL@K metrics for
        retrievers. K values include: R@1, R@3, R@5, R@10.
        '''
        for prediction in self.predict_docs:
            claim_id = prediction['claim_id']
            # Set the predicted docs and the true ones, as in the dataset.
            predict_doc_ids = prediction['doc_ids']
            true_doc_ids = set(self.gold_docs[claim_id])
            # Specify num of relevant docs.
            self.counter['relevant'] += len(true_doc_ids)
            # Update RECALL@K, where K: 1, 3, 5, 10.
            self.counter['correct@1'] += sum(1 for p in predict_doc_ids[:1] if p in true_doc_ids)
            self.counter['correct@3'] += sum(1 for p in predict_doc_ids[:3] if p in true_doc_ids)
            self.counter['correct@5'] += sum(1 for p in predict_doc_ids[:5] if p in true_doc_ids)
            self.counter['correct@10'] += sum(1 for p in predict_doc_ids[:10] if p in true_doc_ids)
            self.counter['correct@20'] += sum(1 for p in predict_doc_ids[:20] if p in true_doc_ids)
        # Eventually, compute cumulative RECALL@K and display.
        self.recall_at_1 = self.divide_safe(self.counter['correct@1'], self.counter['relevant'])
        self.recall_at_3 = self.divide_safe(self.counter['correct@3'], self.counter['relevant'])
        self.recall_at_5 = self.divide_safe(self.counter['correct@5'], self.counter['relevant'])
        self.recall_at_10 = self.divide_safe(self.counter['correct@10'], self.counter['relevant'])
        self.recall_at_20 = self.divide_safe(self.counter['correct@20'], self.counter['relevant'])

    def set_eval_results(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Add metrics results into the results data string, that will be
        further written into a TXT file. Adjusted respectively to eval mode.
        '''
        if self.eval_mode != 'recall@k':
            self.results += f'\n\tHit-one: {self.hit_one}\n\tHit-all: {self.hit_all}'
            self.results += f'\n\tRECALL: {self.recall}\n\tPRECISION: {self.precision}\n\tF1: {self.f1}'
        else:
            self.results += f'\n\tRECALL@1: {self.recall_at_1}'
            self.results += f'\n\tRECALL@3: {self.recall_at_3}'
            self.results += f'\n\tRECALL@5: {self.recall_at_5}'
            self.results += f'\n\tRECALL@10: {self.recall_at_10}'
            self.results += f'\n\tRECALL@20: {self.recall_at_20}'

    def run(self: 'AbstractRetrievalEvaluator') -> None:
        '''
        Main function to run the full evaluation cycle depending on eval mode:
        through eval() for "standard", whereas through recall_at_k() for "recall@k".
        Also, write the results into a TXT file (through: write_eval_results).
        '''
        if self.eval_mode != 'recall@k':
            self.eval()
        else:
            self.calc_recall_at_k()
        self.set_eval_results()
        self.write_eval_results(data=self.results)
        print(self.results)

if __name__ == '__main__':
    ar_eval = AbstractRetrievalEvaluator()
    ar_eval.run()
