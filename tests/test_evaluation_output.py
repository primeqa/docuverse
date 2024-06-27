import unittest
import math
import itertools
import operator
from docuverse.utils.evaluation_output import EvaluationOutput

class TestEvaluationOutput(unittest.TestCase):

    def setUp(self):
        self.num_ranked_queries = 1
        self.num_judged_queries = 1
        self.doc_scores = [[1]]
        self.ranks = [2]
        self.model_name = "model"
        self.rouge_scores = None
        self.metrics = "match"

        self.evaluationOutputObj = EvaluationOutput(
            self.num_ranked_queries, 
            self.num_judged_queries, 
            self.doc_scores,
            self.ranks,
            self.model_name,
            self.rouge_scores,
            metrics=self.metrics
        )

    def test_model_name(self):
        self.assertEqual(self.evaluationOutputObj.model_name, self.model_name)

    def test_num_ranked_queries(self):
        self.assertEqual(self.evaluationOutputObj.num_ranked_queries, self.num_ranked_queries)

    def test_num_judged_queries(self):
        self.assertEqual(self.evaluationOutputObj.num_judged_queries, self.num_judged_queries)

    def test_compute_metrics(self):
        self.evaluationOutputObj.compute_metrics()
        self.assertEqual(self.evaluationOutputObj.ndcg, {2: 0.6309297535714574})
        self.assertEqual(self.evaluationOutputObj.match, {2: 0})
        self.assertEqual(self.evaluationOutputObj.mrr, {2: 0.0})

    def test_str_method(self):
        expected_str = "Model    M@\n" + \
                       "model    0.0      \n"
        self.assertEqual(self.evaluationOutputObj.__str__(), expected_str)

if __name__ == '__main__':
    unittest.main()