import math
import itertools
import operator
from docuverse.utils.ece_brier.ece import expected_calibration_error
from docuverse.utils.ece_brier.brier_score import brier_score

class EvaluationOutput:
    mappable_metrics = ['match', 'mrr', 'ndcg', 'map']
    single_metrics = ['ece', 'brier']
    def __init__(self, num_ranked_queries, num_judged_queries,
                 doc_scores,
                 score_pairs,
                 ranks,
                 model_name="model",
                 rouge_scores=None,
                 num_gold=None,
                 compute_macro_scores=True,
                 metrics="match"):
        """
        Initializes an instance of the EvaluationOutput class to compute and store various
        evaluation metrics for ranked and judged queries, document scores, and other
        relevant parameters. Metrics include MAP, MRR, NDCG, and ROUGE scores, among
        others.

        Attributes:
            model_name: The name of the model being evaluated.
            num_ranked_queries (int): The number of queries for which ranking has been computed.
            num_judged_queries (int): The number of queries for which relevance judgment is available.
            doc_scores (Any): The document scores generated for the queries by the model.
            score_pairs (list[[float,float]]): The paired scores specific to the evaluation setup (per example a list of
                               [output_score, gold_score] pairs).
            ranks (Any): The ranks assigned to the documents by the model output system.
            rouge_scores (Optional): ROUGE scores, if applicable, provided for the evaluation.
            num_gold (Optional[int]): The number of gold-relevant documents available per query.
            compute_macro_scores (bool): Indicates whether macro-averaged scores need to be computed.
            display_metrics (List[str]): A list of selected metrics to display during evaluation.

        Raises:
            No specific errors documented. Implementation assumes proper inputs.
        """
        self.ece = None
        self.brier = None
        self.model_name = model_name
        self.map = None
        self.mrr = None
        self.ndcg = None
        self.match = None
        self.idcg = None
        # for metric in EvaluationOutput.mappable_metrics+EvaluationOutput.single_metrics:
        #     setattr(self, metric, None)
        self.num_ranked_queries = num_ranked_queries
        self.num_judged_queries = num_judged_queries
        self.doc_scores = doc_scores
        self.score_pairs = score_pairs
        self.ranks = ranks
        self.rouge_scores = rouge_scores
        self.display_metrics = metrics.split(",")
        self.num_gold = num_gold
        self.compute_metrics()

    def compute_metrics(self):
        # compute match@rank
        # self.match = {self.doc_scores[r]/self.num_ranked_queries for r in self.ranks}
        # self.ndcg = {}
        def get_zeros(max_rank):
            return [0] * (max_rank + 1)

        def get_zero_dict():
            return {r:0 for r in self.ranks}

        self.match = get_zero_dict()
        self.ndcg = get_zero_dict()
        self.mrr = get_zero_dict()
        self.map = get_zero_dict()

        self.ece = 0
        self.brier = 0

        max_rank = self.ranks[-1]
        self.idcg = get_zeros(max_rank)
        for k in range(1, max_rank+1):
            self.idcg[k] = 1.0/math.log2(k+1)
        self.idcg = list(itertools.accumulate(self.idcg, operator.add))

        for m in self.mappable_metrics:
            setattr(self, m, get_zero_dict())

        for qid, relevant in enumerate(self.doc_scores):
            doc_dcg = get_zeros(max_rank)
            doc_match = get_zeros(max_rank)
            doc_mrr = get_zeros(max_rank)
            doc_map = get_zeros(max_rank)
            rank_first = -1
            found = False
            _mrr = 0
            _dcg = 0

            last_rank = self.ranks[0]
            j = 0
            while j < len(self.ranks) and self.ranks[j] <= len(relevant):
                last_rank = self.ranks[j]
                j += 1
            if j < len(self.ranks):
                last_rank = self.ranks[j]

            for i in range(0, min(len(relevant), self.ranks[-1])):
                if not found and relevant[i]:
                    rank_first = i+1
                    _mrr = 1.0/rank_first
                    found = True
                if relevant[i]:
                    _dcg += 1.0/math.log2(i+2)
                doc_match[i+1] = relevant[i]
                doc_dcg[i+1] = _dcg
                doc_mrr[i+1] = _mrr # 1.0/rank_first if rank_first>0 else 0

            for k in range(j, len(self.ranks)):

                k_rnk = self.ranks[k]
                doc_match[k_rnk] = relevant[rank_first - 1] if rank_first > 0 else False
                doc_dcg[k_rnk] = _dcg # relevant[i]/math.log2(i+2)
                doc_mrr[k_rnk] = _mrr # 1.0/rank_first if rank_first>0 else 0

            doc_map = list(itertools.accumulate(doc_match, operator.add))
            doc_match = list(itertools.accumulate(doc_match, max))
            # doc_dcg = list(itertools.accumulate(doc_dcg, operator.add))

            for i in self.ranks:
                update = i
                if i>last_rank:
                    update = last_rank
                # Normalize by ideal DCG at min(k, num_gold) to get proper NDCG in [0,1]
                ideal_k = min(update, self.num_gold[qid])
                self.ndcg[i] += doc_dcg[update] / self.idcg[ideal_k] if ideal_k > 0 else 0
                self.match[i] += doc_match[update]
                self.mrr[i] += doc_mrr[update]
                self.map[i] = doc_map[update]*1.0/update

            # print(self.match)


        for i in self.ranks:
            self.ndcg[i] /= self.num_judged_queries
            self.match[i] /= self.num_judged_queries
            self.mrr[i] /= self.num_judged_queries

        for metric in [self.ndcg, self.match, self.mrr]:
            metric = {i:v/self.num_judged_queries for i,v in metric.items()}

        if self.rouge_scores:
            self.rouge_match = {}

        # Flatten the nested structure
        flattened = list(itertools.chain.from_iterable(self.score_pairs))

        # Unzip into two separate lists
        first_values, second_values = zip(*flattened)

        # Convert to lists if needed
        self.system_probs = list(first_values)
        self.gold_values = list(second_values)
        self.ece = expected_calibration_error(self.system_probs, self.gold_values)
        self.brier = brier_score(self.system_probs, self.gold_values)


    def __str__(self):
        def display_metric(_metric, display_string=False):
            mappable_metrics = ['match', 'ndcg', 'mrr', 'map']
            display_map = {m:("M" if m=='match' else m.upper())+"@" for m in mappable_metrics}

            metric_map = {m:getattr(self, m) for m in mappable_metrics}

            if display_string:
                if _metric in mappable_metrics:
                    return "".join([f"{display_map[_metric] + str(val):10}" for val in ranks])
                else:
                    return f"{_metric:10}"
            else:
                if _metric in mappable_metrics:
                    return "".join([f"{metric_map[_metric][i]:<10.3}" for i in ranks])
                else:
                    return f"{getattr(self, _metric):<10.3}"
        name = self.model_name
        ranks = self.ranks
        model_name_length = len(self.model_name)+4

        headline = (f"{'Model':{model_name_length}s}" +
                    "".join([display_metric(metric, True) for metric in self.display_metrics
        ]))

        res = f"{name:{model_name_length}s}" + "".join([
            display_metric(metric, False) for metric in self.display_metrics
        ])

        return f"{headline}\n{res}\n"