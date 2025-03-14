import math
import itertools
import operator

class EvaluationOutput:
    mappable_metrics = ['match', 'mrr', 'ndcg', 'map']
    def __init__(self, num_ranked_queries, num_judged_queries, doc_scores,
                 ranks,
                 model_name="model",
                 rouge_scores=None,
                 num_gold=None,
                 compute_macro_scores=True,
                 metrics="match"):
        """

        Initializes an EvaluationOutput object.

        :param num_ranked_queries: The total number of queries for which ranking was done.
        :param num_judged_queries: The total number of queries that were judged.
        :param doc_scores: A dictionary mapping document IDs to their corresponding scores.
        :param ranks: A dictionary mapping query IDs to a list of document IDs in ranked order.
        :param model_name: The name of the model used for ranking (default is "model").
        :param rouge_scores: A dictionary mapping query IDs to their corresponding Rouge scores (default is None).
        :param num_gold: The number of gold standard documents per query (default is None).
        :param compute_macro_scores: A boolean indicating whether to compute macro scores (default is True).
        :param metrics: The metrics to be computed, separated by commas (default is "match").

        :returns: None

        """
        self.model_name = model_name
        self.map = None
        self.mrr = None
        self.ndcg = None
        self.match = None
        self.idcg = None
        for i in EvaluationOutput.mappable_metrics:
            setattr(self, i, None)
        self.num_ranked_queries = num_ranked_queries
        self.num_judged_queries = num_judged_queries
        self.doc_scores = doc_scores
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
                    _dcg = 1.0/math.log2(rank_first+1)
                    found = True
                doc_match[i+1] = relevant[i]
                doc_dcg[i+1] = _dcg # relevant[i]/math.log2(i+2)
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
                # self.ndcg[i] += doc_dcg[update]/self.idcg[min(update, self.num_gold[qid])]
                self.ndcg[i] += doc_dcg[update] / self.idcg[1]
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


    def __str__(self):

        def display_metric(_metric, display_string=False):
            mappable_metrics = ['match', 'ndcg', 'mrr']
            display_map = {m:("M" if m=='match' else m.upper())+"@" for m in mappable_metrics}
            metric_map = {m:getattr(self,m) for m in mappable_metrics}

            if display_string:
                if _metric in mappable_metrics:
                    return "".join([f"{display_map[_metric] + str(val):10}" for val in ranks])
            else:
                if _metric in mappable_metrics:
                    return "".join([f"{metric_map[_metric][i]:<10.3}" for i in ranks])
        name = self.model_name
        ranks = self.ranks
        model_name_length = len(self.model_name)+4

        headline = f"{'Model':{model_name_length}s}" + "".join([
            display_metric(metric, True) for metric in self.display_metrics
        ])

        res = f"{name:{model_name_length}s}" + "".join([
            display_metric(metric, False) for metric in self.display_metrics
        ])

        return f"{headline}\n{res}\n"