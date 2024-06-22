import math
import itertools
import operator


class EvaluationOutput:
    mappable_metrics = ['match', 'mrr', 'ndcg', 'map']
    def __init__(self, num_ranked_queries, num_judged_queries, doc_scores,
                 ranks,
                 model_name="model",
                 rouge_scores=None,
                 compute_macro_scores=True,
                 metrics="match"):
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
            for i in range(0, min(len(relevant), self.ranks[-1])):
                if not found and relevant[i]:
                    rank_first = i+1
                    found = True
                doc_match[i+1] = relevant[i]
                doc_dcg[i+1] = relevant[i]/math.log2(i+2)
                doc_mrr[i+1] = 1.0/rank_first if rank_first>0 else 0

            doc_map = list(itertools.accumulate(doc_match, operator.add))
            doc_match = list(itertools.accumulate(doc_match, max))
            doc_dcg = list(itertools.accumulate(doc_dcg, operator.add))

            for i in self.ranks:
                self.ndcg[i] += doc_dcg[i]/self.idcg[i]
                self.match[i] += doc_match[i]
                self.mrr[i] += doc_mrr[i]
                self.map[i] = doc_map[i]*1.0/i
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

        headline = f"{'Model':10s}" + "".join([
            display_metric(metric, True) for metric in self.display_metrics
        ])

        res = f"{name:10}" + "".join([
            display_metric(metric, False) for metric in self.display_metrics
        ])

        return f"{headline}\n{res}\n"