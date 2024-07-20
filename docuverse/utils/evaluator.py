import os
import sys
from typing import List

from tqdm import tqdm


from docuverse import SearchResult, SearchQueries
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import EvaluationArguments
from . import get_param
from .evaluation_output import EvaluationOutput
from rouge_score.rouge_scorer import RougeScorer


class EvaluationEngine:
    def __init__(self, config):
        self.relevant = None
        self.rouge_scorer = None
        self.compute_rouge_score = None
        self.eval_measure = config.eval_measure
        self.config = config
        self.read(config)

    def read(self, config):
        if isinstance(config, EvaluationArguments):
            for param in vars(config):
                setattr(self, param, getattr(config, param))
        elif isinstance(config, str) and os.path.exists(config):  # It's a file
            pass
        if self.config.compute_rouge:
            self.rouge_scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def compute_score(self, input_queries: SearchQueries, system: List[SearchResult], model_name="model") -> EvaluationOutput:
        if self.compute_rouge_score:
            from rouge_score.rouge_scorer import RougeScorer
            scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        if get_param(input_queries[0], 'relevant', None) is None:
            print("The input question file does not contain answers. Please fix that and restart.")
            return EvaluationOutput()
        ranks = self.config.iranks
        rouge_scores = {r: 0 for r in ranks}  # passage scores
        gt = {-1: -1}
        num_positive = {}
        for q in input_queries:
            if isinstance(q['relevant'], list):
                gt[q['id']] = {id: 1 for id in q['relevant']}
                num_positive[q['id']] = len(q['relevant'])
            else:
                gt[q['id']] = {q['relevant']: 1}
                num_positive[q['id']] = 1

        def skip(out_ranks, record, rid):
            qid = record[0]
            while rid < len(out_ranks) and out_ranks[rid][0] == qid:
                rid += 1
            return rid

        def reverse_map(input_queries):
            rq_map = {}
            for i, q in enumerate(input_queries):
                rq_map[q['id']] = i
            return rq_map

        def update_scores(ranks, rnk, val, op, scores):
            j = 0
            while j < len(ranks) and ranks[j] <= rnk:
                j += 1
            for k in ranks[j:]:
                # scores[k] += 1
                scores[k] = op([scores[k], val])

        def get_doc_id(label):
            # find - from right side because id may have -
            index = label.rfind("-", 0, label.rfind("-"))
            if index >= 0:
                return label[:index]
            else:
                return label

        rqmap = reverse_map(input_queries)

        num_eval_questions = 0
        self.relevant = []
        num_gold = []
        for rid, record in tqdm(enumerate(system),
                                total=len(system),
                                desc='Evaluating questions: '):
            qid = get_param(record.question, 'id', rid)
            query = input_queries[rqmap[qid]]
            num_gold.append(num_positive[qid])
            if '-1' in gt[qid]:
                continue
            num_eval_questions += 1
            tmp_scores = {r: 0 for r in ranks}
            tmp_pscores = {r: 0 for r in ranks}
            self.relevant.append([])
            for aid, answer in enumerate(record.retrieved_passages):
                if aid >= ranks[-1]:
                    break
                docid = SearchData.get_orig_docid(answer['id'])

                # if str(docid) in gt[qid]:  # Great, we found a match.
                #     update_scores(ranks, aid, 1, sum, tmp_scores)
                #     self.relevant[rid].append(1)
                # else:
                #     self.relevant[rid].append(1)
                self.relevant[rid].append(str(docid) in gt[qid])

                if not self.config.compute_rouge:
                    continue
                if len(query['passages']) == 0:
                    scr = 0.
                else:
                    scr = max(
                        [
                            self.rouge_scorer.score(passage, answer['text'])['rouge1'].recall for passage in
                            query['passages']
                        ]
                    )

        _result = EvaluationOutput(num_ranked_queries=num_eval_questions,
                                   num_judged_queries=num_eval_questions,
                                   doc_scores=self.relevant,
                                   num_gold=num_gold,
                                   ranks=self.iranks,
                                   rouge_scores=rouge_scores,
                                   compute_macro_scores=True,
                                   model_name=model_name,
                                   metrics=self.eval_measure)

        if self.config.compute_rouge:
            _result['rouge_scores'] = \
                {r: int(1000 * rouge_scores[r] / num_eval_questions) / 1000.0 for r in ranks}

        return _result

    def __str__(self):
        return ""
