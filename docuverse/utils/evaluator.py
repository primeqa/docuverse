import os
import sys

from rouge import Rouge
from tqdm import tqdm

from docuverse import SearchResult, SearchQueries
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import EvaluationConfig, EvaluationArguments
from .evaluation_output import EvaluationOutput


class EvaluationEngine:
    def __init__(self, config):
        self.rouge_scorer = None
        self.compute_rouge_score = None
        self.config = config
        self.read(config)

    def read(self, config):
        if isinstance(config, EvaluationArguments):
            for param in vars(config):
                setattr(self, param, getattr(config, param))
        elif isinstance(config, str) and os.path.exists(config): # It's a file
            pass
        if self.config.compute_rouge:
            self.rouge_scorer = Rouge()

    def compute_score(self, input_queries: SearchQueries, system: SearchResult) -> EvaluationOutput:
        if self.compute_rouge_score:
            from rouge_score.rouge_scorer import RougeScorer
            scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        if "relevant" not in input_queries[0] or input_queries[0]['relevant'] is None:
            print("The input question file does not contain answers. Please fix that and restart.")
            return EvaluationOutput()
        ranks = self.config.iranks
        docm_scores = {r: 0 for r in ranks}
        rouge_scores = {r: 0 for r in ranks}  # passage scores
        gt = {-1: -1}
        for q in input_queries:
            if isinstance(q['relevant'], list):
                gt[q['id']] = {id: 1 for id in q['relevant']}
            else:
                gt[q['id']] = {q['relevant']: 1}

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
        for rid, record in tqdm(enumerate(system),
                                total=len(system),
                                desc='Evaluating questions: '):
            qid = record['qid']
            query = input_queries[rqmap[qid]]
            if '-1' in gt[qid]:
                continue
            num_eval_questions += 1
            tmp_scores = {r: 0 for r in ranks}
            tmp_pscores = {r: 0 for r in ranks}
            for aid, answer in enumerate(record['answers']):
                if aid >= ranks[-1]:
                    break
                docid = get_doc_id(answer['id'])

                if str(docid) in gt[qid]:  # Great, we found a match.
                    update_scores(ranks, aid, 1, sum, tmp_scores)
                if not self.config.compute_rouge_score:
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
                update_scores(ranks, aid, scr, max, tmp_pscores)

            for r in ranks:
                docm_scores[r] += int(tmp_scores[r] >= 1)
                if self.compute_rouge_score:
                    rouge_scores[r] += tmp_pscores[r]

        _result = EvaluationOutput(num_ranked_queries=num_eval_questions,
                                   num_judged_queries=num_eval_questions,
                                   doc_scores={r: int(1000 * docm_scores[r] / num_eval_questions) / 1000.0 for r in ranks})


        if self.config.compute_rouge_score:
            _result['rouge_scores'] = \
                {r: int(1000 * rouge_scores[r] / num_eval_questions) / 1000.0 for r in ranks}

        return _result

    def __str__(self):
        return ""
