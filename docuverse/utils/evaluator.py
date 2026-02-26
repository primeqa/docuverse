import os
import sys
from typing import List

from tqdm import tqdm


from docuverse import SearchResult, SearchQueries, SearchEngine
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import EvaluationArguments, DocUVerseConfig
from . import get_param, get_orig_docid
from .evaluation_output import EvaluationOutput
from rouge_score.rouge_scorer import RougeScorer
from docuverse.utils.timer import timer

class EvaluationEngine:
    def __init__(self, config):
        self.score_pairs = None
        self.relevant = None
        self.rouge_scorer = None
        self.compute_rouge_score = None
        self.eval_measure = config.eval_measure
        if isinstance(config, DocUVerseConfig):
            self.config = config.eval_config
            self.data_template = config.data_template
            self.query_template = config.query_template
        elif isinstance(config, EvaluationArguments):
            self.config = config
            self.data_template = None
            self.query_template = None
        self.read()

    def read(self, config=None):
        if config is None:
            config = self.config
        if isinstance(config, DocUVerseConfig):
            for param in vars(config):
                setattr(self, param, getattr(config, param))
        elif isinstance(config, str) and os.path.exists(config):  # It's a file
            pass
        if self.config.compute_rouge:
            from rouge_score.rouge_scorer import RougeScorer
            self.rouge_scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def compute_score(self, input_queries: SearchQueries|List[SearchQueries.Query], system: List[SearchResult],
                      model_name="model", **kwargs) -> EvaluationOutput:
        data_id_header = 'id'
        tm = timer(f"{SearchEngine.get_name()}")
        # if 'data_template'in kwargs:
        data_text_header = get_param(get_param(kwargs, 'data_template', self.data_template),
                                     'text_header', 'text')
        data_id_header = get_param(get_param(kwargs, 'data_template', self.data_template),
                                   'id_header', 'id')
        # if 'query_template'in kwargs:
        query_id_header = get_param(get_param(kwargs, 'query_template', self.query_template),
                                    'id_header', 'id')
        relevant_header = get_param(get_param(kwargs, 'query_template', self.query_template),
                                    'relevant_header', "relevant")
        if get_param(input_queries[0], relevant_header, None) is None:
            print("The input question file does not contain answers. Please fix that and restart.")
            return EvaluationOutput()
        ranks = self.config.iranks
        rouge_scores = {r: 0 for r in ranks}  # passage scores
        gt = {-1: -1}
        num_positive = {}
        for q in input_queries:
            rels = get_param(q, relevant_header)
            id = get_param(q, query_id_header)
            if isinstance(rels, list):
                gt[id] = {id: 1 for id in rels}
                num_positive[id] = len(rels)
            else:
                gt[id] = {rels: 1}
                num_positive[id] = 1

        def skip(out_ranks, record, rid):
            qid = record[0]
            while rid < len(out_ranks) and out_ranks[rid][0] == qid:
                rid += 1
            return rid

        def reverse_map(input_queries):
            rq_map = {}
            for i, q in enumerate(input_queries):
                rq_map[q.id] = i
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
        self.score_pairs = []
        num_gold = []
        for rid, record in tqdm(enumerate(system),
                                total=len(system),
                                desc='Evaluating questions: '):
            qid = get_param(record.question, query_id_header)
            if qid not in rqmap or rqmap[qid]>=len(input_queries):
                print(f"Missing queryid {qid}")
                continue
            query = input_queries[rqmap[qid]]
            num_gold.append(num_positive[qid])
            if '-1' in gt[qid]:
                continue
            num_eval_questions += 1
            tmp_scores = {r: 0 for r in ranks}
            tmp_pscores = {r: 0 for r in ranks}
            self.relevant.append([])
            self.score_pairs.append([])
            seen_docids = set()  # Track seen original docids to avoid duplicates
            eval_count = 0  # Count of unique documents evaluated
            for aid, answer in enumerate(record.retrieved_passages):
                if eval_count >= ranks[-1]:
                    break
                docid = get_orig_docid(get_param(answer, f"id|{data_id_header}"))

                # Skip if we've already seen this original document
                if str(docid) in seen_docids:
                    continue
                seen_docids.add(str(docid))
                eval_count += 1

                self.relevant[rid].append(str(docid) in gt[qid])
                self.score_pairs[rid].append([answer.score, 1.0*(str(docid) in gt[qid])])
                if not self.config.compute_rouge:
                    continue
                if len(query['passages']) == 0:
                    scr = 0.
                else:
                    pass
            # print(f"{rid}: {len([i for i, t in enumerate(self.relevant[rid]) if t])} "
            #       f"gold: {len(get_param(record.question, query_id_header, []))}")

        _result = EvaluationOutput(num_ranked_queries=num_eval_questions,
                                   num_judged_queries=num_eval_questions,
                                   doc_scores=self.relevant,
                                   score_pairs=self.score_pairs,
                                   num_gold=num_gold,
                                   ranks=self.config.iranks,
                                   rouge_scores=rouge_scores,
                                   compute_macro_scores=True,
                                   model_name=model_name,
                                   metrics=self.eval_measure)

        if self.config.compute_rouge:
            _result['rouge_scores'] = \
                {r: int(1000 * rouge_scores[r] / num_eval_questions) / 1000.0 for r in ranks}
        tm.add_timing("evaluate_time")
        return _result

    def __str__(self):
        return ""
