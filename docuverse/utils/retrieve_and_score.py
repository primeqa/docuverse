import argparse

from docuverse.utils import save_command_line

from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve and score documents')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--get_best_rouge_passage', default=False,
                        help='Whether to rank the passages by rouge with the gold answer')
    args = parser.parse_args()

    config = DocUVerseConfig(args.config)
    engine = SearchEngine(config)
    scorer = EvaluationEngine(config)

    queries = engine.read_questions(config.input_queries)

    results = engine.search(queries)

    if args.get_best_rouge_passage:
        from rouge_score.rouge_scorer import RougeScorer
        rouge_scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        for res in results:
            gold = res.question['Gold answer']
            scores = []
            for answer in res:
                score = rouge_scorer.score(answer.text, gold)['rouge1']['r']
                scores.append(score)
            inds = sorted(range(len(scores)), key=lambda i: scores[i])
            print(f"Best answer: {res[inds[0]].as_dict()}")
