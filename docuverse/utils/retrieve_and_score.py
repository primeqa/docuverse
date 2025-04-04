import argparse

from docuverse.utils import save_command_line

from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve and score documents')
    parser.add_argument('--config', required=True, help='Configuration file')

    parser.add_argument('--get_best_rouge_passage', default=False, action='store_true',
                        help='Whether to rank the passages by rouge with the gold answer')
    args = parser.parse_args()

    config = DocUVerseConfig(args.config)
    # skip reading the cache
    config.eval_config.no_cache = True
    config.no_cache = True
    engine = SearchEngine(config)
    scorer = EvaluationEngine(config)

    queries = engine.read_questions(config.input_queries)

    results = engine.search(queries)
    scores = scorer.compute_score(queries, results, model_name=engine.get_output_name())
    print(f"Results\n{scores}")

    if args.get_best_rouge_passage:
        from rouge_score.rouge_scorer import RougeScorer
        rouge_scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

        for res in results:
            gold = res.question['Gold Answer']
            scores = []
            for answer in res:
                score = rouge_scorer.score(gold, answer.text)
                scores.append(score['rouge1'].recall)
            inds = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            print(f"Best answer for {res.question.text}: {res[inds[0]]['id']}, recall: {scores[inds[0]]}")
