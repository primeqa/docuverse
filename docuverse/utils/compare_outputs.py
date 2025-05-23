from docuverse import SearchEngine, SearchQueries
from docuverse.engines.search_engine_config_params import DocUVerseConfig

import argparse

from docuverse.utils.evaluator import EvaluationEngine

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Handle DocUVerse configuration flags")
    parser.add_argument('--config1', type=str, required=True, help="Path to the first config file")
    parser.add_argument('--config2', type=str, default=None, required=False, help="Path to the second config file")
    parser.add_argument('--outputs', nargs="+", required=False,
                        default=None, help="Paths to the output files")
    args = parser.parse_args()

    if args.config2 is None:
        args.config2 = args.config1
    config_names = [args.config1, args.config2]
    # queries = [SearchQueries.read(c.input_questions, **vars(c.retriever_config)) for c in config_names]
    configs = [DocUVerseConfig(config) for config in config_names]
    scorers = [EvaluationEngine(config) for config in configs]

    if args.outputs is not None:
        outputs = [SearchEngine.read_output_(of, query_template=c.query_template)
                   for of, c in zip(args.outputs, configs)]
    else:
        outputs = [SearchEngine.read_output_(c.output_file, query_template=c.query_template) for c in configs]
    queries = [[a.question for a in output] for output in outputs]
    scores = [s.compute_score(q, o, "bla") for s,q,o in zip(scorers, queries, outputs)]

