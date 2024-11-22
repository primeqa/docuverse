from docuverse import SearchEngine, SearchQueries
from docuverse.engines.data_template import DataTemplate, default_data_template, default_query_template
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to compute statistical significance")
    parser.add_argument("--ranks", nargs="+", type=int, help="List of ranks")
    parser.add_argument("---measures", "-M", nargs="+", choices=["M", "MRR", "NGCD"], help="List of measures")
    parser.add_argument("baseline", type=str, help="Baseline file")
    parser.add_argument("challenger", type=str, help="Experiment file file")

    args = parser.parse_args()

    out1 = SearchEngine.read_output_(args.baseline, query_template=default_query_template)
    out2 = SearchEngine.read_output_(args.challenger, query_template=default_query_template)

    _config = {
        "ranks": args.ranks,
    }

    config = DocUVerseConfig(_config)

    scorer = EvaluationEngine(config)

    res1 = scorer.compute_score([o.question for o in out1], out1, model_name="baseline")
    res2 = scorer.compute_score([o.question for o in out2], out2, model_name="challenger")



