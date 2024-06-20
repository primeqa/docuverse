# from __future__ import annotations
import json
from datetime import datetime
import os
import sys

from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine

if __name__ == '__main__':
    with open("logfile", "a") as cmdlog:
        cmdlog.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {os.getenv('USER')} - "
                     f"{' '.join([__file__] + sys.argv)}\n")

    config = DocUVerseConfig.get_stdargs_config()
#    config = DocUVerseConfig("experiments/clapnq/setup.yaml")
    engine = SearchEngine(config)

    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update)

    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        output = engine.search(queries)

        with open(config.output_file, "w") as outfile:
            outp = [r.as_list() for r in output]
            outfile.write(json.dumps(outp, indent=2))

        if config.evaluate and config.eval_config is not None:
            scorer = EvaluationEngine(config.eval_config)
            results = scorer.compute_score(queries, output)
            print(f"Results:\n{results}")