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
                     f"{' '.join(sys.argv)}\n")

    config = DocUVerseConfig.get_stdargs_config()
#    config = DocUVerseConfig("experiments/clapnq/setup.yaml")
    engine = SearchEngine(config)

    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update)

    output = None
    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        output = engine.search(queries)

        engine.write_output(output)
    else:
        output = None
        queries = None

        # with open(config.output_file, "w") as outfile:
        #     outp = [r.as_list() for r in output]
        #     outfile.write(json.dumps(outp, indent=2))

        # if config.evaluate and config.eval_config is not None:
        #     scorer = EvaluationEngine(config.eval_config)
        #     results = scorer.compute_score(queries, output, model_name=config.index_name)
        #     print(f"Results:\n{results}")
    if config.evaluate and config.eval_config is not None:
        scorer = EvaluationEngine(config.eval_config)
        if queries is None:
            queries = engine.read_questions(config.input_queries)

        if output is None:
            output = engine.read_output(config.output_file)
            # with open(config.output_file, "r") as inp:
            #     output = json.load(inp)
        results = scorer.compute_score(queries, output, model_name=config.index_name)
        print(f"Results:\n{results}")
