# from __future__ import annotations
import json
from datetime import datetime
import os
import sys
from docuverse.utils.timer import timer

from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine

if __name__ == '__main__':
    with open("logfile", "a") as cmdlog:
        cmdlog.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {os.getenv('USER')} - "
                     f"{' '.join(sys.argv)}\n")
    tm = timer("ingest_and_test")
    config = DocUVerseConfig.get_stdargs_config()
#    config = DocUVerseConfig("experiments/clapnq/setup.yaml")
    engine = SearchEngine(config)
    tm.add_timing("initialize")
    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update)
        tm.add_timing("ingest")

    output = None
    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        output = engine.search(queries)
        tm.add_timing("search")
        engine.write_output(output)
        tm.add_timing("write_output")
    else:
        output = None
        queries = None

    if config.evaluate and config.eval_config is not None:
        scorer = EvaluationEngine(config)
        if queries is None:
            queries = engine.read_questions(config.input_queries)

        if output is None:
            output = engine.read_output(config.output_file)
        results = scorer.compute_score(queries, output, model_name=engine.get_output_name())
        metrics_file = config.output_file.replace(".json", ".metrics")
        print(f"Results:\n{results}")
        with open(metrics_file, "w") as out:
            out.write(str(results))
        tm.add_timing("evaluate")

    timer.display_timing(tm.milliseconds_since_beginning(), num_chars=0, num_words=0, sorted_by="%", reverse=True)