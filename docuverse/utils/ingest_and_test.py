# from __future__ import annotations
import json
from datetime import datetime
import os
import sys
import io
from docuverse.utils.timer import timer

from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine
from docuverse.utils import save_command_line


if __name__ == '__main__':
    save_command_line(args=sys.argv)
    tm = timer("ingest_and_test")
    config = DocUVerseConfig.get_stdargs_config()
#    config = DocUVerseConfig("experiments/clapnq/setup.yaml")
    engine = SearchEngine(config)
    tm.add_timing("initialize")
    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update, skip=config.skip)
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
        tm.add_timing("evaluate")
        ostring = io.StringIO()
        print(timer.display_timing)
        timer.display_timing(tm.milliseconds_since_beginning(), num_chars=0, num_words=0, sorted_by="%",
                             reverse=True, output_stream=ostring)
        # timer.display_timing(tm.milliseconds_since_beginning(), num_chars=0, num_words=0, sorted_by="%",
        #                      reverse=True)
        print(f"Results:\n{results}")
        with open(metrics_file, "w") as out:
            out.write(str(results))
            out.write(f"\n#Command: python {' '.join(sys.argv)}\n")
            out.write(f"Timing:\n")
            out.write(ostring.getvalue()+"\n")
        print(f"Timing:\n")
        print(ostring.getvalue()+"\n")

    else:
        timer.display_timing(tm.milliseconds_since_beginning(), num_chars=0, num_words=0, sorted_by="%",
                             reverse=True)

