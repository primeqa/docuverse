# from __future__ import annotations
import sys
import io

from docuverse.utils import prepare_for_save_and_backup, save_command_line
from docuverse.utils.timer import timer

from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine


def write_metrics_file(metrics_file, _results, _timing, _config):
    prepare_for_save_and_backup(metrics_file)
    with open(metrics_file, "w") as out:
        out.write(str(_results)+"\n")
        command_line = ' '.join(sys.argv)
        out.write(f"\n#Command: python {command_line}\n")
        out.write(f"Timing:\n")
        out.write(_timing + "\n")
        out.write("=" * 30 + "\n")
        out.write(f"****** Config: *******\n")
        out.write(_config.to_yaml().replace("\n", "\\n") + "\n")
        out.write("=" * 30 + "\n")


def main_cli():
    global config, engine, results
    save_command_line(args=sys.argv)
    tm = timer("ingest_and_test")
    config = DocUVerseConfig.get_stdargs_config()
    #    config = DocUVerseConfig("experiments/clapnq/setup.yaml")
    engine = SearchEngine(config, name="ingest_and_test")
    # tm.add_timing("initialize")
    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update, skip=config.skip)
        # tm.add_timing("ingest")
    output = None
    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        output = engine.search(queries)
        # tm.add_timing("search")
        engine.write_output(output)
        # tm.add_timing("write_output")
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
        metrics_file = config.output_file[:config.output_file.find(
            '.json')] + '.metrics' if '.json' in config.output_file else config.output_file + '.metrics'
        # tm.add_timing("evaluate")
        ostring = io.StringIO()
        # print(timer.display_timing)
        timer.display_timing(tm.milliseconds_since_beginning(), keys={'queries': len(queries)}, sorted_by="%",
                             reverse=True, output_stream=ostring)
        timing = ostring.getvalue()

        print(f"Results:\n{results}\n")
        write_metrics_file(metrics_file, results, timing, config)
        print(f"Timing: ")
        print(timing)

    else:
        timer.display_timing(tm.milliseconds_since_beginning(), keys={'queries': len(queries)}, sorted_by="%",
                             reverse=True)


if __name__ == '__main__':
    main_cli()

