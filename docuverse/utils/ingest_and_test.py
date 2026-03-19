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


def _save_timing_statistics(output_file, timing_stats=None):
    """Save timing statistics to a .timing.json file derived from the output filename.

    Args:
        output_file: the output file path from config
        timing_stats: comma-separated string of stat names, or None for defaults
    """
    if not output_file:
        return
    # Strip .json or .jsonl extension, then append .timing.json
    for ext in ('.jsonl', '.json'):
        if output_file.endswith(ext):
            base = output_file[:-len(ext)]
            break
    else:
        base = output_file
    timing_file = base + '.timing.json'
    stats = [s.strip() for s in timing_stats.split(",")] if timing_stats else None
    timer.save_statistics(timing_file, stats=stats, save_samples=True)
    print(f"Timing statistics saved to {timing_file}")


def main_cli():
    global config, engine, results
    save_command_line(args=sys.argv)
    tm = timer("ingest_and_test")
    config = DocUVerseConfig.get_stdargs_config()
    engine = SearchEngine(config, name="ingest_and_test")
    corpus = None
    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update, skip=config.skip)
    output = None
    if config.retrieve:
        queries = engine.read_questions(config.input_queries)
        output = engine.search(queries)
        # tm.add_timing("search")
        engine.write_output(output)
        # tm.add_timing("write_output")
    else:
        queries = None

    timer_keys = {'queries': 1, 'docs': 1, 'default': 1}
    if queries is not None:
        timer_keys['queries'] = len(queries)
    if corpus is not None:
        timer_keys['docs'] = len(corpus)
    key_associations = {"ingest": "docs",
                        "search": "queries",
                        "eval": "queries",
                        "write_output": "queries",
                        }
    if config.evaluate and config.eval_config is not None:
        scorer = EvaluationEngine(config)
        if queries is None:
            queries = engine.read_questions(config.input_queries)

        if output is None:
            output = engine.read_output(config.output_file)
        results = scorer.compute_score(queries, output, model_name=engine.get_output_name())
        json_off = config.output_file.find('.json')
        metrics_file = config.output_file[:json_off] + '.metrics' if json_off>=0 \
            else config.output_file + '.metrics'
        # tm.add_timing("evaluate")
        ostring = io.StringIO()
        # print(timer.display_timing)
        timer.display_timing(tm.milliseconds_since_beginning(),
                             keys=timer_keys, sorted_by="%",
                             key_associations=key_associations,
                             reverse=True,
                             output_stream=ostring,
                             stat_list=config.timing_stats
                             )
        timing = ostring.getvalue()

        print(f"Results:\n{results}\n")
        write_metrics_file(metrics_file, results, timing, config)
        if results.gold_values is not None:
            from docuverse.utils.ece_brier.ece import plot_reliability_diagram
            diagram_name =  metrics_file.replace(".metrics", ".png")
            plot_reliability_diagram(results.system_probs, results.gold_values, save_path=diagram_name)
        print(f"Timing: ")
        print(timing)

    else:
        timer.display_timing(tm.milliseconds_since_beginning(),
                             keys=timer_keys,
                             key_associations=key_associations,
                             sorted_by="%",
                             reverse=True,
                             stat_list=config.timing_stats
                             )

    # Save detailed timing statistics to disk
    _save_timing_statistics(config.output_file, getattr(config, 'timing_stats', None))


if __name__ == '__main__':
    main_cli()

