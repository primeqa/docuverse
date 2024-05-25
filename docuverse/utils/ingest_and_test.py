# from __future__ import annotations
import json
from docuverse import SearchEngine, SearchQueries
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import DocUVerseConfig
# from docuverse.utils import DenseEmbeddingFunction
from docuverse.utils.evaluator import EvaluationEngine
# from docuverse.utils.text_tiler import TextTiler

# from transformers.hf_argparser import HfArgumentParser

if __name__ == '__main__':
    config = DocUVerseConfig.get_stdargs()

    engine = SearchEngine(config.search_config)
    scorer = None

    if config.evaluate and config.eval_config is not None:
        scorer = EvaluationEngine(config.eval_config)

    if config.ingest or config.update:
        # corpus = SearchData.read_data(config.input_passages, **vars(config.search_config))
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update)

    if config.retrieve:
        queries = SearchQueries.read(config.input_queries, **vars(config.search_config))
        # queries = queries[:10]
        output = engine.search(queries)
        # output.save(config.output_file)
        with open(config.output_file, "w") as outfile:
            outp = [r.as_list() for r in output]
            outfile.write(json.dumps(outp, indent=2))
        if config.evaluate:
            results = scorer.compute_score(queries, output)
            print(f"Results:\n{results}")