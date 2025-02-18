import json
from datetime import datetime
import os
import sys
import io
from docuverse.utils.timer import timer
from docuverse.utils import save_command_line
import simple_colors

from docuverse import SearchEngine, SearchQueries
from docuverse.engines.search_engine_config_params import DocUVerseConfig
from docuverse.utils.evaluator import EvaluationEngine

if __name__ == '__main__':
    save_command_line(args=sys.argv)
    config = DocUVerseConfig.get_stdargs_config()

    engine = SearchEngine(config)
    if config.ingest or config.update:
        corpus = engine.read_data(config.input_passages)
        engine.ingest(corpus, update=config.update, skip=config.skip)

    # run retrieval in a loop
    num = 0
    qtemplate = config.query_template
    while True:
        query = input(simple_colors.yellow("Type a query (<enter to quit>): ", 'bold'))
        if query in ['exit', 'done', '']:
            break

        inp = SearchQueries.Query(template=qtemplate,
                                  **{qtemplate.text_header:query,
                                     qtemplate.id_header: num,
                                     qtemplate.answers_header: [],
                                     }
                                  )
        res = engine.search([inp])[0]
        for ans in res.retrieved_passages[:3]:
            print(f" {ans.as_dict()}")
