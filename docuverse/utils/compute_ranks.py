import sys
from docuverse.utils import save_command_line, get_param, get_orig_docid
from docuverse.engines import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig

if __name__ == '__main__':
    save_command_line(args=sys.argv)
    config = DocUVerseConfig.get_stdargs_config()
    engine = SearchEngine(config)
    output = engine.read_output()

    qids = []
    ranks = []
    data_id_header = config.data_template.id_header
    relevant_header = config.query_template.relevant_header
    for o in output:
        qid = o.question.id
        r = []
        qids.append(qid)
        for i, a in enumerate(o):
            docid = get_orig_docid(get_param(a, f"id|{data_id_header}"))
            if docid in o.question[relevant_header]:
                r.append(i)
        ranks.append(r)

    print("\n".join([f"{q} {r}" for q, r in zip(qids, ranks)]))