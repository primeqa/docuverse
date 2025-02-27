import os
import re
import json

from docuverse.engines import SearchData
from docuverse.engines.data_template import default_query_template, DataTemplate
from docuverse.utils import get_param, at_most


class SearchQueries(SearchData):
    class Query:
        def __init__(self, template: DataTemplate, **kwargs):
            self.template = template
            self.args = kwargs
            # self.__dict__.update(kwargs)

        def __getitem__(self, key: str, default=None):
            return self.get(key, default)

        # def __setattr__(self, key, value):
        #     # setattr(self, key, value)
        #     self.args[key] = value

        def as_dict(self):
            return self.args

        def as_json(self, **kwargs):
            return json.dumps(self.as_dict(), **kwargs)

        def __str__(self):
            return f"{type(self)}({str(self.as_dict())})"

        def get(self, key, default=None):
            return self.args.get(key, default)

        @property
        def id(self):
            return get_param(self, self.template.id_header)

        @property
        def text(self):
            return get_param(self, self.template.text_header)

    def __init__(self, preprocessor, filenames, **data):
        super().__init__(filenames, **data)
        self.queries = preprocessor.get_queries()

    def __iter__(self):
        return iter(self.queries)

    def __getitem__(self, i: int):
        return self.queries[i]

    _ms = re.compile("'\\s+'")

    def __iadd__(self, q: Query):
        self.queries.append(q)

    def append(self, q: Query):
        self.queries.append(q)

    @staticmethod
    def read(query_file, template=default_query_template, **kwargs):
        return SearchQueries.read_question_data(in_files=query_file, template=template, **kwargs)

    @classmethod
    def read_question_data(cls, in_files, fields=None, lang="en",
                           remv_stopwords=False,
                           url=None,
                           query_template=default_query_template,
                           relevant_map=None,
                           **kwargs):
        def get_filename(in_file, local_file):
            qfile = local_file
            if not os.path.exists(qfile):
                qfile = os.path.join(os.path.dirname(in_file), qfile)
            return qfile

        questions = []
        max_num_questions = get_param(kwargs, 'max_num_questions', -1)
        max_num_questions = -1 if max_num_questions is None else int(max_num_questions)
        ignore_empty_questions = bool(get_param(kwargs, 'ignore_empty_questions', False))
        if isinstance(in_files, str|dict):
            in_files = [in_files]
        elif isinstance(in_files, list):
            if isinstance(in_files[0], dict):
                in_files = [in_files]
        else:
            raise RuntimeError(f"Invalid argument 'in_files' type: {type(in_files)}")

        for in_file in in_files:
            if isinstance(in_file, dict|list):
                question_data = in_file
            elif isinstance(in_file, str):
                question_data = cls._read_data(in_file)
            if 'question_file' in question_data and 'goldstandard_file' in question_data:
                tfile = get_filename(in_file, question_data['goldstandard_file'])
                goldstandard = cls._read_data(tfile)
                gs_map = {}
                for gs in goldstandard:
                    qid = get_param(gs, query_template.truth_id)
                    doc_id = get_param(gs, query_template.truth_label)
                    if qid in gs_map:
                        gs_map[qid].append(doc_id)
                    else:
                        gs_map[qid] = [doc_id]
                qfile = get_filename(in_file, question_data['question_file'])
                tquestions = cls.read_question_data(qfile, fields=fields,
                                                    lang=lang,
                                                    url=url,
                                                    query_template=query_template,
                                                    relevant_map=gs_map,
                                                    **kwargs)
                questions.extend(tquestions)
            else:
                out_text_header = query_template.text_header.split("|")[0]
                out_id_header = query_template.id_header.split("|")[0]
                out_relevant_header = query_template.relevant_header.split("|")[0]
                for it, row in enumerate(at_most(question_data, max_num_questions)):
                    question = get_param(row, query_template.text_header)
                    if url is not None:
                        question = (re.sub(url, lang, 'URL', question), remv_stopwords)
                    rels = None
                    if relevant_map is not None:
                        rels = get_param(relevant_map, get_param(row, query_template.id_header, None), None)
                    if rels is None:
                        rels = get_param(row, query_template.relevant_header, "")
                    if ignore_empty_questions and (rels is None or not rels):
                        continue
                    if isinstance(rels, str):
                        rels = rels.split(",")
                    itm = {out_text_header: question,
                           out_id_header: get_param(row, query_template.id_header, str(it)),
                           out_relevant_header: rels
                           }
                    if query_template.extra_fields is not None:
                        for extra in query_template.extra_fields:
                            if extra in row:
                                val = row[extra]
                                if isinstance(val, str):
                                    if val.find('[') >= 0 > val.find("None"):
                                        # Assume it's some sort of json field
                                        if val.find("'") >= 0:
                                            if val.find("' '") >= 0:
                                                val = re.sub(cls._ms, "','", val)
                                            val = val.replace("'", '"')
                                        try:
                                            # print(f"Loading {it}: {val}")
                                            val = json.loads(val)
                                            itm[extra] = val
                                        except Exception as e:
                                            print(f"Cannot parse field {row[extra]}: {e}")
                                    else:
                                        itm[extra] = val
                                elif isinstance(val, list|dict):
                                    itm[extra] = val

                    answers = get_param(row, query_template.answers_header, "")
                    if isinstance(answers, str):
                        if "::" in answers:
                            itm['answers'] = answers.split("::")
                        else:
                            itm['answers'] = [answers]
                    elif isinstance(answers, list):
                        itm['answers'] = answers
                    itm['passages'] = answers

                    questions.append(SearchQueries.Query(template=query_template, **itm))

        return questions
