import os
import re

from docuverse.engines import SearchData
from docuverse.engines.data_template import default_query_template
from docuverse.utils import get_param


class SearchQueries(SearchData):
    class Query:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getitem__(self, key: str, default=None):
            return getattr(self, key, default)

        def __setattr__(self, key, value):
            # setattr(self, key, value)
            self.__dict__[key] = value

        def as_list(self):
            return self.__dict__

        def __str__(self):
            return f"{type(self)}({str(self.as_list())})"

    def __init__(self, preprocessor, filenames, **data):
        super().__init__(filenames, **data)
        self.queries = preprocessor.get_queries()

    def __iter__(self):
        return iter(self.queries)

    def __getitem__(self, i: int):
        return self.queries[i]

    @staticmethod
    def read(query_file, template=default_query_template, **kwargs):
        return SearchQueries.read_question_data(in_files=query_file, template=template, **kwargs)

    @classmethod
    def read_question_data(cls, in_files, fields=None, lang="en",
                           remv_stopwords=False,
                           url=None,
                           query_template=default_query_template,
                           **kwargs):
        def get_filename(in_file, local_file):
            qfile = local_file
            if not os.path.exists(qfile):
                qfile = os.path.join(os.path.dirname(in_file), qfile)
            return qfile

        questions = []
        if isinstance(in_files, str):
            in_files = [in_files]
        elif not isinstance(in_files, list):
            raise RuntimeError(f"Invalid argument 'in_files' type: {type(in_files)}")

        for in_file in in_files:
            with open(in_file, "r", encoding="utf-8") as file_stream:
                delim = "," if ".csv" in in_file else "\t"
                # csv_reader = \
                #     csv.DictReader(file_stream, fieldnames=fields, delimiter=delim) \
                #         if fields is not None \
                #         else csv.DictReader(file_stream, delimiter=delim)
                # next(csv_reader)
                question_data = cls._read_data(in_file)
                if 'question_file' in question_data and 'goldstandard_file' in question_data:
                    qfile = get_filename(in_file, question_data['question_file'])
                    tquestions = cls.read_question_data(qfile, fields=fields,
                                                        lang=lang,
                                                        url=url,
                                                        query_template=query_template,
                                                        **kwargs)
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
                    test_questions = []
                    for q in tquestions:
                        if q.id in gs_map:
                            q.relevant = get_param(gs_map, q.id, '')
                            test_questions.append(q)
                    questions.extend(test_questions)
                else:
                    for it, row in enumerate(question_data):
                        question = get_param(row, query_template.text_header)
                        if url is not None:
                            question = (re.sub(url, lang, 'URL', question), remv_stopwords)
                        rels = get_param(row, query_template.relevant_header, "")
                        if isinstance(rels, str):
                            rels = rels.split(",")
                        itm = {'text': question,
                               'id': get_param(row, query_template.id_header, str(it)),
                               'relevant': rels
                               }
                        answers = get_param(row, query_template.answers_header, "")
                        if isinstance(answers, str):
                            if "::" in answers:
                                itm['answers'] = answers.split("::")
                            else:
                                itm['answers'] = [answers]
                        elif isinstance(answers, list):
                            itm['answers'] = answers
                        itm['passages'] = answers

                        questions.append(SearchQueries.Query(**itm))
        return questions
