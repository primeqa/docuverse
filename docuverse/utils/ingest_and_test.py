# from __future__ import annotations

# import bz2
# import os, re, json, csv
# from argparse import ArgumentParser
#
# from tqdm import tqdm
# import numpy as np
# from typing import List, Union
# from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
# import logging
# import sys
# import pyizumo
# import copy

from docuverse import SearchEngine, SearchQueries
from docuverse.engines import SearchData
from docuverse.engines.search_engine_config_params import DocUVerseConfig
# from docuverse.utils import DenseEmbeddingFunction
from docuverse.utils.evaluator import EvaluationEngine
# from docuverse.utils.text_tiler import TextTiler

# from transformers.hf_argparser import HfArgumentParser

# nlp = None
# product_counts = {}
# import urllib3

#urllib3.disable_warnings()

# languages = ['en', 'es', 'fr', 'pt', 'ja', 'de', 'ja-new']

# settings = {}
# coga_mappings = {}
# docname2url = {}
# normalize_text = False
# tiler = None
# rouge_scorer = None
# default_cache_dir=os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")
# cache_dir = default_cache_dir

# def setup_argparse_old():
#     parser = ArgumentParser(description="Script to create/use ElasticSearch indices")
#     parser.add_argument('--input_passages', '-p', nargs="+", default=None)
#     parser.add_argument('--input_queries', '-q', default=None)
#     # Add documentation on configuration
#     parser.add_argument("--config", default=None, help="Path to the config file containing all parameters. Can be a mf-coga config. "
#                                                        "See documentation for examples.")
#
#     parser.add_argument('--db_engine', '-e', default='es-dense',
#                         choices=['es-dense', 'es-elser', 'es-bm25', 'es-dense'], required=False)
#     parser.add_argument('--output_file', '-o', default=None, help="The output rank file.")
#     parser.add_argument("--model_on_server", action="store_true",
#                         help="If present, the given model is assumed to exist on the ES server.")
#     parser.add_argument("--hybrid", choices=['rrf'], default="rrf",
#                         help="The type of hybrid combination")
#     parser.add_argument('--top_k', '-k', type=int, default=10, )
#     parser.add_argument('--model_name', '-m')
#     parser.add_argument('--actions', default="ir",
#                         help="The actions that can be done: i(ingest), r(retrieve), R(rerank), u(update), e(evaluate)")
#     parser.add_argument("--normalize_embs", action="store_true", help="If present, the embeddings are normalized.")
#     parser.add_argument("--evaluate", action="store_true",
#                         help="If present, evaluates the results based on test data, at the provided ranks.")
#     parser.add_argument("--ranks", default="1,5,10,100", help="Defines the R@i evaluation ranks.")
#     parser.add_argument('--data', default=None, type=str, help="The directory containing the data to use. The passage "
#                                                                "file is assumed to be args.data/psgs.tsv and "
#                                                                "the question file is args.data/questions.tsv.")
#     parser.add_argument("--data_type", default="auto", type=str, choices=["auto", 'pqa', 'sap', 'beir', 'rh'],
#                         help=("The type of the dataset to use. If auto, then the type will be determined"
#                               "by the file extension: .tsv->pqa, .json|.jsonl -> sap, csv -> SAP question"))
#     parser.add_argument("--ingestion_batch_size", default=40, type=int,
#                         help="For elastic search only, sets the ingestion batch "
#                              "size (default 40).")
#     parser.add_argument("--replace_links", action="store_true", default=False,
#                         help="If turned on, it will replace urls in text with URL<no>")
#     parser.add_argument("--max_doc_length", type=int, default=None,
#                         help="If provided, the documents will be split into chunks of <max_doc_length> "
#                              "*word pieces* (in regular English text, about 2 word pieces for every word). "
#                              "If not provided, the passages in the file will be ingested, truncated.")
#     parser.add_argument("--stride", type=int, default=None,
#                         help="Argument that works in conjunction with --max_doc_length: it will define the "
#                              "increment of the window start while tiling the documents.")
#     parser.add_argument("--max_num_documents", type=int, default=None,
#                         help="If defined, it will restrict the ingestion to the first <max_num_documents> documents")
#     parser.add_argument("--docid_map", nargs="+", default=None,
#                         help="If defined, this provides a link to a file mapping docid values to loio values.")
#     parser.add_argument("-I", "--index_name", type=str, default=None,
#                         help="Defines the index name to use. If not specified, it is built as " \
#                              "{args.data}_{args.db_engine}_{args.model_name if args.db_engine=='es-dense' else 'elser'}_index")
#     parser.add_argument("--doc_based", action="store_true", default=False,
#                         help="If present, the document text will be ingested, otherwise the ingestion will be done"
#                              " at passage level.")
#     parser.add_argument("--hana_file2url", nargs="+", default=None,
#                         help="The file mapping the docid to the url to the title")
#     parser.add_argument("--remove_stopwords", action="store_true", default=False,
#                         help="If defined, the stopwords are removed from text before indexing.")
#     parser.add_argument("--docids_to_ingest", default=None, help="If provided, only the documents with the "
#                                                                  "ids in the file will be added.")
#     parser.add_argument("--product_name", default=None, help="If set, this product name will be used "
#                                                              "for all documents")
#     parser.add_argument("--server", default="CONVAI",
#                         choices=['SAP', 'CONVAI', 'SAP_TEST', 'AILANG', 'local', 'RESCONVAI'],
#                         help="The server to connect to.")
#     parser.add_argument("--lang", "--language_code", default="en", choices=languages)
#     parser.add_argument("--host", default=None, help="Gives the IP for the ES server; can be used to "
#                                                      "override the defaults based on --server")
#     parser.add_argument("-N", "--normalize_text", default=False, action="store_true",
#                         help="If provided, the text is normalized for UTF8 encoding.")
#     parser.add_argument("--pyizumo_tokenization", action="store_true", default=False,
#                         help="If provided, will use the PyIzumo tokenizer to tokenize the text instead"
#                              " of the ES tokenizers.")
#     parser.add_argument("--compute_rouge", action="store_true",
#                         help="If provided, will compute the ROUGE score between the answers and the gold passage "
#                              "(note: it will be pretty slow for long documents).")
#     parser.add_argument("--duplicate_removal", choices=["none", "rouge", "exact"], default="none",
#                         help="Defines the strategy for removing duplicates (default: don't remove). It can be 'rouge' (based on rouge similarity) or 'exact' "
#                              "(exact match)")
#     parser.add_argument('--rouge_duplicate_threshold', default=-1, type=float,
#                         help="The rouge-l F1 similarity for dropping duplicated in the result (default 0.7)")
#     parser.add_argument("--title_handling", type=str, default="all", choices=["all", "first", "none"],
#                         help="Defines the policy of adding titles to the passages: can be 'all', 'first', or 'none'."
#                              "'all' will add the document title to every tile, 'first' will add to the first split tile, "
#                              "and 'none' will not add it at all.")
#
#     parser.add_argument("--cache_usage", type=bool, default=True, help="Turns on or off read caching.")
#     parser.add_argument("--cache_dir", type=str, default=default_cache_dir,
#                         help=f"Specifies the cache directory - by default {default_cache_dir}")
#     return parser
#
#
# def get_pyizumo_tokenized_text(text=None, tokenized_text=None, language_code=None):
#     tok_text = []
#
#     if not tokenized_text:
#         if text == None or language_code == None:
#             raise RuntimeError("text and language_code needed if tokenized_text not provided")
#         global nlp
#         if not nlp:
#             nlp = pyizumo.load(language_code, parsers=['token', 'sentence'])
#         tokenized_text = nlp(text)
#
#     for sent in tokenized_text.sentences:
#         sent_tokens = []
#         for tok in sent.tokens:
#             if 'components' in tok.properties:
#                 for part in tok.properties['components']:
#                     sent_tokens.append(part['text'])
#             else:
#                 sent_tokens.append(tok.text)
#         tok_text.append(" ".join(sent_tokens))
#
#     token_text = "\n".join(tok_text)
#
#     return token_text
#
#
# def compute_embedding(model: DenseEmbeddingFunction, input_query, normalize_embs):
#     query_vector = model.encode(input_query)
#     if normalize_embs:
#         query_vector = model.normalize([query_vector])[0]
#     return query_vector
#
#
# def compute_score(input_queries, results, compute_rouge_score=False):
#     if compute_rouge_score:
#         from rouge_score.rouge_scorer import RougeScorer
#         scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     if "relevant" not in input_queries[0] or input_queries[0]['relevant'] is None:
#         print("The input question file does not contain answers. Please fix that and restart.")
#         sys.exit(12)
#     ranks = [int(r) for r in args.ranks.split(",")]
#     scores = {r: 0 for r in ranks}
#     pscores = {r: 0 for r in ranks}  # passage scores
#     gt = {-1: -1}
#     for q in input_queries:
#         if isinstance(q['relevant'], list):
#             gt[q['id']] = {id: 1 for id in q['relevant']}
#         else:
#             gt[q['id']] = {q['relevant']: 1}
#
#     def skip(out_ranks, record, rid):
#         qid = record[0]
#         while rid < len(out_ranks) and out_ranks[rid][0] == qid:
#             rid += 1
#         return rid
#
#     def reverse_map(input_queries):
#         rq_map = {}
#         for i, q in enumerate(input_queries):
#             rq_map[q['id']] = i
#         return rq_map
#
#     def update_scores(ranks, rnk, val, op, scores):
#         j = 0
#         while j < len(ranks) and ranks[j] <= rnk:
#             j += 1
#         for k in ranks[j:]:
#             # scores[k] += 1
#             scores[k] = op([scores[k], val])
#
#     def get_doc_id(label):
#         # find - from right side because id may have -
#         index = label.rfind("-", 0, label.rfind("-"))
#         if index >= 0:
#             return label[:index]
#         else:
#             return label
#
#     rqmap = reverse_map(input_queries)
#
#     num_eval_questions = 0
#     for rid, record in tqdm(enumerate(results),
#                             total=len(results),
#                             desc='Evaluating questions: '):
#         qid = record['qid']
#         query = input_queries[rqmap[qid]]
#         if '-1' in gt[qid]:
#             continue
#         num_eval_questions += 1
#         tmp_scores = {r: 0 for r in ranks}
#         tmp_pscores = {r: 0 for r in ranks}
#         for aid, answer in enumerate(record['answers']):
#             if aid >= ranks[-1]:
#                 break
#             docid = get_doc_id(answer['id'])
#
#             if str(docid) in gt[qid]:  # Great, we found a match.
#                 update_scores(ranks, aid, 1, sum, tmp_scores)
#             if not compute_rouge_score:
#                 continue
#             if len(query['passages']) == 0:
#                 scr = 0.
#             else:
#                 scr = max(
#                     [
#                         rouge_scorer.score(passage, answer['text'])['rouge1'].recall for passage in query['passages']
#                     ]
#                 )
#             update_scores(ranks, aid, scr, max, tmp_pscores)
#
#         for r in ranks:
#             scores[r] += int(tmp_scores[r] >= 1)
#             if compute_rouge_score:
#                 pscores[r] += tmp_pscores[r]
#
#     _result = {"num_ranked_queries": num_eval_questions,
#                "num_judged_queries": num_eval_questions,
#                "doc_scores":
#                    {r: int(1000 * scores[r] / num_eval_questions) / 1000.0 for r in ranks},
#                }
#     if compute_rouge_score:
#         _result['rouge_scores'] = \
#             {r: int(1000 * pscores[r] / num_eval_questions) / 1000.0 for r in ranks}
#
#     return _result
#
#
#
# def init_settings():
#     def union(a, b):
#         if a == {}:
#             return copy.deepcopy(b)
#         else:
#             c = copy.deepcopy(a)
#             for k in b.keys():
#                 if k in a:
#                     c[k] = union(a[k], b[k])
#                 else:
#                     c[k] = copy.deepcopy(b[k])
#         return c
#
#     global settings, coga_mappings, standard_mappings
#     config = json.load(open(f"{os.path.join(os.path.dirname(__file__), 'elastic_config.json')}"))
#     standard_mappings = config['settings']['standard']
#     for lang in languages:
#         settings[lang] = union(config['settings']['common'],
#                                config['settings'][lang if lang in config['settings'] else 'en']
#                                )
#         coga_mappings[lang] = union(config['mappings']['common'],
#                                     config['mappings'][lang if lang in config['mappings'] else 'en']
#                                     )
#
#
# def init_settings_lang(lang):
#     global settings, coga_mappings
#
#     lang_stemmer = {
#         'es': "light_spanish",
#         'fr': "light_french",
#         'en': "light_english",
#         'de': "light_german",
#         'pt': "light_portuguese",
#     }
#
#     lang_stop = {
#         'es': "_spanish_",
#         'fr': "_french_",
#         'en': ["a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at",
#                "be", "been", "but", "by", "can", "de", "did", "do", "does", "for", "from",
#                "had", "has", "have", "he", "her", "him", "his", "how", "if", "in", "into",
#                "is", "it", "its", "more", "my", "nbsp", "new", "no", "non", "not", "of",
#                "on", "one", "or", "other", "our", "she", "so", "some", "such", "than",
#                "that", "the", "their", "then", "there", "these", "they", "this", "those",
#                "thus", "to", "up", "us", "use", "was", "we", "were", "what", "when", "where",
#                "which", "while", "why", "will", "with", "would", "you", "your", "yours"],
#         'de': "_german_",
#         'pt': "_portuguese_",
#     }
#
#     settings = {
#         "number_of_replicas": 0,
#         "number_of_shards": 1,
#         "refresh_interval": "1m",
#         "analysis": {
#             "filter": {
#                 "light_stemmer": {
#                     "type": "stemmer",
#                     "language": lang_stemmer[lang]
#                 },
#                 "lang_stop": {
#                     "ignore_case": "true",
#                     "type": "stop",
#                     "stopwords": lang_stop[lang]
#                 }
#             },
#             "char_filter": {
#                 "icu_normalize": {
#                     "type": "icu_normalizer",
#                     "name": "nfkc",
#                     "mode": "compose"
#                 }
#             },
#             "analyzer": {
#                 "text_no_stop": {
#                     "filter": [
#                         "lowercase",
#                         "light_stemmer"
#                     ],
#                     "tokenizer": "whitespace" if args.pyizumo_tokenization else "standard",
#                     "char_filter": [
#                         "icu_normalize"
#                     ],
#                 },
#                 "text_stop": {
#                     "filter": [
#                         "lowercase",
#                         "lang_stop",
#                         "light_stemmer"
#                     ],
#                     "tokenizer": "whitespace" if args.pyizumo_tokenization else "standard",
#                     "char_filter": [
#                         "icu_normalize"
#                     ],
#                 },
#                 "whitespace_lowercase": {
#                     "tokenizer": "whitespace",
#                     "filter": [
#                         "lowercase"
#                     ]
#                 }
#             },
#             "normalizer": {
#                 "keyword_lowercase": {
#                     "filter": [
#                         "lowercase"
#                     ]
#                 }
#             }
#         }
#     }
#     coga_mappings = {
#         "_source": {
#             "enabled": "true"
#         },
#         "dynamic": "false",
#         "properties": {
#             "url": {
#                 "type": "text"
#             },
#             "title": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "fileTitle": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "title_paraphrases": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "productId": {
#                 "type": "keyword"
#             },
#             "deliverableLoio": {
#                 "type": "keyword",
#             },
#             "filePath": {
#                 "type": "keyword",
#             },
#             "text": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "plainTextContent": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "title_and_text": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "app_name": {
#                 "type": "text",
#                 "analyzer": "text_no_stop",
#                 "search_analyzer": "text_stop",
#                 "term_vector": "with_positions_offsets",
#                 "index_options": "offsets",
#                 "store": "true"
#             },
#             "collection": {
#                 "type": "text",
#                 "fields": {
#                     "exact": {
#                         "normalizer": "keyword_lowercase",
#                         "type": "keyword",
#                         "doc_values": "false"
#                     }
#                 }
#             }}
#     }
#
#
# def create_es_client(fingerprint, api_key, host):
#     # global ES_SSL_FINGERPRINT, ES_API_KEY, client
#     ES_SSL_FINGERPRINT = os.getenv(fingerprint)
#     ES_API_KEY = os.getenv(api_key)
#     _client = Elasticsearch(f"https://{host}:9200",
#                             ssl_assert_fingerprint=(ES_SSL_FINGERPRINT),
#                             api_key=ES_API_KEY,
#                             request_timeout=60
#                             )
#     try:
#         _ = _client.info()
#     except Exception as e:
#         print(f"Error: {e}")
#         raise e
#
#     return _client
#
#
# def extract_answers(res):
#     rout = []
#     for rank, r in enumerate(res):
#         rout.append({'id': r['_id'], 'score': r['_score'], 'text': r['_source']['text']})
#     return rout
#
#
# def build_elastic_query(qid, text, db_engine, model_name, hybrid_mode, model_on_server, vector_field_name):
#     _knn = None
#     _query = None
#     _rank = None
#     if db_engine in ["es-dense"]:
#         _knn = {
#             "field": vector_field_name,
#             "k": args.top_k,
#             "num_candidates": 1000,
#         }
#         if model_on_server:
#             _knn["query_vector_builder"] = {
#                 "text_embedding": {
#                     "model_id": args.model_name,
#                     "model_text": text
#                 }
#             }
#         else:
#             _knn['query_vector'] = compute_embedding(model, text, args.normalize_embs)
#
#         if hybrid_mode == "rrf":
#             _query = {
#                 "bool": {
#                     "must": {
#                         "multi_match": {
#                             "query": query_text,
#                             "fields": ['text', 'title']
#                         }
#                     },
#                 }
#             }
#             _rank = {"rrf": {
#                 "window_size": 200
#             }}
#     elif args.db_engine == 'es-bm25':
#         _query = {
#             "bool": {
#                 "must": {
#                     "multi_match": {
#                         "query": text,
#                         "fields": ['text', 'title']
#                     }
#                 },
#             }
#         }
#     elif args.db_engine == "es-elser":
#         _query = {
#             "text_expansion": {
#                 "ml.tokens": {
#                     "model_id": args.model_name,
#                     "model_text": query_text
#                 }
#             }
#         }
#     return _query, _knn, _rank
#
#
# def remove_duplicates(results, duplicate_removal, rouge_duplicate_threshold):
#     res = results
#     if duplicate_removal == "none":
#         return res
#     if len(res) == 0:
#         return results
#     ret = []
#     if duplicate_removal == "exact":
#         seen = {res[0]['_source']['text']: 1}
#         ret = [res[0]]
#         for r in res[1:]:
#             text_ = r['_source']['text']
#             if text_ not in seen:
#                 seen[text_] = 1
#                 ret.append(r)
#     elif duplicate_removal == "rouge":
#         for r in res[1:]:
#             found = False
#             text_ = r['_source']['text']
#             for c in ret:
#                 scr = rouge_scorer.score(c['_source']['text'], text_)
#                 if scr['rougel'].fmeasure >= rouge_duplicate_threshold:
#                     found = True
#                     break
#             if not found:
#                 ret.append(r)
#     return ret
#
#
# def get_keys_to_index(all_keys_to_index):
#     keys_to_index = []
#     for k in all_keys_to_index:
#         if k not in input_passages[0]:
#             print(f"Dropping key {k} - they are not in the passages")
#         else:
#             keys_to_index.append(k)
#     return keys_to_index
#
# def old_main():
#     from datetime import datetime
#
#     with open("logfile", "a") as cmdlog:
#         cmdlog.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {os.getenv('USER')} - "
#                      f"{' '.join(sys.argv)}\n")
#     parser = setup_argparse_old()
#
#     args = parser.parse_args()
#
#     if not args.model_name:
#         if args.db_engine == 'es-elser':
#             args.model_name = ".elser_model_1"
#         elif args.db_engine.startswith('es-dense'):
#             args.model_name = 'all-MiniLM-L6-v2'
#
#     if args.data_type == "beir":
#         if args.input_passages is None:
#             args.input_passages = os.path.join(args.data, "corpus.jsonl")
#         if args.input_queries is None:
#             args.input_queries = os.path.join(args.data, "queries.jsonl")
#
#     if args.duplicate_removal == "rouge":
#         if args.rouge_duplicate_threshold < 0:
#             args.rouge_duplicate_threshold = 0.7
#     elif args.rouge_duplicate_threshold > 0:
#         args.duplicate_removal = "rouge"
#
#     if args.compute_rouge or args.duplicate_removal == "rouge":
#         from rouge_score.rouge_scorer import RougeScorer
#
#         rouge_scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     cache_dir = args.cache_dir
#     use_cache = args.cache_usage
#     print(f"Using cache dir: {cache_dir}")
#     server_map = {
#         'CONVAI': "convaidp-nlp.sl.cloud9.ibm.com",
#         'AILANG': "ai-lang-conv-es.sl.cloud9.ibm.com",
#         'local': "localhost",
#         'RESCONVAI': "convai.sl.res.ibm.com"
#     }
#
#     normalize_text = args.normalize_text
#
#     if args.host is None:
#         if args.server == "SAP" or args.server == "SAP_TEST":
#             print("The SAP server has been retired. Use CONVAI or AILANG")
#             sys.exit(33)
#         if args.server in server_map.keys():
#             args.host = server_map[args.server]
#         else:
#             print(f"Server {args.server} is not known!")
#             sys.exit(12)
#
#     if args.index_name is None:
#         index_name = (
#             f"{args.data}_{args.db_engine}_{args.model_name if args.db_engine == 'es-dense' else 'elser' if args.db_engine == 'es-elser' else 'bm25'}_index").lower()
#     else:
#         index_name = args.index_name.lower()
#
#     index_name = re.sub('[^a-z0-9]', '-', index_name)
#
#     if not nlp:
#         nlp = pyizumo.load(args.lang, parsers=['token', 'sentence'])
#
#     do_ingest = 'i' in args.actions
#     do_retrieve = 'r' in args.actions
#     do_rerank = 'R' in args.actions
#     do_update = 'u' in args.actions
#     doc_based_ingestion = args.doc_based
#     docname2url = {}
#     docname2title = {}
#     if args.hana_file2url is not None:
#         for file_ in args.hana_file2url:
#             with open(file_) as inp:
#                 # fl = csv.reader(inp, delimiter="\t")
#                 for ln in inp.readlines():
#                     line = ln.strip().split("\t")
#                     docname2url[line[0]] = line[1]
#                     docname2title[line[0]] = line[2].strip()
#
#     docid_filter = []
#     if args.docids_to_ingest is not None:
#         with open(args.docids_to_ingest) as inp:
#             for line in inp:
#                 line = line.replace(".txt", "").strip()
#                 docid_filter.append(line)
#
#     docid2loio = {}
#     if args.docid_map is not None:
#         for file_ in args.docid_map:
#             with open(file_) as inp:
#                 for line in inp:
#                     a = line.split()
#                     docid2loio[a[0]] = a[1]
#
#     model = None
#     if args.db_engine in ['es-dense'] or args.max_doc_length is not None:
#         import torch
#
#         batch_size = 64
#
#         if args.db_engine == 'es-elser':
#             model = DenseEmbeddingFunction('all-MiniLM-L6-v2')
#         else:
#             if args.model_on_server:
#                 if args.model_name.find("multilingual-e5"):
#                     model = DenseEmbeddingFunction("intfloat/multilingual-e5-base")
#                 else:
#                     model = DenseEmbeddingFunction(args.model_name)
#             else:
#                 model = DenseEmbeddingFunction(args.model_name)
#
#     tiler = TextTiler(max_doc_size=args.max_doc_length,
#                       stride=args.stride,
#                       tokenizer=model.tokenizer)
#
#     print(f"Using the {args.server} server at {args.host}")
#
#     server_labels = {
#         "CONVAI": "ES",
#         "AILANG": "AILANG",
#         "RESCONVAI": "RESCONVAI",
#         "local": "LOCAL"
#     }
#
#     if args.server in server_labels:
#         server_ = server_labels[args.server]
#         client = create_es_client(f"{server_}_SSL_FINGERPRINT", f"{server_}_API_KEY",
#                                   host=args.host)
#     else:
#         print(f"Server {args.server} is unknown. Exiting..")
#         sys.exit(12)
#
#     init_settings()
#     stopwords = None
#     if do_ingest or do_update:
#         max_documents = args.max_num_documents
#
#         input_passages = read_data(input_files=args.input_passages,
#                                    tiler=tiler,
#                                    lang=args.lang,
#                                    fields=["id", "text", "title"],
#                                    remove_url=args.replace_links,
#                                    max_doc_size=args.max_doc_length,
#                                    stride=args.stride,
#                                    tokenizer=model.tokenizer if model is not None else None,
#                                    max_num_documents=max_documents,
#                                    doc_based=doc_based_ingestion,
#                                    docname2url=docname2url,
#                                    docname2title=docname2title,
#                                    remove_stopwords=args.remove_stopwords,
#                                    docid_filter=docid_filter,
#                                    uniform_product_name=args.product_name,
#                                    data_type=args.data_type,
#                                    docid_map=docid2loio,
#                                    cache_dir=cache_dir,
#                                    use_cache=use_cache,
#                                    title_handling=args.title_handling
#                                    )
#
#         hidden_dim = -1
#         passage_vectors = []
#         if args.db_engine in ['es-dense']:
#             if args.model_on_server:
#                 if 'ml' in client.__dict__:
#                     r = client.ml.get_trained_models(model_id=args.model_name)
#                     hidden_dim = r['trained_model_configs'][0]['inference_config']['text_embedding']['embedding_size']
#                 else:
#                     hidden_dim = 384  # Some default value, the system might crash if it's wrong.
#             else:
#                 print("Encoding corpus documents:")
#                 passage_vectors = model.encode([passage['text'] for passage in input_passages], _batch_size=batch_size)
#
#                 hidden_dim = len(passage_vectors[0])
#                 if args.normalize_embs:
#                     passage_vectors = model.normalize(passage_vectors)
#
#         logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
#
#         if args.pyizumo_tokenization:
#             for passage in tqdm(input_passages, desc="Getting pyizumo word tokens"):
#                 passage['title'] = get_pyizumo_tokenized_text(text=passage['title'], language_code=args.language_code)
#                 passage['text'] = get_pyizumo_tokenized_text(text=passage['text'], language_code=args.language_code)
#
#         vector_field_name = None
#         all_keys_to_index = ['title', 'id', 'url', 'productId',  # 'versionId',
#                          'filePath', 'deliverableLoio', 'text', 'app_name', 'courseGrainedProductId']
#         if args.db_engine in ['es-dense', 'es-bm25']:
#             mappings = coga_mappings[args.lang]
#             processors = []
#             if args.db_engine in ['es-dense']:
#                 if args.model_on_server:
#                     vector_field_name = "ml"
#                     pipeline_name = f"{args.model_name}-test"
#                     processors = [{
#                         "inference": {
#                             "model_id": args.model_name,
#                             "target_field": "ml",
#                             "field_map": {
#                                 "text": "text_field"
#                             }
#                         }
#                     }]
#                     on_failure = [{
#                         "set": {
#                             "description": "Index document to 'failed-<index>'",
#                             "field": "_index",
#                             "value": "failed-{{{_index}}}"
#                         }
#                     },
#                         {
#                             "set": {
#                                 "description": "Set error message",
#                                 "field": "ingest.failure",
#                                 "value": "{{_ingest.on_failure_message}}"
#                             }
#                         }]
#                     vector_field_name = f"ml.predicted_value"
#                 else:
#                     vector_field_name = "vector"
#                     pipeline_name = None
#                     on_failure = None
#
#                 mappings['properties'][vector_field_name] = {
#                     "type": "dense_vector",
#                     "similarity": "cosine",
#                     "dims": hidden_dim,
#                     "index": "true"
#                 }
#
#             create_update_index(client, index_name, settings=settings[args.lang],
#                                 mappings=mappings,
#                                 do_update=do_update)
#
#             if args.model_on_server and len(processors) > 0:
#                 client.ingest.put_pipeline(processors=processors, id=pipeline_name, on_failure=on_failure)
#
#             logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
#             bulk_batch = args.ingestion_batch_size
#
#             num_passages = len(input_passages)
#             t = tqdm(total=num_passages, desc="Ingesting dense documents: ", smoothing=0.05)
#             # print(input_passages[0].keys())
#             keys_to_index = get_keys_to_index(all_keys_to_index=all_keys_to_index)
#             for k in range(0, num_passages, bulk_batch):
#                 actions = [
#                     {
#                         "_index": index_name,
#                         "_id": row['id'],
#                         "_source": {k: row[k] for k in keys_to_index}
#                     }
#                     for pi, row in enumerate(input_passages[k:min(k + bulk_batch, num_passages)])
#                 ]
#                 if args.db_engine in ['es-dense']:
#                     if not args.model_on_server:
#                         for pi, (action, row) in enumerate(
#                                 zip(actions, input_passages[k:min(k + bulk_batch, num_passages)])):
#                             action["_source"]['vector'] = passage_vectors[pi + k]
#                 try:
#                     bulk(client, actions=actions, pipeline=pipeline_name)
#                 except Exception as e:
#                     print(f"Got an error in indexing: {e}")
#                 t.update(bulk_batch)
#             t.close()
#             if len(actions) > 0:
#                 try:
#                     bulk(client=client, actions=actions, pipeline=pipeline_name)
#                 except Exception as e:
#                     print(f"Got an error in indexing: {e}, {len(actions)}")
#         elif args.db_engine == "es-elser":
#             mappings = coga_mappings[args.lang]
#             mappings['properties']['ml.tokens'] = {"type": "rank_features"}
#
#             processors = [
#                 {
#                     "inference": {
#                         "model_id": args.model_name,
#                         "target_field": "ml",
#                         "field_map": {
#                             "text": "text_field"
#                         },
#                         "inference_config": {
#                             "text_expansion": {
#                                 "results_field": "tokens"
#                             }
#                         }
#                     }}
#             ]
#             bulk_batch = args.ingestion_batch_size
#             create_update_index(client,
#                                 index_name=index_name,
#                                 settings=settings[args.lang],
#                                 mappings=mappings,
#                                 do_update=do_update)
#
#             client.ingest.put_pipeline(processors=processors, id=args.model_name + "-test")
#             actions = []
#             keys_to_index = get_keys_to_index(all_keys_to_index)
#             num_passages = len(input_passages)
#             t = tqdm(total=num_passages, desc="Ingesting documents (w ELSER): ", smoothing=0.05)
#             # for ri, row in tqdm(enumerate(input_passages), total=len(input_passages), desc="Indexing passages"):
#             for k in range(0, num_passages, bulk_batch):
#                 actions = [
#                     {
#                         "_index": index_name,
#                         "_id": row['id'],
#                         "_source": {k: row[k] for k in keys_to_index}
#                     }
#                     for pi, row in enumerate(input_passages[k:min(k + bulk_batch, num_passages)])
#                 ]
#
#                 failures = 0
#                 while failures < 5:
#                     try:
#                         res = bulk(client=client, actions=actions, pipeline=args.model_name + "-test")
#                         break
#                     except Exception as e:
#                         print(f"Got an error in indexing: {e}, {len(actions)}")
#                     failures += 5
#                 t.update(bulk_batch)
#             t.close()
#
#             if len(actions) > 0:
#                 try:
#                     bulk(client=client, actions=actions, pipeline=args.model_name + "-test")
#                 except Exception as e:
#                     print(f"Got an error in indexing: {e}, {len(actions)}")
#
#             tiler.print_product_histogram()
#     ### QUERY TIME
#
#     if do_retrieve:
#         loio2docid = {}
#         if args.docid_map is not None:
#             with open(args.docid_map) as inp:
#                 for line in inp:
#                     a = line.split()
#                     loio2docid[a[1]] = a[0]
#
#         if args.evaluate:
#             input_queries, unmapped_ids = read_data(args.input_queries,
#                                                     lang=args.lang,
#                                                     fields=["id", "text", "relevant", "answers"],
#                                                     docid_map=loio2docid, return_unmapped=True,
#                                                     remove_stopwords=args.remove_stopwords,
#                                                     data_type=args.data_type,
#                                                     doc_based=doc_based_ingestion,
#                                                     cache_dir=cache_dir)
#             print("Unmapped ids:", unmapped_ids)
#         else:
#             input_queries = read_data(args.input_queries,
#                                       lang=args.lang,
#                                       fields=["id", "text"],
#                                       remove_stopwords=args.remove_stopwords,
#                                       data_type=args.data_type,
#                                       doc_based=doc_based_ingestion,
#                                       cache_dir=cache_dir)
#
#         if not do_ingest:
#             vector_field_name = "ml.predicted_value" if args.model_on_server else "vector"
#
#         result = []
#         for query in tqdm(input_queries):
#             qid = query['id']
#             query_text = query['text']
#             query, knn, rank = build_elastic_query(qid=qid,
#                                                    db_engine=args.db_engine,
#                                                    text=query_text,
#                                                    model_name=args.model_name,
#                                                    hybrid_mode=args.hybrid,
#                                                    vector_field_name=vector_field_name,
#                                                    model_on_server=args.model_on_server)
#             res = client.search(
#                 index=index_name,
#                 knn=knn,
#                 query=query,
#                 rank=rank,
#                 size=args.top_k,
#                 source_excludes=['vector', 'ml.predicted_value']
#             )
#             res = remove_duplicates(res._body['hits']['hits'],
#                                     args.duplicate_removal,
#                                     args.rouge_duplicate_threshold)
#             result.append({'qid': qid, 'text': query_text,
#                            "answers": extract_answers(res)})
#
#         if do_rerank:
#             pass
#
#         if args.output_file is not None:
#             if args.output_file.endswith(".json"):
#                 with open(args.output_file, 'w', encoding="utf8") as out:
#                     json.dump(result, out, indent=2, ensure_ascii=False)
#             elif args.output_file.endswith(".jsonl"):
#                 with open(args.output_file, 'w', encoding="utf8") as out:
#                     for r in result:
#                         json.dump(r, out, ensure_ascii=False)
#                         out.write("\n")
#
#         if args.evaluate:
#             score = compute_score(input_queries, result)
#
#             with open(args.output_file.replace(".json", ".metrics"), "w") as out:
#                 json.dump(score, out, indent=2)
#
if __name__ == '__main__':
    config = DocUVerseConfig.get_stdargs()

    engine = SearchEngine(config.search_config)
    scorer = None

    if config.evaluate and config.eval_config is not None:
        scorer = EvaluationEngine(config.eval_config)

    if config.ingest or config.update:
        corpus = SearchData.read_data(config.input_passages, **vars(config.search_config))
        engine.ingest(corpus)

    if config.retrieve:
        queries = SearchQueries.read(config.input_queries, **vars(config.search_config))
        output = engine.search(queries)
        output.save(config.output_file)
        if config.evaluate:
            results = scorer.compute_score(queries, output)
            print(f"Results:\n{results}")