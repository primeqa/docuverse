import glob as glob_module
import itertools
import os
import orjson
import csv
import re
import time
from itertools import chain
from functools import partial
from typing import Dict, List
from tqdm import tqdm
import queue
from docuverse.utils import at_most, open_stream, file_is_of_type, parallel_process, get_orig_docid
import pickle

from docuverse.engines import data_template
from docuverse.engines.data_template import (
    DataTemplate,
    default_data_template,
    sap_data_template,
    beir_data_template
)
from docuverse.utils.text_tiler import TextTiler
from docuverse.utils import get_param


# Module-level function for computing tokenized length (needed for pickling in multiprocessing)
def _compute_tokenized_length(itm, tiler):
    """Compute tokenized length for an item. This function is defined at module level to be picklable."""
    return tiler.get_tokenized_length(itm['text'], forced_tok=True)


class DefaultProcessor:
    product_counts = {}
    stopwords = None

    @classmethod
    def cleanup(cls, text, lang='en', remv_stopwords=False):
        return cls.remove_stopwords(text, lang=lang, do_replace=remv_stopwords)

    def __init__(self, data_template: DataTemplate = default_data_template, _stopwords=None, lang: str = "en"):
        self.template = data_template
        self.lang = lang
        self.read_stopwords()

    @staticmethod
    def read_stopwords():
        if DefaultProcessor.stopwords is None:
            stopword_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                         "resources", "stopwords.json")
            stopwords_list = orjson.loads("".join(open(stopword_file).readlines()))
            stopwords = {}
            for lang, vals in stopwords_list.items():
                stopwords[lang] = re.compile(f"\\b({'|'.join(vals)})\\b", re.IGNORECASE)

    def _init(self, **kwargs):
        pass

    def __call__(self, unit, id, data_template, **kwargs):
        itm = {
            'id': id,
            'title': self.cleanup(get_param(unit, data_template.title_header, "")),
            'text': self.cleanup(get_param(unit, data_template.text_header))
        }
        url = get_param(unit, 'document_url|url', "")
        if url != "":
            itm['url'] = url
        if data_template.extra_fields is not None:
            for key in data_template.extra_fields:
                if key in unit:
                    itm[key] = unit[key]
        return itm

    @classmethod
    def remove_stopwords(cls, text: str, lang: str = "en", do_replace: bool = False) -> str:
        if not do_replace or cls.stopwords[lang] is None:
            return text
        else:
            return re.sub(r' {2,}', ' ', re.sub(cls.stopwords[lang], " ", text))

    @staticmethod
    def increment_product_counts(product_id):
        if product_id not in DefaultProcessor.product_counts:
            DefaultProcessor.product_counts[product_id] = 1
        else:
            DefaultProcessor.product_counts[product_id] += 1


class SAPProccesor(DefaultProcessor):

    def __init__(self, data_template: DataTemplate = sap_data_template, hana_file2url: List[str] = None):
        super().__init__(data_template)
        self.docname2url_title = {}
        self._init(hana_file2url=hana_file2url)

    def _init(self, hana_file2url, **kwargs):
        if hana_file2url is not None:
            for file_ in hana_file2url:
                with open(file_) as inp:
                    # fl = csv.reader(inp, delimiter="\t")
                    for ln in inp.readlines():
                        line = ln.strip().split("\t")
                        self.docname2url_title[line[0]] = [line[1], line[2].strip()]

    @staticmethod
    def process_product_id(url_fields, uniform_product_name, data_type):
        """
        Process the product ID based on the given fields, uniform product name, and data type.

        Parameters:
        - fields (list): A list of fields.
        - uniform_product_name (str): The uniform product name.
        - data_type (str): The data type.

        Returns:
        - str: The processed product ID.

        Example Usage:
        ```python
        fields = ["field1", "field2", "field3"]
        uniform_product_name = "Uniform Product"
        data_type = "sap"

        result = process_product_id(fields, uniform_product_name, data_type)
        print(result)  # Output: ""

        data_type = "other"

        result = process_product_id(fields, uniform_product_name, data_type)
        print(result)  # Output: ""
        ```
        """
        if data_type == "sap":
            product_id_ = "" if len(url_fields) == 0 \
                else url_fields[-3] if (len(url_fields) > 3 and url_fields[-3] != '#') \
                else 'SAP_BUSINESS_ONE'
            if uniform_product_name:
                product_id_ = uniform_product_name
            if product_id_.startswith("SAP_SUCCESSFACTORS"):
                product_id_ = "SAP_SUCCESSFACTORS"
            return product_id_
        else:
            return ""

    @staticmethod
    def get_course_product_id(product_id):
        """
        @param product_id: The product ID used to determine the course product ID for SAP.
        @return: The course product ID based on the given product ID.
        """
        if product_id.find('S4HANA') >= 0:
            return "S4"
        elif product_id.find("SUCCESS_FACTORS") >= 0:
            return "SFSF"
        elif product_id.find("BUSSINESONE") >= 0:
            return "B1"
        else:
            return product_id

    @staticmethod
    def process_url(doc_url: str, data_type: str = ""):
        """
            process_url(doc_url:str, data_type:str="") -> Tuple[str, List[str]]
            This method processes a given URL and returns the modified URL and a list of fields.
            Parameters:
            - doc_url (str): The URL to be processed.
            - data_type (str): Optional. The data type to specify the processing. Default is an empty string.

            Returns:
            - Tuple[str, List[str]]: A tuple containing the processed URL and a list of fields.

            Example:
            >>> doc_url = "https://example.com/some_document.html?locale=en-US"
            >>> data_type = "sap"
            >>> self.process_url(doc_url, data_type)
            ("https://example.com/some_document", ["https:", "", "example.com", "some_document"])

            Note:
            - If the data_type is "sap", the URL is modified by removing the query string, replacing the last part
            with a filename (removing the .html extension), and returning the modified URL and a list of fields
            extracted from the modified URL.
            - If the data_type is not "sap", an empty string and a list of empty strings are returned.

        """
        if data_type == "sap":
            url = re.sub(r'\?locale=.*', "", doc_url)
            fields = url.split("/")
            fields[-1] = fields[-1].replace(".html", "")
            return url, fields
        else:
            return "", ["", "", "", "", "", ""]

    @staticmethod
    def fix_title(title: str):
        return re.sub(r' {2,}', ' ', title.replace(" | SAP Help Portal", ""))

    @staticmethod
    def find_document_id(args):
        for docid_name in ['document_id', 'docid', 'id']:
            if docid_name in args:
                return args[docid_name]
        return ""

    def __call__(self, **kwargs):
        doc_url = kwargs.get("doc_url")
        data_type = kwargs.get("data_type", "")

        full_url, fields = self.process_url(doc_url, data_type)
        uniform_product_name = kwargs.get("uniform_product_name", "")
        product_id = self.process_product_id(fields, uniform_product_name, data_type)
        title = kwargs.get(self.template.title_header)
        docid = str(self.find_document_id(kwargs))
        if docid != "":
            if docid.endswith(".txt"):
                docid = docid[:-4]
            if docid in self.docname2url_title:
                url, title = self.docname2url_title[docid]

        self.increment_product_counts(product_id)

        return {
            'productId': product_id,
            'deliverableLoio': ("" if doc_url == "" else fields[-2]),
            'filePath': "" if doc_url == "" else fields[-1],
            'title': title,
            'url': doc_url,
            'app_name': "",
            'courseGrainedProductId': self.get_course_product_id(product_id)
        }


class SearchData:
    """
    An abstraction of the search data, encompassing SearchCorpus, SearchQueries, and SearchResult. It has named fields
    for 'text' and 'title', and knows how to read the data from disk. Other than that, it's a glorified dictionary.
    """
    default_cache_dir = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")
    processor_map = {
        'default': DefaultProcessor(),
        'auto': DefaultProcessor(),
        'sap': SAPProccesor()
    }

    default_tiler = TextTiler(tokenizer=None, count_type="char", max_doc_length=20000, stride=2000,
                              aligned_on_sentences=False)

    class Entry:
        def __init__(self, config: Dict[str, str]):
            self.__dict__.update(config)

        def get_text(self):
            return getattr(self, "text")

        def get_id(self):
            return getattr(self, "id")

    def __init__(self, filenames,
                 data_template=default_data_template,
                 **kwargs):
        self.entries = []
        self.tiler = None
        self.template = data_template

    def get_text(self, i: int) -> str:
        return self.entries[i][self.template.text_header]

    def __getitem__(self, i: int) -> Entry:
        return SearchData.Entry(**self.entries[i])

    def __len__(self) -> int:
        return len(self.entries)

    def get_cached_filename(input_file: str,
                            max_doc_size: int,
                            stride: int,
                            aligned: bool = True,
                            tiler: TextTiler = None,
                            title_handling="all",
                            cache_dir: str = default_cache_dir):
        def prune_list(list):
            return [d for d in list if d]
        tok_dir_name = os.path.basename(tiler.tokenizer.name_or_path) \
            if (tiler is not None and tiler.tokenizer is not None) \
            else "none"
        if tok_dir_name == "":
            tok_dir_name = os.path.basename(os.path.dirname(tiler.tokenizer.name_or_path))
        extension = "pickle.xz"
        cache_file_name = os.path.join(cache_dir,
                                       "_".join(
                                           prune_list([
                                               f"{input_file.replace('/', '__')}",
                                               f"{max_doc_size}",
                                               f"{stride}",
                                               f"{aligned}" if aligned else "unaligned",
                                               f"{title_handling}",
                                               f"trim={tiler.text_trim_to}" if tiler.text_trim_to else "",
                                               f"{tok_dir_name}.{extension}"
                                           ])
                                       ))
        print(f"Cache filename is {cache_file_name}")
        return cache_file_name

    @staticmethod
    def read_cache_file_if_needed(cache_file_name, input_file):
        passages = []

        if (input_file is None or
                os.path.exists(cache_file_name) and os.path.getmtime(cache_file_name) > os.path.getmtime(input_file)):
            input_stream = open_stream(cache_file_name, write=False, binary=True)
            # for line in tqdm(input_stream, desc="Reading cache file:"):
            #     passages.append(orjson.loads(line.decode('utf-8'))
            passages = pickle.load(input_stream)
            input_stream.close()

        return passages

    @staticmethod
    def write_cache_file(cache_filename, passages, use_cache=True):
        if not use_cache:
            return
        output_stream = open_stream(cache_filename, write=True, binary=True)
        pickle.dump(passages, output_stream)
        # for p in passages:
        #     output_stream.write(f"{orjson.dumps(p)}\n".encode("utf-8"))
        output_stream.close()

    @classmethod
    def process_text(cls, unit:str, tiler, max_doc_size, stride, id=None, remove_url=True, doc_url=None,
                     uniform_product_name=None, data_type="sap", title_handling="all", processor=None,
                     data_template=default_data_template, docid_filter={}, doc_based=True):
        """
        Convert a given document or passage (from 'output.json') to a dictionary, splitting the text as necessary.
        :param unit: the paragraph/document structure to proces
        :param tiler: Tiler instance - the Tiler object that creates the tiles from the input text
        :param max_doc_size: int - the maximum size (in word pieces) of the resulting sub-document/sub-passage texts
        :param stride: int - the stride/overlap for consecutive pieces
        :param remove_url: Boolean - if true, URL in the input text will be replaced with "URL"
        :param id: the output id, if needed (by default, taken from the input unit)
        :param tokenizer: Tokenizer - the tokenizer to use while splitting the text into pieces
        :param doc_url: str - the url of the document.
        :param uniform_product_name: str - if not None, all documents will receive this productId
        :param data_type: str - the type of data being read. One of "sap" or 'default', with more types
               possibly being added.
        :param title_handling: str - one of "all", "first", or "none" - defines how the title is being handled:
               * "all" - titles are added to all pieces of the text
               * "first" - titles are added to the first piece of the split text
               * "none" - titles are not added to the any piece of the split text.
        :param processor: Processor - the processor to use to extract the additional keys for indexing. By default, only
               'text' and 'title' are extracted, but for the 'sap' data we also extract productId, coarseProductId,
               'deliverableLoio', 'filePath', 'url', 'appname', etc.
        :param data_template: DataTemplate - the template for the data - it defines what fields to look for (e.g.,
               'title', 'text')
        :param docid_filter: dict - the filter for document IDs to ingest
        :param doc_based: Boolean - if true, processing is done document based, otherwise it's paragraph based (tiles
               from paragraphs are ingested as units, instead of tiles extracted from the document)

        :return - a list of indexable items, each containing a title, id, text, and url.
        """

        if processor is None:
            processor = cls.processor_map[data_type]

        if id is None:
            id = get_param(unit, data_template.id_header)
        if tiler is None:
            tiler = cls.default_tiler

        itm = processor(unit=unit, id=id, remove_url=remove_url, doc_url=doc_url,
                        uniform_product_name=uniform_product_name, data_type=data_type, title_handling=title_handling,
                        data_template=data_template)

        if docid_filter and id not in docid_filter: # or 'title' not in itm or (not itm['title']):
            return []
        else:
            if doc_based:
                return tiler.create_tiles(id_=id,
                                          text=itm['text'],
                                          title=itm['title'],
                                          max_doc_size=max_doc_size,
                                          stride=stride,
                                          remove_url=remove_url,
                                          template=itm,
                                          title_handling=title_handling)
            else:
                for pi, passage in enumerate(unit[data_template.passage_header]):
                    tpassages = []
                    passage_id = get_param(passage, data_template.passage_id_header, str(pi))
                    try:
                        tpassages.extend(
                            tiler.create_tiles(
                                id_=f"{id}-{passage_id}",
                                text=passage,
                                title=itm['title'],
                                max_doc_size=max_doc_size,
                                stride=stride,
                                remove_url=remove_url,
                                template=itm,
                                title_handling=title_handling)
                        )
                    except Exception as e:
                        print(f"Error while processing passage {id}-{passage_id}: {e}")
                    return tpassages
                return None

    @staticmethod
    def remove_stopwords(txt, **kwargs):
        return txt

    @classmethod
    def read_filter(cls, val):
        if isinstance(val, str):
            if not os.path.exists(val):
                raise RuntimeError(f"Filter file {val} does not exist.")
            keys = {}
            with open(val) as inp:
                for line in inp:
                    keys[line.strip()] = 1
            return keys
        return val

    @classmethod
    def read_data(cls,
                  input_files,
                  lang="en",
                  remove_url=False,
                  tokenizer=None,
                  tiler: TextTiler | str = None,
                  max_doc_length: int | None = None,
                  stride: int | None = None,
                  no_cache: bool = False,
                  cache_dir: str = default_cache_dir,
                  title_handling: str = 'all',
                  data_template: DataTemplate = default_data_template,
                  verbose=False,
                  **kwargs):

        use_cache = not no_cache
        doc_based = kwargs.get('doc_based', True)
        docid_map = kwargs.get('docid_map', {})
        aligned_on_sentences = get_param(kwargs, 'aligned_on_sentences', True)
        num_threads = kwargs.get('num_preprocessor_threads', 1)
        max_num_documents = kwargs.get('max_num_documents')
        if max_num_documents is None:
            max_num_documents = 100000000
        else:
            max_num_documents = int(max_num_documents)
        data_type = kwargs.get('data_type', 'auto')
        if data_type in ['sap']:
            data_template = sap_data_template
        elif data_type == "beir":
            data_template = beir_data_template

        docid_filter   = cls.read_filter(kwargs.get('docid_filter', {}))
        exclude_docids = cls.read_filter(kwargs.get('exclude_docids', {}))
        uniform_product_name = kwargs.get('uniform_product_name')
        unmapped_ids = []
        return_unmapped_ids = kwargs.get('return_unmapped', None)

        from docuverse.utils.timer import timer
        tm = timer("Data Loading")

        # ── Normalise input_files into a flat list ──────────────────────────
        pre_loaded_data = None              # set if caller passes dicts directly
        if isinstance(input_files, list):
            if isinstance(input_files[0], dict):
                pre_loaded_data = input_files
                use_cache = False
                files = []
            else:
                files = input_files
        elif isinstance(input_files, str):
            files = [input_files]
        else:
            raise RuntimeError(f"Unsupported type for {input_files}")

        # Remember the raw spec before expansion (for cache key)
        raw_spec = files[0] if len(files) == 1 and isinstance(files[0], str) else None

        # Expand globs / braces  (e.g. "data*.jsonl", "data{1,2,3}.jsonl.bz2")
        files = cls._expand_file_globs(files)

        # Strip optional "product:" prefix from each entry
        productId = uniform_product_name
        resolved_files = []
        for f in files:
            if isinstance(f, str) and ":" in f and not f.startswith("/") and not os.path.exists(f):
                productId, f = f.split(":", 1)
            resolved_files.append(f)
        files = resolved_files

        if len(files) > 1:
            print(f"Input files ({len(files)}): {files[:5]}"
                  f"{'...' if len(files) > 5 else ''}")

        # ── Step 1: Build a single cache filename ───────────────────────────
        # For globs / multi-file inputs the cache key is derived from the
        # sanitised pattern so one file encompasses all documents.
        cache_filename = None
        cache_key = None
        if use_cache and files:
            if raw_spec and cls._is_glob_pattern(raw_spec):
                cache_key = cls._sanitize_glob_for_cache(raw_spec)
            elif len(files) == 1:
                cache_key = files[0]
            else:
                # Explicit list – join sorted basenames into a single key
                cache_key = "+".join(sorted(os.path.basename(f) for f in files))
            cache_filename = cls.get_cached_filename(
                cache_key, max_doc_size=max_doc_length, stride=stride,
                aligned=aligned_on_sentences, title_handling=title_handling,
                tiler=tiler)

        # ── Step 2: Check cache → return early if hit ───────────────────────
        if cache_filename:
            check_file = raw_spec or cache_key
            cached_passages = cls.read_cache_file_if_needed(cache_filename, check_file)
            if cached_passages:
                passages = (cached_passages[:max_num_documents]
                            if 'max_num_documents' in kwargs else cached_passages)
                num_docs = len(passages)
                if docid_filter:
                    passages = [d for d in passages if get_orig_docid(d['id']) in docid_filter]
                elif exclude_docids:
                    passages = [d for d in passages if get_orig_docid(d['id']) not in exclude_docids]
                print(f"Loaded {len(passages)} passages from cache "
                      f"(filtered {num_docs - len(passages)}).")
                if verbose:
                    cls.compute_statistics(passages, tiler=tiler)
                if return_unmapped_ids:
                    return passages, unmapped_ids
                return passages

        tm.add_timing("cache_check")

        # ── Step 3: Read all raw documents (in parallel when multiple files) ─
        if pre_loaded_data is not None:
            all_data = pre_loaded_data
        elif len(files) == 1:
            if verbose:
                print(f"Reading {files[0]}", end='')
            all_data = cls._read_data(files[0], max_num_documents)
            if verbose:
                read_time = tm.mark_and_return_time()
                print(f" done: {read_time}")
        else:
            # Multiple files – read in parallel with forked processes
            file_data = cls._read_files_parallel(
                files, max_num_docs=max_num_documents,
                num_threads=num_threads)
            # Concatenate in original order
            all_data = []
            for f in files:
                all_data.extend(file_data.get(f, []))
            del file_data
            if verbose:
                read_time = tm.mark_and_return_time()
                print(f"Read {len(all_data)} documents from {len(files)} files: {read_time}")

        if max_num_documents and len(all_data) > max_num_documents:
            all_data = all_data[:max_num_documents]
        print(f"Total raw documents: {len(all_data):,}")
        tm.add_timing("read_files")

        # ── Step 4: Process / tokenize all documents ────────────────────────
        process_text_func = partial(cls.process_text,
                                    tiler=tiler,
                                    max_doc_size=max_doc_length,
                                    stride=stride,
                                    remove_url=remove_url,
                                    uniform_product_name=productId,
                                    data_type=data_type,
                                    data_template=data_template,
                                    doc_based=doc_based,
                                    docid_filter=docid_filter)

        if num_threads <= 0:
            passages = []
            for doc in tqdm(all_data, desc="Processing docs"):
                try:
                    items = process_text_func(unit=doc)
                except Exception as e:
                    items = []
                if items:
                    if verbose:
                        for item in items:
                            passages.append({**item, 'tlen': tiler.get_tokenized_length(item['text'],
                                                                                         forced_tok=True)})
                    else:
                        passages.extend(items)
        else:
            if verbose:
                post_func = partial(_compute_tokenized_length, tiler=tiler)
                post_label = "tlen"
            else:
                post_func = None
                post_label = None
            results = parallel_process(process_func=process_text_func,
                                       post_func=post_func,
                                       post_label=post_label,
                                       data=all_data, num_threads=num_threads,
                                       msg="Tokenizing")
            # Guard against None entries from crashed worker processes
            passages = list(chain.from_iterable(r for r in results if r is not None))
            del results

        del all_data
        tm.add_timing("tokenize")

        if verbose:
            print(f"Processed {len(passages):,} passages")

        # ── Save cache ──────────────────────────────────────────────────────
        if cache_filename:
            cls.write_cache_file(cache_filename, passages, use_cache)

        if verbose:
            cls.compute_statistics(passages, tiler=tiler)
        if return_unmapped_ids:
            return passages, unmapped_ids
        else:
            return passages

    @classmethod
    def _read_csv_query_file(cls, data_template, data_type, in_file):
        import pandas as pd
        data = pd.read_csv(in_file)
        passages = []
        # docid_map = kwargs.get('docid_map', {})
        for i in range(len(data)):
            itm = cls.processor_map[data_type](unit=data, data_template=data_template)
            psgs = []
            ids = []
            for val, loio in [[f'passage {k}', f'loio {k}'] for k in range(1, 4)]:
                if type(data[val][i]) == str:
                    psgs.append(data[val][i])
                    loio = str(data[loio][i]).replace('\t', '')
                    if loio == 'nan':
                        loio = ""
                    if type(loio) is not str or (loio != "" and loio.find("loio") == -1):
                        print(f"Error: the loio {loio} does not have the word 'loio' in it.")
                        continue
                    else:
                        loio_v = loio.replace('loio', '')
                    if loio == "":
                        continue
                    # if loio_v in docid_map:
                    #     if docid_map[loio_v] not in ids:
                    #         ids.append(docid_map[loio_v])
                    # else:
                    #     ids.append(loio_v)
                    #     unmapped_ids.append(loio_v)
            itm['passages'] = psgs
            itm['relevant'] = ids

        return passages

    @staticmethod
    def _is_glob_pattern(s: str) -> bool:
        """Return True if the string contains glob or brace-expansion characters."""
        return any(c in s for c in ('*', '?', '[', '{'))

    @staticmethod
    def _sanitize_glob_for_cache(pattern: str) -> str:
        """Replace glob/brace characters with readable tokens for use as a cache key.

        Examples::

            "data/corpus_*.jsonl.bz2"    -> "data/corpus_STAR.jsonl.bz2"
            "data/shard_{1,2,3}.jsonl"   -> "data/shard_BR1-2-3BR.jsonl"
            "data/part_??.jsonl"         -> "data/part_QMQM.jsonl"
        """
        import re as _re

        # Braces: {a,b,c} -> BRa-b-cBR
        def _replace_braces(m):
            inner = m.group(1).replace(',', '-')
            return f"BR{inner}BR"
        result = _re.sub(r'\{([^}]+)\}', _replace_braces, pattern)

        result = result.replace('*', 'STAR')
        result = result.replace('?', 'QM')
        result = result.replace('[', 'SB').replace(']', 'SB')
        return result

    @classmethod
    def _expand_file_globs(cls, files: list) -> list:
        """Expand glob and brace patterns in a list of file paths.

        Strings containing glob characters (``*``, ``?``, ``[``) or brace
        patterns (``{a,b}``) are expanded.  Non-glob strings and non-string
        entries (e.g. pre-loaded dicts) are passed through unchanged.

        Returns:
            Expanded list of file paths / objects, sorted per-glob.
        """
        expanded = []
        for f in files:
            if not isinstance(f, str):
                expanded.append(f)
                continue

            # Strip the optional "product:" prefix before glob-testing
            prefix = ""
            raw = f
            if ":" in f and not f.startswith("/"):
                # Handle "productId:path/pattern" — only split on the first colon
                # but avoid splitting absolute paths like /foo/bar
                parts = f.split(":", 1)
                if not os.path.exists(f):
                    prefix = parts[0] + ":"
                    raw = parts[1]

            has_glob = any(c in raw for c in ('*', '?', '['))

            # Expand brace patterns {a,b,c} — Python's glob doesn't handle them
            if '{' in raw and ',' in raw:
                brace_expanded = cls._expand_braces(raw)
            else:
                brace_expanded = [raw]

            if has_glob or len(brace_expanded) > 1:
                matched = []
                for pattern in brace_expanded:
                    hits = sorted(glob_module.glob(pattern))
                    if hits:
                        matched.extend(hits)
                    else:
                        # No match — keep the original (will fail later with a clear error)
                        matched.append(pattern)
                expanded.extend(prefix + m for m in matched)
            else:
                expanded.append(f)

        return expanded

    @staticmethod
    def _expand_braces(pattern: str) -> List[str]:
        """Expand a single brace pattern like ``data{1,2,3}.jsonl`` into a list.

        Handles one level of braces. Nested braces are not supported.
        """
        import re as _re
        m = _re.search(r'\{([^}]+)\}', pattern)
        if not m:
            return [pattern]
        prefix = pattern[:m.start()]
        suffix = pattern[m.end():]
        alternatives = m.group(1).split(',')
        return [prefix + alt.strip() + suffix for alt in alternatives]

    @classmethod
    def _read_files_parallel(cls, file_paths: List[str], max_num_docs: int = -1,
                             num_threads: int = 4) -> Dict[str, list]:
        """Read multiple data files in parallel using forked processes.

        Uses ``ProcessPoolExecutor`` with fork context so that child
        processes inherit the parent's already-imported modules (fast
        startup) and bypass the GIL for CPU-bound decompression.

        Args:
            file_paths: list of file paths to read
            max_num_docs: per-file document limit
            num_threads: number of reader processes

        Returns:
            Dict mapping file_path -> list of records
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_workers = min(len(file_paths), max(1, num_threads))
        print(f"Reading {len(file_paths)} files in parallel "
              f"({n_workers} forked processes)...")

        try:
            ctx = mp.get_context('fork')
        except ValueError:
            ctx = mp.get_context('spawn')

        results = {}
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(cls._read_data, p, max_num_docs): p
                       for p in file_paths}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Reading files"):
                path = futures[future]
                try:
                    data = future.result()
                    results[path] = data
                except Exception as e:
                    print(f"  ERROR reading {path}: {e}")
                    results[path] = []

        return results

    @staticmethod
    def _read_data(filename, max_num_docs=-1) -> List[Dict[str, str]]:
        data = None

        try:
            with open_stream(filename) as in_file:
                if file_is_of_type(filename, extensions=".tsv"):
                    # csv_reader = csv.DictReader(in_file, delimiter="\t")
                    # data = [doc for doc in at_most(csv_reader, max_num_docs)]
                    # # Try to address some list formatting within tsv
                    # for doc in data:
                    #     for key, val in doc.items():
                    #         if val.find("[") >= 0:
                    #             doc[key] = val.replace("[", "").replace("]", "").replace(",","").replace("'","").split(" ")
                    def convert_to_list(val):
                        try:
                            val = [str(s) for s in ast.literal_eval(val)]
                        except:
                            pass
                        return val
                    import pandas as pd
                    import ast
                    indata = pd.read_csv(filename, sep="\t")
                    for key in indata.keys():
                            # Check if column has any string values containing '[' and convert it to a list.
                            # For some reason, that's how arrays are represented in the csv column.
                            has_bracket_strings = False
                            for val in indata[key].dropna():
                                if isinstance(val, str) and '[' in val:
                                    has_bracket_strings = True
                                    break
                            
                            if has_bracket_strings:
                                indata[key] = indata[key].map(convert_to_list)
                    data = indata.to_dict(orient="records")
                elif file_is_of_type(filename, extensions=".csv"):
                    csv_reader = csv.DictReader(in_file, delimiter=",")
                    data = [doc for doc in at_most(csv_reader, max_num_docs)]
                elif file_is_of_type(filename, ['.json', '.jsonl']):
                    if filename.find('.jsonl') >= 0:
                        data = [orjson.loads(line) for line in tqdm(at_most(in_file, max_num_docs),
                                                                  desc=f"Reading {filename}:")]
                    else:
                        data = orjson.loads("".join(in_file.readlines()))
                else:
                    raise RuntimeError(f"Unknown file extension: {os.path.splitext(filename)[1]}")
        except Exception as e:
            raise RuntimeError(f"Error reading {filename}: {e}")
        if max_num_docs > 0:
            data = data[:max_num_docs]
        return data

    @classmethod
    def _read_json_file(cls, process_text, in_file, filename, data_template,
                        docid_filter, doc_based, max_num_documents):
        tpassages = []
        if filename.find('.jsonl') >= 0:
            data = [orjson.loads(line) for line in in_file]
        else:
            data = orjson.loads("".join(in_file.readlines()))

        read_docs = 0
        for di, doc in tqdm(enumerate(data),
                            total=min(max_num_documents, len(data)),
                            desc="Reading json documents",
                            smoothing=0.05):
            if di >= max_num_documents:
                break
            docid = str(doc[data_template.id_header])

            if ".txt" in docid:
                docid = docid.replace(".txt", "")

            if docid_filter != [] and docid not in docid_filter:
                continue
            try:
                if doc_based:
                    tpassages.extend(process_text(unit=doc))
                else:
                    for pi, passage in enumerate(doc[data_template.passage_header]):
                        passage_id = get_param(passage, data_template.passage_id_header, str(pi))
                        tpassages.extend(
                            process_text(id=f"{doc[data_template.id_header]}-{passage_id}",
                                         unit=passage))
            except Exception as e:
                print(f"Error at line {di}: {e} - skipping document {doc[data_template.id_header]}")
                # raise e
            read_docs += 1

        return tpassages, read_docs

    @classmethod
    def _read_csv_file(cls, process_text, in_file, max_num_documents):
        tpassages = []
        csv_reader = csv.DictReader(in_file, delimiter="\t")
        next(csv_reader)
        docs_read = 0
        for ri, row in tqdm(enumerate(csv_reader)):
            if docs_read >= max_num_documents:
                break
            assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
            tpassages.extend(
                process_text(unit=row))
            docs_read += 1
        return tpassages, docs_read

    @classmethod
    def compute_statistics(cls, corpus, tiler=None):
        min_idx, max_idx, avg_idx, total_idx = range(0, 4)
        token_based = [1000, 0, 0, 0]
        char_based = [1000, 0, 0, 0]  
        char_vals = []
        token_vals = []
        tiles = 0
        doc_ids = {}

        def update_stat(input, vector):
            vector[min_idx] = min(vector[min_idx], input)
            vector[max_idx] = max(vector[max_idx], input)
            vector[avg_idx] += input
            vector[total_idx] += 1

        def compute_stats(vector):
            return [
                vector[min_idx],
                vector[max_idx],
                1.0 * vector[avg_idx] / max(char_based[total_idx], 1)
            ]

        tq = tqdm(desc="Computing statistics", total=len(corpus))
        for i, entry in enumerate(corpus):
            tq.update(1)
            txt = get_param(entry, 'text')
            char_len = len(txt)
            char_vals.append(char_len)
            update_stat(char_len, char_based)
            if 'tlen' in entry:
                token_length = entry['tlen']
            else:
                token_length = tiler.get_tokenized_length(text=txt, forced_tok=True)
            update_stat(token_length, token_based)
            token_vals.append(token_length)
            tiles += 1
            doc_ids[get_orig_docid(get_param(entry, 'id'))] = 1
        tq.close()

        stats = {
            'num_docs': len(doc_ids),
            'num_tiles': tiles,
            'char': compute_stats(char_based),
            'token': compute_stats(token_based)
        }

        second_len = 20
        print("=" * 60)
        print('Statistics:')
        print("=" * 60)
        print(f"{'Number of documents:':20s}{stats['num_docs']:<10d}")
        print(f"{'Number of tiles:':20s}{stats['num_tiles']:<10d}")
        print(f"{'#tiles per document:':20s}{stats['num_tiles'] / stats['num_docs']:<10.2f}")
        print(f"{'':20s}{'Character-based:':<{second_len}s}{'Token-based:':<{second_len}s}")
        print(
            f"{'  Minimum length:':20s}{stats['char'][min_idx]:<{second_len}d}{stats['token'][min_idx]:<{second_len}d}")
        print(
            f"{'  Maximum length:':20s}{stats['char'][max_idx]:<{second_len}d}{stats['token'][max_idx]:<{second_len}d}")
        print(
            f"{'  Average length:':20s}{stats['char'][avg_idx]:<{second_len}.1f}{stats['token'][avg_idx]:<{second_len}.1f}")
        print("=" * 60)
        from docuverse.utils.text_histogram import histogram
        print("Char histogram:\n")
        histogram(char_vals)
        print("Token histogram:\n")
        if tiler and tiler.tokenizer:
            histogram(token_vals)
        else:
            print("No token information available.")
