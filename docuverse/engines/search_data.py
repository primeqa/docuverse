import itertools
import os
import json
import csv
import re
import time
from functools import partial
from multiprocessing import Manager, Queue, Process
from typing import Dict, List
from tqdm import tqdm
import queue
from docuverse.utils import at_most

from docuverse.engines import data_template
from docuverse.engines.data_template import (
    DataTemplate,
    default_data_template,
    sap_data_template,
    beir_data_template
)
from docuverse.utils.text_tiler import TextTiler
from docuverse.utils import get_param


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
            stopwords_list = json.load(open(stopword_file))
            stopwords = {}
            for lang, vals in stopwords_list.items():
                stopwords[lang] = re.compile(f"\\b({'|'.join(vals)})\\b", re.IGNORECASE)

    def _init(self, **kwargs):
        pass

    def __call__(self, unit, id, data_template, **kwargs):
        itm = {
            'id': id,
            'title': self.cleanup(get_param(unit, data_template.title_header)),
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
        tok_dir_name = os.path.basename(tiler.tokenizer.name_or_path) if tiler is not None else "none"
        if tok_dir_name == "":
            tok_dir_name = os.path.basename(os.path.dirname(tiler.tokenizer.name_or_path))
        cache_file_name = os.path.join(cache_dir, "_".join([f"{input_file.replace('/', '__')}",
                                                            f"{max_doc_size}",
                                                            f"{stride}",
                                                            f"{aligned}" if aligned else "unaligned",
                                                            f"{title_handling}",
                                                            f"{tok_dir_name}"]) + ".jsonl.bz2")
        print(f"Cache filename is {cache_file_name}")
        return cache_file_name

    @staticmethod
    def _open_file(file_name: str, write: bool = False):
        if write:
            mode = "w"
            cache_dir = os.path.dirname(file_name)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
        else:
            mode = "r"
            if not os.path.exists(file_name):
                raise RuntimeError(f"File {file_name} does not exist!")
                # return None
        input_stream = None
        if file_name.endswith(".bz2"):
            import bz2
            input_stream = bz2.open(file_name, mode)
        elif file_name.endswith(".gz"):
            import gzip
            input_stream = gzip.open(file_name, mode)
        else:  # if file_name.endswith(".jsonl"):
            input_stream = open(file_name, mode)
        return input_stream

    @staticmethod
    def read_cache_file_if_needed(cache_file_name, input_file):
        passages = []

        if os.path.exists(cache_file_name) and os.path.getmtime(cache_file_name) > os.path.getmtime(input_file):
            input_stream = SearchData._open_file(cache_file_name, write=False)
            for line in tqdm(input_stream, desc="Reading cache file:"):
                passages.append(json.loads(line.decode('utf-8')))

            input_stream.close()

        return passages

    @staticmethod
    def write_cache_file(cache_filename, passages, use_cache=True):
        if not use_cache:
            return
        output_stream = SearchData._open_file(cache_filename, write=True)
        for p in passages:
            output_stream.write(f"{json.dumps(p)}\n".encode("utf-8"))
        output_stream.close()

    @classmethod
    def process_text(cls,
                     tiler,
                     unit,
                     max_doc_size,
                     stride,
                     id=None,
                     remove_url=True,
                     doc_url=None,
                     uniform_product_name=None,
                     data_type="sap",
                     title_handling="all",
                     processor=None,
                     data_template=default_data_template,
                     docid_filter={},
                     doc_based=True
                     ):
        """
        Convert a given document or passage (from 'output.json') to a dictionary, splitting the text as necessary.
        :param tiler: Tiler instance - the Tiler object that creates the tiles from the input text
        :param unit: the paragraph/document structure to proces
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

        itm = processor(unit=unit, id=id, remove_url=remove_url, doc_url=doc_url,
                        uniform_product_name=uniform_product_name, data_type=data_type, title_handling=title_handling,
                        data_template=data_template)

        if docid_filter and id not in docid_filter or 'title' not in itm or (not itm['title']):
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
                    return tpassages

    @staticmethod
    def remove_stopwords(txt, **kwargs):
        return txt

    @staticmethod
    def get_orig_docid(id):
        index = id.rfind("-", 0, id.rfind("-"))
        if index >= 0:
            return id[:index]
        else:
            return id

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
        url = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
        data_type = kwargs.get('data_type', 'auto')
        if isinstance(input_files, list):
            files = input_files
        elif isinstance(input_files, str):
            files = [input_files]
        else:
            raise RuntimeError(f"Unsupported type for {input_files}")
        docname2url = kwargs.get('docname2url', None)
        docs_read = 0
        remv_stopwords = bool(kwargs.get('remove_stopwords', False))
        unmapped_ids = []
        return_unmapped_ids = kwargs.get('return_unmapped', None)
        if data_type in ['sap']:
            data_template = sap_data_template
        elif data_type == "beir":
            data_template = beir_data_template

        docid_filter = kwargs.get('docid_filter', [])
        uniform_product_name = kwargs.get('uniform_product_name')
        from docuverse.utils.timer import timer
        tm = timer("Data Loading")
        passages = []

        for input_file in files:
            docs_read = 0
            if input_file.find(":") >= 0:
                productId, input_file = input_file.split(":")
            else:
                productId = uniform_product_name

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

            if use_cache:
                cache_filename = cls.get_cached_filename(input_file,
                                                         max_doc_size=max_doc_length, stride=stride,
                                                         aligned=aligned_on_sentences,
                                                         title_handling=title_handling, tiler=tiler)
                cached_passages = cls.read_cache_file_if_needed(
                    cache_filename,
                    input_file)
                if cached_passages:
                    passages.extend(cached_passages[:max_num_documents]
                                    if 'max_num_documents' in kwargs
                                    else cached_passages)
                    continue
            if verbose:
                print(f"Reading {input_file}", end='')
            tpassages = []
            data = cls._read_data(input_file, max_num_documents)

            if verbose:
                read_time = tm.mark_and_return_time()
                print(f" done: {read_time}")

            if num_threads == 1:
                for doc in tqdm(data, desc="Reading docs:"):
                    try:
                        items = process_text_func(unit=doc)
                    except Exception as e:
                        items = []
                    if items:
                        if verbose:
                            for item in items:
                                tpassages.append({**item, 'tlen': tiler.get_tokenized_length(item['text'],
                                                                                             forced_tok=True)})
                        else:
                            tpassages.extend(items)
            else:
                doc_queue = Queue()
                manager = Manager()
                d = manager.dict()
                import multiprocessing as mp

                def processor(inqueue, d, tiler):
                    pid = mp.current_process().pid
                    while True:
                        try:
                            id, text = inqueue.get(block=True, timeout=1)
                        except queue.Empty:
                            break
                        except Exception as e:
                            break

                        try:
                            items = process_text_func(unit=text)
                            d[id] = [{**item,
                                      'tlen': tiler.get_tokenized_length(item['text'], forced_tok=True)}
                                     for item in items]
                        except Exception as e:
                            d[id] = []

                for i, doc in enumerate(data):
                    doc_queue.put([i, doc])
                processes = []
                for i in range(num_threads):
                    p = Process(target=processor, args=(doc_queue, d, tiler))
                    processes.append(p)
                    p.start()
                tk = tqdm(desc="Reading docs:", total=doc_queue.qsize())
                c = doc_queue.qsize()
                while c>0:
                    c1 = doc_queue.qsize()
                    if c != c1:
                        tk.update(c - c1)
                        c = c1
                    time.sleep(0.1)
                print(f"Dropped out of the while loop: {doc_queue.qsize()}")
                tk.clear()
                for i, p in enumerate(processes):
                    p.join()
                for i in range(len(data)):
                    tpassages.extend(d[i])

            if verbose:
                read_time = tm.mark_and_return_time()
                print(f"Processed in {read_time}")
            if use_cache:
                cls.write_cache_file(cache_filename, tpassages, use_cache)
            passages.extend(tpassages)
            max_num_documents -= docs_read

        if verbose:
            cls.compute_statistics(passages, tiler=tiler)
            # read_time = tm.mark_and_return_time()
            # print(f"Statistics computed in {read_time}")
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

    @classmethod
    def _read_data(self, filename, max_num_docs=-1) -> List[Dict[str, str]]:
        data = None

        try:
            with SearchData._open_file(filename) as in_file:
                if SearchData.is_of_type(filename, extensions=[".tsv"]):
                    csv_reader = csv.DictReader(in_file, delimiter="\t")
                    data = [doc for doc in at_most(csv_reader, max_num_docs)]
                if SearchData.is_of_type(filename, extensions=[".csv"]):
                    csv_reader = csv.DictReader(in_file, delimiter=",")
                    data = [doc for doc in at_most(csv_reader, max_num_docs)]
                elif SearchData.is_of_type(filename, ['.json', '.jsonl']):
                    if filename.find('.jsonl') >= 0:
                        data = [json.loads(line) for line in tqdm(at_most(in_file, max_num_docs),
                                                                  desc=f"Reading {filename}:")]
                    else:
                        data = json.load(in_file)
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
            data = [json.loads(line) for line in in_file]
        else:
            data = json.load(in_file)

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
    def is_of_type(cls, input_file, extensions):
        """
        Check if the given file is of any of the specified file types, cross-product with compressed extensions.

        Parameters:
            input_file (str): The file path or name to be checked.
            extensions (list): A list of file extensions to check against.

        Returns:
            bool: True if the file is of any of the specified types, False otherwise.
        """
        return any(input_file.endswith(f"{ext[0]}{ext[1]}")
                   for ext in itertools.product(extensions, ['', ".bz2", ".gz", ".xz"]))

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
                1.0 * vector[avg_idx] / char_based[total_idx]
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
            doc_ids[SearchData.get_orig_docid(get_param(entry, 'id'))] = 1
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
        histogram(token_vals)
