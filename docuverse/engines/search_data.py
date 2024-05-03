import os
import json
import csv
import re
from typing import Dict, List
from tqdm import tqdm
from docuverse.utils.text_tiler import TextTiler


class DefaultProcessor:
    def __init__(self, title_name:str= "title", stopwords:List[str]=[], lang:str="en"):
        self.title = title_name
        self.stopwords = stopwords
        self.lang = lang

    def _init(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        # itm = {self.title: item[self.title]} if self.title in item else {self.title: ""}
        itm = kwargs
        if 'title' not in itm:
            itm['title'] = ""
        return itm

    def remove_stopwords(self, text:str, lang:str, do_replace: bool=False) -> str:
        global stopwords, settings
        if not do_replace:
            return text
        else:
            if stopwords is None:
                stopwords = re.compile(
                    "\\b(?:" + "|".join(settings["analysis"]["filter"][f"{lang}_stop"]["stopwords"]) + ")\\b",
                    re.IGNORECASE)
            return re.sub(r' {2,}', ' ', re.sub(stopwords, " ", text))

class SAPProccesor(DefaultProcessor):

    def __init__(self, title_name:str= "title", hana_file2url:List[str]=None):
        super().__init__(title_name)
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

    def process_product_id(self, url_fields, uniform_product_name, data_type):
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

    def get_course_product_id(self, product_id):
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

    def process_url(self, doc_url: str, data_type: str = ""):
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
            >>> process_url(doc_url, data_type)
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

    def fix_title(title):
        return re.sub(r' {2,}', ' ', title.replace(" | SAP Help Portal", ""))

    def __call__(self, **kwargs):
        global product_counts
        doc_url = kwargs.get("doc_url")
        data_type = kwargs.get("data_type", "")

        full_url, fields = self.process_url(doc_url, data_type)
        uniform_product_name = kwargs.get("uniform_product_name", "")
        product_id = self.process_product_id(fields, uniform_product_name, data_type)
        title = kwargs.get("title")
        docid = kwargs.get('docid', None)
        if docid is not None:
            if docid.endswith(".txt"):
                docid = docid[:-4]
            if docid in self.docname2url_title:
                url, title = self.docname2url_title[docid]

        if product_id not in product_counts:
            product_counts[product_id] = 1
        else:
            product_counts[product_id] += 1

        itm = {
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
        'sap': SAPProccesor(title_name='title')
    }

    class Entry:
        def __init__(self, dict: Dict[str, str]):
            self.__dict__.update(dict)

    def __init__(self, filenames, text_field_name: str="text", title_field_name: str="title", **data):
        self.entries = []
        self.tiler = None
        # self.dict = data
        # self.__dict__.update(data)

    def get_text(self, i:int) -> str:
        return ""

    def __getitem__(self, i: int):
        return self.entries[i]

    def get_cached_filename(input_file: str,
                            max_doc_size: int,
                            stride: int,
                            tiler: TextTiler,
                            title_handling="all",
                            cache_dir: str = default_cache_dir):
        tok_dir_name = os.path.basename(tiler.tokenizer.name_or_path)
        if tok_dir_name == "":
            tok_dir_name = os.path.basename(os.path.dirname(tiler.tokenizer.name_or_path))
        cache_file_name = os.path.join(cache_dir, "_".join([f"{input_file.replace('/', '__')}",
                                                            f"{max_doc_size}",
                                                            f"{stride}",
                                                            f"{title_handling}",
                                                            f"{tok_dir_name}"]) + ".jsonl.bz2")
        print(f"Cache filename is {cache_file_name}")
        return cache_file_name

    @staticmethod
    def open_cache_file(cache_file_name: str, write: bool = False):
        if write:
            mode = "w"
            cache_dir = os.path.dirname(cache_file_name)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
        else:
            mode = "r"
            if not os.path.exists(cache_file_name):
                return None
        input_stream = None
        if cache_file_name.endswith(".jsonl.bz2"):
            import bz2
            input_stream = bz2.open(cache_file_name, mode)
        elif cache_file_name.endswith(".jsonl.gz"):
            import gzip
            input_stream = gzip.open(cache_file_name, mode)
        elif cache_file_name.endswith(".jsonl"):
            input_stream = open(cache_file_name, mode)
        else:
            print(f"Unknown file extension for file: {cache_file_name}")
            raise RuntimeError(f"Unknown file extension for file: {cache_file_name}")
        return input_stream

    @staticmethod
    def read_cache_file_if_needed(cache_file_name, input_file):
        passages = []

        if os.path.exists(cache_file_name) and os.path.getmtime(cache_file_name) > os.path.getmtime(input_file):
            input_stream = SearchData.open_cache_file(cache_file_name, write=False)
            for line in input_stream:
                passages.append(json.loads(line.decode('utf-8')))

            input_stream.close()

        return passages

    @staticmethod
    def write_cache_file(cache_filename, passages, use_cache=True):
        if not use_cache:
            return
        output_stream = SearchData.open_cache_file(cache_filename, write=True)
        for p in passages:
            output_stream.write(f"{json.dumps(p)}\n".encode("utf-8"))
        output_stream.close()

    def process_text(self,
                     tiler,
                     id,
                     title,
                     text,
                     max_doc_size,
                     stride,
                     remove_url=True,
                     tokenizer=None,
                     doc_url=None,
                     uniform_product_name=None,
                     data_type="sap",
                     title_handling="all",
                     processor=None,
                     ):
        """
        Convert a given document or passage (from 'output.json') to a dictionary, splitting the text as necessary.
        :param id: str - the prefix of the id of the resulting piece/pieces
        :param title: str - the title of the new piece
        :param text: the input text to be split
        :param max_doc_size: int - the maximum size (in word pieces) of the resulting sub-document/sub-passage texts
        :param stride: int - the stride/overlap for consecutive pieces
        :param remove_url: Boolean - if true, URL in the input text will be replaced with "URL"
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
        :return - a list of indexable items, each containing a title, id, text, and url.
        """

        if processor is None:
            processor = self.processor_map[data_type]

        itm = processor(id=id, title=title, text=text, remove_url=remove_url, doc_url=doc_url,
                        uniform_product_name=uniform_product_name, data_type=data_type, title_handling=title_handling)

        return tiler.create_tiles(id_=id,
                                  text=text,
                                  title=title,
                                  max_doc_size=max_doc_size,
                                  stride=stride,
                                  remove_url=remove_url,
                                  template=itm,
                                  title_handling=title_handling)

    @staticmethod
    def read_data(self, input_files,
                  lang,
                  fields=None,
                  remove_url=False,
                  tokenizer=None,
                  tiler=None,
                  max_doc_size=None,
                  stride=None,
                  use_cache=True,
                  cache_dir=default_cache_dir,
                  title_handling='all',
                  **kwargs):
        def remove_stopwords(text: str, lang, do_replace: bool = False) -> str:
            global stopwords, settings
            if not do_replace:
                return text
            else:
                if stopwords is None:
                    stopwords = re.compile(
                        "\\b(?:" + "|".join(settings["analysis"]["filter"][f"{lang}_stop"]["stopwords"]) + ")\\b",
                        re.IGNORECASE)
                return re.sub(r' {2,}', ' ', re.sub(stopwords, " ", text))


        passages = []
        doc_based = kwargs.get('doc_based', True)
        docid_map = kwargs.get('docid_map', {})
        max_num_documents = kwargs.get('max_num_documents', 1000000000)
        url = r'https?://(?:www\.)?(?:[-a-zA-Z0-9@:%._\+~#=]{1,256})\.(:?[a-zA-Z0-9()]{1,6})(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)*\b'
        data_type = kwargs.get('data_type', 'auto')
        if fields is None:
            num_args = 3
        else:
            num_args = len(fields)
        if isinstance(input_files, list):
            files = input_files
        elif isinstance(input_files, str):
            files = [input_files]
        else:
            raise RuntimeError(f"Unsupported type for {input_files}")
        docname2url = kwargs.get('docname2url', None)
        docs_read = 0
        remv_stopwords = kwargs.get('remove_stopwords', False)
        unmapped_ids = []
        return_unmapped_ids = kwargs.get('return_unmapped', None)

        for input_file in files:
            docs_read = 0
            if input_file.find(":") >= 0:
                productId, input_file = input_file.split(":")
            else:
                productId = None
            cached_passages = SearchData.read_cache_file_if_needed(
                SearchData.get_cached_filename(input_file,
                                               max_doc_size=max_doc_size,
                                               stride=stride,
                                               title_handling=title_handling,
                                               tiler=tiler),
                input_file)
            if cached_passages:
                passages.extend(cached_passages)
                continue
            print(f"Reading {input_file}")
            tpassages = []
            with open(input_file) as in_file:
                if input_file.endswith(".tsv"):
                    # We'll assume this is the PrimeQA standard format
                    csv_reader = \
                        csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                            if fields is not None \
                            else csv.DictReader(in_file, delimiter="\t")
                    next(csv_reader)
                    for ri, row in tqdm(enumerate(csv_reader)):
                        if docs_read >= max_num_documents:
                            break
                        assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
                        if 'answers' in row:
                            if remove_url:
                                row['text'] = remove_stopwords(re.sub(url, lang, 'URL', row['text']), remv_stopwords)
                            itm = {'text': (row["title"] + ' ' if 'title' in row else '') + row["text"],
                                   'id': row['id']}
                            if 'title' in row:
                                itm['title'] = remove_stopwords(row['title'], lang, remv_stopwords)
                            if 'relevant' in row:
                                itm['relevant'] = row['relevant'].split(",")
                            if 'answers' in row:
                                itm['answers'] = row['answers'].split("::")
                                itm['passages'] = itm['answers']
                            tpassages.append(itm)
                        else:
                            tpassages.extend(
                                self.process_text(tiler=tiler,
                                                 id=row['id'],
                                                 title=remove_stopwords(row['title'], lang,
                                                                        remv_stopwords) if 'title' in row else '',
                                                 text=remove_stopwords(row['text'], lang, remv_stopwords),
                                                 max_doc_size=max_doc_size,
                                                 stride=stride,
                                                 remove_url=remove_url,
                                                 tokenizer=tokenizer,
                                                 doc_url=url,
                                                 uniform_product_name=None,
                                                 data_type=data_type,
                                                 title_handling=title_handling
                                             ))
                elif input_file.endswith('.json') or input_file.endswith(".jsonl"):
                    # This should be the SAP or BEIR json format
                    if input_file.endswith('.json'):
                        data = json.load(in_file)
                    else:
                        data = [json.loads(line) for line in open(input_file).readlines()]
                    uniform_product_name = kwargs.get('uniform_product_name', default=productId)
                    docid_filter = kwargs.get('docid_filter', [])
                    # data_type = get_attr(kwargs, 'data_type', 'sap')
                    if data_type in ['auto', 'sap']:
                        txtname = "document"
                        psg_txtname = "text"
                        docidname = "document_id"
                        titlename = "title"
                        data_type = "sap"
                    elif data_type == "beir":
                        txtname = "text"
                        docidname = "_id"
                        titlename = 'title'

                    for di, doc in tqdm(enumerate(data),
                                        total=min(max_num_documents, len(data)),
                                        desc="Reading json documents",
                                        smoothing=0.05):
                        if di >= max_num_documents:
                            break
                        docid = doc[docidname]

                        if ".txt" in docid:
                            docid = docid.replace(".txt", "")

                        if docid_filter != [] and docid not in docid_filter:
                            continue
                        url = doc['document_url'] if 'document_url' in doc else \
                            doc['url'] if 'url' in doc else ""
                        title = doc[titlename] if 'title' in doc else None
                        if title is None:
                            title = ""
                        # if docname2url and docid in docname2url:
                        #     url = docname2url[docid]
                        #     title = docname2title[docid]
                        try:
                            if doc_based:
                                tpassages.extend(
                                    self.process_text(tiler=tiler,
                                                     id=doc[docidname],
                                                     title=remove_stopwords(title, lang, remv_stopwords),
                                                     text=remove_stopwords(doc[txtname], remv_stopwords),
                                                     max_doc_size=max_doc_size,
                                                     stride=stride,
                                                     remove_url=remove_url,
                                                     tokenizer=tokenizer,
                                                     doc_url=url,
                                                     uniform_product_name=uniform_product_name,
                                                     data_type=data_type
                                                 ))
                            else:
                                for pi, passage in enumerate(doc['passages']):
                                    passage_id = passage['passage_id'] if 'passage_id' in passage else pi
                                    tpassages.extend(
                                        self.process_text(tiler=tiler,
                                                         id=f"{doc[docidname]}-{passage_id}",
                                                         title=remove_stopwords(title, lang, remv_stopwords),
                                                         text=remove_stopwords(passage[psg_txtname], remv_stopwords),
                                                         max_doc_size=max_doc_size,
                                                         stride=stride,
                                                         remove_url=remove_url,
                                                         tokenizer=tokenizer,
                                                         doc_url=url,
                                                         uniform_product_name=uniform_product_name,
                                                         data_type=data_type,
                                                         title_handling=title_handling
                                                     ))
                        except Exception as e:
                            print(f"Error at line {di}: {e}")
                            raise e
                        docs_read += 1
                elif kwargs.get('read_sap_qfile', default=False) or input_file.endswith(".csv"):
                    import pandas as pd
                    data = pd.read_csv(in_file)
                    passages = []
                    docid_map = kwargs.get('docid_map', default={})
                    for i in range(len(data)):
                        itm = {}
                        itm['id'] = i
                        itm['text'] = remove_stopwords(data.Question[i].strip(), remv_stopwords)
                        itm['answers'] = data['Gold answer'][i]
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
                                if loio_v in docid_map:
                                    if docid_map[loio_v] not in ids:
                                        ids.append(docid_map[loio_v])
                                else:
                                    ids.append(loio_v)
                                    unmapped_ids.append(loio_v)
                        itm['passages'] = psgs
                        itm['relevant'] = ids
                        tpassages.append(itm)
                    self.write_cache_file(
                        self.get_cached_filename(input_file, max_doc_size, stride, tiler, title_handling=title_handling),
                        tpassages,
                        use_cache)
                    if return_unmapped_ids:
                        return tpassages, unmapped_ids
                else:
                    raise RuntimeError(f"Unknown file extension: {os.path.splitext(input_file)[1]}")
            self.write_cache_file(
                self.get_cached_filename(input_file, max_doc_size, stride, tiler, title_handling),
                tpassages,
                use_cache)
            passages.extend(tpassages)
            max_num_documents -= docs_read

        if return_unmapped_ids:
            return passages, unmapped_ids
        else:
            return passages
