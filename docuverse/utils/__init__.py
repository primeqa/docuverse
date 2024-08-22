import itertools
import re
import os
import time
from multiprocessing import Queue, Manager, Process
from typing import List, Union

# from .embedding_function import DenseEmbeddingFunction
import yaml
import json
import copy
import queue

from tqdm import tqdm


def get_param(dictionary, key: str, default: str | None | bool = None):
    def recursive_get(_dictionary, key, default):
        if _dictionary is None:
            return default
        if key.find(".") >= 0:
            keys = key.split(".")
            res = default
            if isinstance(dictionary, dict):
                dd = _dictionary
            else:
                dd = _dictionary.__dict__
            for k in keys:
                if k in dd:
                    dd = dd[k]
                else:
                    return res
            return dd
        else:
            return dictionary.get(key, default)

    if key is None:
        return default
    elif key.find("|") > 0:
        weird_value = ":+:+"
        keys = key.split("|")
        for k in keys:
            k = recursive_get(dictionary, k, weird_value)
            if k != weird_value:
                return k
        return default
    else:
        return recursive_get(dictionary, key, default)

def get_config_dir(config_path: str | None = None) -> str:
    def find_docuverse_base(path: str, basename="docuverse"):
        dir = path
        head = os.path.basename(dir)
        found = False
        while dir:
            if head == basename:
                return dir
            dir = os.path.dirname(dir)
            head = os.path.basename(dir)
        return None

    if get_param(os.environ, 'DOCUVERSE_CONFIG_PATH') is not None:
        config_dir = os.environ['DOCUVERSE_CONFIG_PATH']
    elif config_path is None or not os.path.exists(config_path):
        possible_paths = [find_docuverse_base(os.path.abspath(config_path)),
                          '.', os.path.join(os.path.dirname(__file__), "../../../..")]
        for path in possible_paths:
            config_dir = os.path.abspath(os.path.join(path, "config"))
            if os.path.exists(config_dir):
                return config_dir
        raise RuntimeError(f"Could not find config file in {possible_paths}")
    else:
        return config_path
    return config_dir




def read_config_file(config_file):
    patt = re.compile(r"{{(.*?)}}")
    if not os.path.exists(config_file):
        config_file = os.path.join(get_config_dir(os.path.dirname(config_file)), os.path.basename(config_file))
    def replace(local_dict:dict, global_dict:dict, parent_key: str=""):
        not_done = False
        for key, val in local_dict.items():
            if isinstance(val, str):
                m = re.search(patt, val)
                while m:
                    skey = f"{parent_key}.{m.group(1)}" if parent_key != "" else m.group(1)
                    rr = get_param(global_dict, skey)
                    start="{{"
                    end="}}"
                    if not isinstance(rr, str) or rr.find("{{") < 0:
                        val = val.replace(f"{start}{m.group(1)}{end}", str(rr))
                    else:
                        not_done = True
                    m=re.search(patt, val)
                local_dict[key] = val
            elif isinstance(val, dict):
                # Call recursively
                not_done |= replace(local_dict=val, global_dict=global_dict,
                                    parent_key=f"{parent_key}.{key}" if parent_key != "" else key)
        return not_done

    config = {}
    if config_file.endswith(".yml") or config_file.endswith(".yaml"):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    elif config_file.endswith(".json"):
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        raise RuntimeError(f"Config file type not supported: {config_file}")
    # Hack to resolve variables
    not_done = True
    num_iters=0
    while not_done:
        tconfig = copy.deepcopy(config)
        not_done = replace(local_dict=config, global_dict=config, parent_key="")
        num_iters += 1
        if tconfig==config:
            break
        if num_iters > 10:
            raise RuntimeError(f"Could not resolve the variables in {config_file}")
    return config

class Limiter:
    def __init__(self, _obj, max_num_docs):
        self.iter = iter(_obj)
        self.limit = max_num_docs

    def __iter__(self):
        self.read = 0
        return self

    def __next__(self):
        if self.read < self.limit:
            self.read += 1
            r = next(self.iter)
            return r
        else:
            raise StopIteration

def at_most(_obj, limit):
    if limit > 0:
        return iter(Limiter(_obj, limit))
    else:
        return iter(_obj)

def open_stream(file_name: str, write: bool = False, binary=False):
    if write:
        mode = "w"
        cache_dir = os.path.dirname(file_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    else:
        mode = "r"
        if not os.path.exists(file_name):
            raise RuntimeError(f"File {file_name} does not exist!")
    if binary:
        mode = f"{mode}b"

    input_stream = None
    if file_name.endswith(".bz2"):
        import bz2
        input_stream = bz2.open(file_name, mode)
    elif file_name.endswith(".gz"):
        import gzip
        input_stream = gzip.open(file_name, mode)
    elif file_name.endswith(".xz"):
        import lzma
        input_stream = lzma.open(file_name, mode)
    else:  # if file_name.endswith(".jsonl"):
        input_stream = open(file_name, mode)
    return input_stream

def file_is_of_type(input_file, extensions: Union[str, List[str]]):
    """
    Check if the given file is of any of the specified file types, cross-product with compressed extensions.

    Parameters:
        input_file (str): The file path or name to be checked.
        extensions (list): A list of file extensions to check against.

    Returns:
        bool: True if the file is of any of the specified types, False otherwise.
    """
    if isinstance(extensions, str):
        extensions = [extensions]
    return any(input_file.endswith(f"{ext[0]}{ext[1]}")
               for ext in itertools.product(extensions, ['', ".bz2", ".gz", ".xz"]))

def parallel_process(process_func, data, num_threads, post_func=None, post_label=None,
                     msg="Processing items:"):
    """
    This method parallelizes the processing of a list of documents using multiple threads.

    Parameters:
    - process_func: The function to apply to each document. It should take a document as input and return a list of processed items.
    - data: The list of documents to process.
    - num_threads: The number of threads to use for processing the documents.
    - post_func (optional): A function to apply to each processed item after applying the process_func. It should take a processed item as input and return a new item.
    - post_label (optional): A label to use when storing the result of the post_func.
    - msg (optional): The message to display

    Returns:
    - A list of processed items from all the documents.

    Example usage:
    ```python
    def process_document(unit):
        # Process the document and return a list of items
        ...

    data = [...]  # List of documents
    num_threads = 4
    processed_items = parallel_process(process_document, data, num_threads)
    ```
    """
    if num_threads == 1:
        return [process_func(dt) for dt in tqdm(data, desc=msg)]
    doc_queue = Queue()
    manager = Manager()
    d = manager.dict()
    import multiprocessing as mp
    def processor(inqueue, d):
        pid = mp.current_process().pid
        while True:
            try:
                id, text = inqueue.get(block=True, timeout=1)
            except queue.Empty:
                break
            except Exception as e:
                break

            try:
                items = process_func(text)
                if post_func is not None:
                    d[id] = [{**item,
                              post_label: post_func(item)}
                              for item in items]
                else:
                    d[id] = items

            except Exception as e:
                d[id] = []

    for i, doc in enumerate(data):
        doc_queue.put([i, doc])
    processes = []
    for i in range(num_threads):
        p = Process(target=processor, args=(doc_queue, d))
        processes.append(p)
        p.start()
    tk = tqdm(desc=msg, total=doc_queue.qsize())
    c = doc_queue.qsize()
    while c > 0:
        c1 = doc_queue.qsize()
        if c != c1:
            tk.update(c - c1)
            c = c1
        time.sleep(0.1)
    print(f"Dropped out of the while loop: {doc_queue.qsize()}")
    tk.clear()
    for i, p in enumerate(processes):
        p.join()
    tpassages = []
    for i in range(len(data)):
        tpassages.extend(d[i])
    return tpassages
