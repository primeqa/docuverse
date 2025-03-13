import copy
import itertools
import json
import os
import queue
import re
import time
from multiprocessing import Queue, Manager, Process
from typing import List, Union
from jinja2 import Template, Undefined

# from .embedding_function import DenseEmbeddingFunction
import yaml
from tqdm.auto import tqdm


def get_param(dictionary: dict|list[dict]|object, key: str, default: str | None | bool = None):
    def recursive_get(_dictionary, key, default):
        if _dictionary is None:
            return default
        if key.find(".") >= 0:
            keys = key.split(".")
            res = default
            if isinstance(_dictionary, dict):
                dd = _dictionary
            else:
                dd = _dictionary.__dict__
            for k in keys:
                if not isinstance(dd, dict):
                    dd = dd.__dict__
                if k in dd:
                    dd = dd[k]
                else:
                    return res
            return dd
        else:
            return _dictionary.get(key, default)

    weird_value = ":+:+"
    if key is None:
        return default
    elif key.find("|") > 0:
        keys = key.split("|")
        for k in keys:
            k = recursive_get(dictionary, k, weird_value)
            if k != weird_value:
                return k
        return default
    else:
        if isinstance(dictionary, list):
            for dct in dictionary:
                val = recursive_get(dct, key, weird_value)
                if val != weird_value:
                    return val
            return default
        else:
            return recursive_get(dictionary, key, default)

def get_config_dir(config_path: str | None = None) -> str:
    def find_docuverse_base(path: str, basename="docuverse"):
        dir = path
        head = os.path.basename(dir)
        found = False
        while dir:
            if head == basename and os.path.basename(os.path.dirname(dir)) != basename:
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

class NullUndefined(Undefined):
  def __getattr__(self, key):
    return ''

# def read_config_file(config_file):
#     if not os.path.exists(config_file):
#         config_file = os.path.join(get_config_dir(os.path.dirname(config_file)), os.path.basename(config_file))
#     config = {}
#     if config_file.endswith(".yml") or config_file.endswith(".yaml"):
#         with open(config_file, "r") as f:
#             config = yaml.safe_load(f)
#     elif config_file.endswith(".json"):
#         with open(config_file, "r") as f:
#             config = json.load(f)
#     else:
#         raise RuntimeError(f"Config file type not supported: {config_file}")
#     # Hack to resolve variables
#     not_done = True
#     num_iters=0
#     config = Template(config, undefined=NullUndefined).render()
#     # while not_done:
#     #     tconfig = copy.deepcopy(config)
#     #     not_done = replace(local_dict=config, global_dict=config, parent_key="")
#     #     num_iters += 1
#     #     if tconfig==config:
#     #         break
#     #     if num_iters > 10:
#     #         raise RuntimeError(f"Could not resolve the variables in {config_file}")
#     return config


def read_config_file(config_file):
    patt = re.compile(r"{{(.*?)}}")
    if not os.path.exists(config_file):
        config_file = os.path.join(get_config_dir(os.path.dirname(config_file)), os.path.basename(config_file))
    def replace(local_dict:dict, global_dict:dict, parent_key: str=""):
        def get_value_recursive(global_dict, val, parent_key):
            parents = parent_key.split(".")
            rr = None
            while rr is None:
                if len(parents) > 0:
                    skey = f"{'.'.join(parents)}.{val}"
                else:
                    skey = val
                rr = get_param(global_dict, skey)
                if rr is not None:
                    return rr
                if len(parents) == 0:
                    return "None"
                parents = parents[:-1]

        not_done = False
        for key, val in local_dict.items():
            if isinstance(val, str):
                m = re.search(patt, val)
                while m:
                    rr = get_value_recursive(global_dict, m.group(1).replace(" ",""), parent_key)
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
                     msg="Processing result:"):
    """
    This method parallelizes the processing of a list of documents using multiple threads.

    Parameters:
    - process_func: The function to apply to each document. It should take a document as input and return a list of processed result.
    - data: The list of documents to process.
    - num_threads: The number of threads to use for processing the documents.
    - post_func (optional): A function to apply to each processed item after applying the process_func. It should take a processed item as input and return a new item.
    - post_label (optional): A label to use when storing the result of the post_func.
    - msg (optional): The message to display

    Returns:
    - A list of processed result from all the documents.

    Example usage:
    ```python
    def process_document(unit):
        # Process the document and return a list of result
        ...

    data = [...]  # List of documents
    num_threads = 4
    processed_items = parallel_process(process_document, data, num_threads)
    ```
    """
    def apply_funcs(text):
        result = process_func(text)
        if post_func is not None:
            if isinstance(result[0], dict):
                result = [{**item,
                          post_label: post_func(item)}
                         for item in result]
            else:
                setattr(result, post_label, post_func(result))
        return result

    if num_threads <= 1:
        return [apply_funcs(dt) for dt in tqdm(data, desc=msg, smoothing=1)]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    num_items = len(data)

    doc_queue = Queue()
    manager = Manager()
    d = manager.dict()
    import multiprocessing as mp
    def processor(inqueue, d, thread_number, size):
        pid = mp.current_process().pid
        with tqdm(desc=f"{msg}/thread {thread_number}", leave=False,
                  position=thread_number+1, total=2*size) as tk1:
            while True:
                try:
                    id, text = inqueue.get(block=True, timeout=1)
                except queue.Empty:
                    break
                except Exception as e:
                    break
                try:
                    res = apply_funcs(text)
                    d[id] = res
                    tk1.update(1)
                except ImportError as e:
                    print(f"Error in thread {thread_number}: {e}")
                    break
                except Exception as e:
                    d[id] = []

    for i, doc in enumerate(data):
        doc_queue.put([i, doc])
    processes = []
    tk = tqdm(desc=f"{msg}:", total=doc_queue.qsize(), leave=True, position=0)
    c = doc_queue.qsize()
    for i in range(num_threads):
        p = Process(target=processor, args=(doc_queue, d, i, doc_queue.qsize()/num_threads,))
        processes.append(p)
        p.start()
    while c > 0:
        c1 = doc_queue.qsize()
        if c != c1:
            tk.update(c - c1)
            c = c1
        time.sleep(0.1)
    # print(f"Dropped out of the while loop: {doc_queue.qsize()}")
    for p in processes:
        p.join()
    tk.clear()
    tk.close()
    return list(d[i] for i in range(num_items))

def ask_for_confirmation(text, answers=['yes', 'no', 'skip'], default:str='yes') -> str:
    """
    ask_for_confirmation(text, answers=['yes', 'no', 'skip'], default='yes')

    Prompts the user with a given text and waits for an answer from a list of possible answers.
    If the user provides no input, a default answer is returned.

    Parameters:
        text (str): The message text to display to the user.
        answers (list of str): A list of acceptable answers. Default is ['yes', 'no', 'skip'].
        default (str): The default answer to return if the user provides no input. Default is 'yes'.

    Returns:
        str: The user's response or the default answer if no input is provided.
    """
    display_answers = ", ".join(a.title() if a==default else a for a in answers)
    print(text)
    while True:
        r = input(f"Say: {display_answers}, <enter>={default}:").strip()
        if r=="":
            return default
        elif r in answers:
            return r
        else:
            print(f"Please type one of {answers}, not {r}!")

def convert_to_single_vectors(embs):
    return [embs[[i], :] for i, _ in enumerate(embs)]

def vector_is_empty(vector):
    return (
            (getattr(vector, '_nnz', None) is not None and vector._nnz() == 0) or
            (getattr(vector, 'count_nonzero', None) is not None and int(vector.count_nonzero()) == 0)
    )

def prepare_for_save_and_backup(output_file, overwrite=False):
    path = os.path.dirname(output_file)
    if not os.path.exists(path):
        os.makedirs(path)
    if not overwrite and os.path.exists(output_file):
        # Make a copy before writing over
        import shutil
        template, extension = os.path.splitext(output_file)
        i = 1
        # template = output_file.replace(f".{extension}", "")
        while os.path.exists(f"{template}.bak{i}{extension}"):
            i += 1
        shutil.copy2(output_file, f"{template}.bak{i}{extension}")

def log_program():
    """
    Logs the execution of a program, including the timestamp, the user who executed
    the program, and the command-line arguments used.

    This function appends the log information to a file named "logfile". The
    timestamp is captured at the time of the function execution, the username is
    retrieved from the environment variable 'USER', and the command-line arguments
    are read from the `sys.argv` list.

    Raises:
        EnvironmentError: If there is no 'USER' environment variable set.
        IOError: If there is an error in appending to the log file.

    """
    from datetime import datetime
    import sys
    with open("logfile", "a") as cmdlog:
        cmdlog.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {os.getenv('USER')} - "
                     f"{' '.join(sys.argv)}\n")

def get_orig_docid(id):
    """
    Determines and returns the original document ID by processing the input ID.

    If the input ID is an integer, it is returned as-is. For string IDs, it identifies
    and returns the substring up to the second-to-last occurrence of a hyphen ("-"),
    or returns the original string if there are fewer than two hyphens.

    Args:
        id (int | str): The ID to process. It can be an integer or a string
            containing hyphens.

    Returns:
        int | str: The processed original document ID. If the input ID is an
            integer, it is returned directly. If it is a string, the function
            returns the substring up to the second-to-last hyphen, or the original
            string if the hyphen criteria are not met.
    """
    if isinstance(id, int):
        return id
    index = id.rfind("-", 0, id.rfind("-"))
    if index >= 0:
        return id[:index]
    else:
        return id
