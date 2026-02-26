import itertools
import os
import queue
import time
import sys
import re
import copy
import json

if sys.platform == "darwin":
    print("We're on a Mac !!")
    import multiprocess as mp
else:
    import multiprocessing as mp

from typing import List, Union, Any
from jinja2 import Template, Undefined

# from .embedding_function import DenseEmbeddingFunction
import yaml
from tqdm.auto import tqdm


def detect_device():
    import torch
    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
    return device


from typing import Any, Union

# Extract constant
_SENTINEL = object()


def _is_number(k: str) -> Union[int, None]:
    """Extract helper function - convert string to int or return None"""
    try:
        return int(k)
    except ValueError:
        return None


def _recursive_get(dictionary: Any, key: str, default: Any) -> Any:
    """Extract helper function - recursively get value from nested structures"""
    if dictionary is None:
        return default
    elif isinstance(dictionary, dict) and key in dictionary:
        return dictionary.get(key)
    elif "." in key:
        keys = key.split(".")
        current = dictionary if isinstance(dictionary, dict) else dictionary.__dict__

        for k in keys:
            if isinstance(current, list):
                index = _is_number(k)
                if index is not None:  # Fix bug: was 'val' instead of 'vv'
                    current = current[index]
                    continue
                else:
                    raise RuntimeError(f"Could not find key {k} in list: {current}")
            if not isinstance(current, dict):
                current = current.__dict__
            if k in current:
                current = current[k]
            else:
                return default
        return current
    else:
        return dictionary.get(key, default) if hasattr(dictionary, 'get') else default


def get_param(dictionary: Union[dict, list[dict], object], key: str,
              default: Union[str, None, bool, int, float, dict] = None) -> Any:
    """Get parameter from dictionary with support for nested keys and fallback options"""
    if key is None:
        return default
    elif "|" in key:
        # Handle fallback keys separated by |
        keys = key.split("|")
        for k in keys:
            result = _recursive_get(dictionary, k, _SENTINEL)
            if result is not _SENTINEL:
                return result
        return default
    else:
        if isinstance(dictionary, list):
            # Search through list of dictionaries
            for dct in dictionary:
                val = _recursive_get(dct, key, _SENTINEL)
                if val is not _SENTINEL:
                    return val
            return default
        else:
            return _recursive_get(dictionary, key, default)

def get_config_dir(config_path: str | None = None) -> str:
    def find_docuverse_base(path: str, basename="docuverse"):
        dir = path
        head = os.path.basename(dir)
        found = False
        while dir:
            if head == basename and os.path.basename(os.path.dirname(dir)) != basename:
                return dir
            dir = os.path.dirname(dir)
            if dir == "/":
                return None
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

# Constants

def _resolve_variable(global_dict, variable_name, parent_key):
    """
    Resolve a variable by searching through parent keys in the configuration.
    
    Args:
        global_dict (dict): The full configuration dictionary
        variable_name (str): The variable name to resolve. The key can be multi-level - levels are split with '.'.
        parent_key (str): The parent key path to start searching from
        
    Returns:
        The resolved value or "None" as a string if not found
    """
    variable_name = variable_name.replace(" ", "")
    parents = parent_key.split(".")

    while True:
        if parents:
            search_key = f"{'.'.join(parents)}.{variable_name}"
        else:
            search_key = variable_name

        result = get_param(global_dict, search_key)
        if result is not None:
            return result

        if not parents:
            return "None"

        parents = parents[:-1]


def _process_dictionary(local_dict, global_dict, parent_key=""):
    """
    Process a dictionary to resolve templated variables.
    
    Args:
        local_dict (dict): The dictionary to process
        global_dict (dict): The full configuration dictionary
        parent_key (str): The parent key path for variable resolution
        
    Returns:
        bool: True if some variables couldn't be fully resolved yet
    """
    VARIABLE_PATTERN = re.compile(r"{{(.*?)}}")
    has_unresolved = False

    for key, value in local_dict.items():
        if isinstance(value, str):
            # Process string values that may contain variables
            new_value = value
            match = re.search(VARIABLE_PATTERN, new_value)

            while match:
                var_name = match.group(1)
                resolved_value = _resolve_variable(global_dict, var_name, parent_key)

                # Check if the resolved value itself contains variables
                if isinstance(resolved_value, str) and "{{" in resolved_value:
                    has_unresolved = True
                    break

                # Replace the variable with its resolved value
                new_value = new_value.replace(f"{{{{{var_name}}}}}", str(resolved_value))
                match = re.search(VARIABLE_PATTERN, new_value)

            local_dict[key] = new_value

        elif isinstance(value, dict):
            # Process nested dictionaries
            next_parent = f"{parent_key}.{key}" if parent_key else key
            nested_unresolved = _process_dictionary(value, global_dict, next_parent)
            has_unresolved = has_unresolved or nested_unresolved

    return has_unresolved


def load_config_from_file(file_path):
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        file_path (str): Path to the configuration file
        
    Returns:
        dict: The loaded configuration
        
    Raises:
        RuntimeError: If the file type is not supported
    """
    if file_path.endswith((".yml", ".yaml")):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise RuntimeError(f"Config file type not supported: {file_path}")

def _is_scalar(value):
    """Check if a value is a scalar (non-iterable or string). A bit of a over-caution, but better safe than sorry."""
    scalar_types = (int, float, str, bool, complex, type(None))
    return isinstance(value, scalar_types)


def _replace_leaf_keys(config: dict[str, Any], override_vals: dict[str, str]) -> dict[str, Any]:
    """
    Recursively replaces scalar leaf values in a nested dictionary with values
    from the override_vals dictionary. If a key in the override_vals dictionary
    matches a scalar leaf key in the config dictionary, the leaf value in config
    will be replaced with the corresponding value from override_vals.

    Args:
        config (dict[str, Any]): A nested dictionary with scalar leaf values
            that may be replaced.
        override_vals (dict[str, str]): A dictionary where keys correspond
            to the scalar keys in the config dictionary that need to be
            replaced, and values are the new values to assign.

    Returns:
        dict[str, Any]: The updated dictionary with the specified scalar
        leaf values replaced by the override values.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = _replace_leaf_keys(value, override_vals)
        elif _is_scalar(value):
            if key in override_vals:
                config[key] = override_vals[key]
    return config


def read_config_file(config_file, override_vals: dict[str, str]=None) -> dict[str, Any]:
    """
    Reads a configuration file, resolves templated variables within the file, and returns the
    parsed configuration as a dictionary.

    The function supports YAML and JSON configuration files. Variables within the configuration
    file are expressed using the syntax `{{variable}}` and are resolved by recursively searching
    through the configuration dictionary, considering the current and parent keys.

    Args:
        override_vals:
        config_file (str): Path to the configuration file to read. The file may contain
            variables to be resolved.

    Returns:
        dict: A dictionary representation of the processed and resolved configuration file.

    Raises:
        RuntimeError: If the file type is not supported or if variable resolution exceeds the
            allowed number of iterations (10).
    """
    MAX_RESOLUTION_ITERATIONS = 10

    # Resolve file path if it doesn't exist
    if not os.path.exists(config_file):
        config_file = os.path.join(get_config_dir(os.path.dirname(config_file)),
                                   os.path.basename(config_file))

    # Load configuration from file
    config = load_config_from_file(config_file)
    if override_vals is not None:
        config = _replace_leaf_keys(config, override_vals)

    # Resolve variables in the configuration
    iterations = 0
    has_unresolved = True

    while has_unresolved and iterations < MAX_RESOLUTION_ITERATIONS:
        previous_config = copy.deepcopy(config)
        has_unresolved = _process_dictionary(config, config)
        iterations += 1

        # If no changes were made in this iteration, we're done
        if previous_config == config:
            break

    # Check if we hit the maximum number of iterations
    if iterations >= MAX_RESOLUTION_ITERATIONS and has_unresolved:
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
    if binary: # or file_is_of_type(file_name, ['bz2', 'xz', 'gz']):
        mode = f"{mode}b"
    else:
        mode = f"{mode}t"

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

_parallel_process_func = None
_parallel_post_func = None
_parallel_post_label = None

def _parallel_queue_worker(inqueue, result_queue, thread_number, expected_items, msg):
    """Worker with its own tqdm progress bar. Uses fork-inherited globals."""
    import queue as queue_module

    with tqdm(desc=f"{msg}/thread {thread_number}", leave=False,
              position=thread_number + 1, total=expected_items) as tk:
        while True:
            try:
                idx, text = inqueue.get(block=True, timeout=1)
            except queue_module.Empty:
                break
            except Exception:
                break
            try:
                result = _parallel_process_func(text)
                if _parallel_post_func is not None:
                    if isinstance(result[0], dict):
                        result = [{**item,
                                  _parallel_post_label: _parallel_post_func(item)}
                                 for item in result]
                    else:
                        setattr(result, _parallel_post_label, _parallel_post_func(result))
                result_queue.put((idx, result))
            except Exception as e:
                print(f"Error in thread {thread_number}: {e}")
                result_queue.put((idx, []))
            tk.update(1)

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

    import multiprocessing as mp

    # Use 'fork' context for fast startup (avoids re-importing modules).
    # Fall back to 'spawn' if fork is unavailable (e.g., macOS with CUDA).
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        ctx = mp.get_context('spawn')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set module-level globals so forked workers inherit them
    global _parallel_process_func, _parallel_post_func, _parallel_post_label
    _parallel_process_func = process_func
    _parallel_post_func = post_func
    _parallel_post_label = post_label

    num_docs = len(data)
    inqueue = ctx.Queue()
    result_queue = ctx.Queue()

    for i, doc in enumerate(data):
        inqueue.put((i, doc))

    items_per_worker = (num_docs + num_threads - 1) // num_threads
    processes = []
    for i in range(num_threads):
        p = ctx.Process(target=_parallel_queue_worker,
                        args=(inqueue, result_queue, i, items_per_worker, msg))
        processes.append(p)
        p.start()

    # Collect results with main progress bar at position 0
    results = [None] * num_docs
    tk = tqdm(desc=msg, total=num_docs, position=0)
    collected = 0
    while collected < num_docs:
        try:
            idx, result = result_queue.get(timeout=30)
            results[idx] = result
            collected += 1
            tk.update(1)
        except Exception:
            if not any(p.is_alive() for p in processes):
                break
    tk.close()

    for p in processes:
        p.join()

    return results

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
    try:
        while True:
            r = input(f"Say: {display_answers}, <enter>={default}:").strip()
            if r=="":
                return default
            elif r in answers:
                return r
            else:
                print(f"Please type one of {answers}, not {r}!")
    except EOFError:
        import simple_colors

        print(f" <No input available: returning '{simple_colors.red(default)}'>")
        return default

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


id_format = re.compile("(.*)-(\\d+)-(\\d+)")


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
    # index = id.rfind("-", 0, id.rfind("-"))
    m = id_format.match(id)
    if m:
        return m.group(1)
    else:
        return id
    # if index >= 0:
    #     return id[:index]
    # else:
    #     return id

def save_command_line(args, output="logfile"):
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
    with open(output, "a") as cmdlog:
        cmdlog.write(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {os.getenv('USER')} - "
                     f"python {' '.join(args)}\n")


def _trim_json(data, max_string_len: int=9999):
    """
    Trims JSON-compatible data structures to ensure string values do not exceed a
    specified length. This function recursively traverses dictionaries, lists,
    and strings, limiting string values to a maximum length of 9999 characters.

    Args:
        data: The JSON-compatible data structure to be trimmed. This can be a
            dictionary, list, or string.

    Returns:
        The trimmed JSON-compatible data structure with string values limited to a
        maximum length of 9999 characters.

    """
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = _trim_json(v)
    elif isinstance(data, list):
        data = [_trim_json(v) for v in data]
    elif isinstance(data, str):
        data = data[:max_string_len]
    return data

def convert_to_type(model_torch_dtype = None):
    import torch
    dtype_mapping = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.complex64": torch.complex64,
        "torch.complex128": torch.complex128,
    }
    if model_torch_dtype is None:
        return torch.float32
    else:
        return dtype_mapping.get(model_torch_dtype, torch.float32)
