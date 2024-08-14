import re
import os
# from .embedding_function import DenseEmbeddingFunction
import yaml
import json
import copy


def get_param(dictionary, key: str, default: str | None = None):
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
    if get_param(os.environ, 'DOCUVERSE_CONFIG_PATH') is not None:
        config_dir = os.environ['DOCUVERSE_CONFIG_PATH']
    elif config_path is None or not os.path.exists(config_path):
        possible_paths = ['.', os.path.join(os.path.dirname(__file__), "../../../..")]
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
