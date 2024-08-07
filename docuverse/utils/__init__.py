import re

from .embedding_function import DenseEmbeddingFunction
import yaml
import json
import copy


def get_param(dictionary, key: str, default: str | None = None):
    def recursive_get(_dictionary, key, default):
        if key.find(".") >= 0:
            keys = key.split(".")
            res = default
            dd = _dictionary
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


def read_config_file(config_file):
    patt = re.compile(r"{{(.*?)}}")
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
