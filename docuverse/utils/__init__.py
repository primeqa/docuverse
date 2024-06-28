from .embedding_function import DenseEmbeddingFunction
import yaml
import json


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
        return dictionary.get(key, default)


def read_config_file(config_file):
    config = {}
    if config_file.endswith(".yml"):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    elif config_file.endswith(".json"):
        with open(config_file, "r") as f:
            config = json.load(f)
    else:
        raise RuntimeError(f"Config file type not supported: {config_file}")
    return config

class limiter:
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
        return iter(limiter(_obj, limit))
    else:
        return iter(_obj)
