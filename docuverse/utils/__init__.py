from .embedding_function import DenseEmbeddingFunction
import yaml
import json

def get_param(dictionary, key: str, default: str | None = None):
    if key is None:
        return default
    elif key.find("|") > 0:
        weird_value = ":+:+"
        keys = key.split("|")
        for k in keys:
            k = dictionary.get(k, weird_value)
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
