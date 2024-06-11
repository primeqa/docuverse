from .embedding_function import DenseEmbeddingFunction


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
