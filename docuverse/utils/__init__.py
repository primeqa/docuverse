from .embedding_function import DenseEmbeddingFunction

def get_param(dictionary, key, default=None):
    return dictionary[key] if key in dictionary else default