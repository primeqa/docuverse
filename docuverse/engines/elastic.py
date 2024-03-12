try:
    from elasticsearch import Elasticsearch
except:
    print(f"You need to install elasticsearch to be using ElasticSearch functionality!")
    raise RuntimeError("fYou need to install elasticsearch to be using ElasticSearch functionality!")


class ElasticEngine:
    def __init__(self, **kwargs):
        pass

    def search(self, text, **kwargs):
        pass

    def _create_query(self, text="", **kwargs):
        pass

    def ingest_documents(self, documents, **kwargs):
        pass