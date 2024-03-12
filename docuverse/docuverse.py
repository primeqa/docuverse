from transformers import SentenceTransformers
from typing import List, Dict
from argparse import ArgumentParser

class DocuVerse:
    class SearchResult:
        def __init__(self):
            pass

    def __init__(self, **kwargs):
        pass

    def search(self, text, **kwargs) -> SearchResult:
        pass

    def ingest_documents(self, documents: List[Dict[str: str]]):
        pass