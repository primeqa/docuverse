from docuverse.utils import get_param
from dataclasses import dataclass

@dataclass
class ConfigParams:
    index: str
    title_field: str    
    text_field: str
    productId_field: str
    n_docs: int
    search_type: str
    fields: list
    filter_fields: list
    duplicate_removal: str
    rouge_duplicate_threshold: float
    
    def __init__(self, config: dict):
        self.index = config['index']
        self.title_field = config['title_field']
        self.text_field = config['text_field']
        self.productId_field = config['productId_field']
        self.fields = config['fields']
        self.n_docs = config['n_docs']
        self.search_type = config['search-type']
        self.filter_fields = get_param(config, 'filter_fields', None)
        # "Defines the strategy for removing duplicates (default: don't remove). It can be 'rouge' (based on rouge similarity) or 'exact' (exact match)")
        # choices=["none", "rouge", "exact"], default="none",
        self.duplicate_removal = get_param(config, 'duplicate_removal', None)
        # "The rouge-l F1 similarity for dropping duplicated in the result (default 0.7)"
        # default=-1, type=float,
        self.rouge_duplicate_threshold = get_param(config, 'rouge_duplicate_threshold', 0.7)
