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
    filters: dict
    duplicate_removal: str
    rouge_duplicate_threshold: float
    
    def __init__(self, config: dict):
        self.index = config['index']
        self.title_field = get_param(config, 'title_field', None)
        self.text_field = get_param(config, 'text_field', None)
        self.productId_field = get_param(config, 'productId_field', None)
        self.fields = get_param(config, 'fields', None)
        self.n_docs = get_param(config, 'n_docs', 30)
        self.search_type = get_param(config, 'search-type', None)
        self.filters = get_param(config, 'filters', None)
        # "Defines the strategy for removing duplicates (default: don't remove). It can be 'rouge' (based on rouge similarity) or 'exact' (exact match)")
        # choices=["none", "rouge", "exact"], default="none",
        self.duplicate_removal = get_param(config, 'duplicate_removal', None)
        # "The rouge-l F1 similarity for dropping duplicated in the result (default 0.7)"
        # default=-1, type=float,
        self.rouge_duplicate_threshold = get_param(config, 'rouge_duplicate_threshold', 0.7)
