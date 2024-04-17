from typing import Dict, List
class SearchData:
    """
    An abstraction of the search data, encompassing SearchCorpus, SearchQueries, and SearchResult. It has named fields
    for 'text' and 'title', and knows how to read the data from disk. Other than that, it's a glorified dictionary.
    """
    class Entry:
        def __init__(self, dict: Dict[str, str]):
            return

        def __init__(self, filenames, text_field_name: str="text", title_field_name: str="title"):
            self.dict = dict
            # self.__dict__.update(data)

        def get_text(self, i:int) -> str:
            return
