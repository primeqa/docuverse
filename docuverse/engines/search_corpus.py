class SearchCorpus(object):
    """
    A class for processing and managing a corpus of documents.

    This class reads, parses, and provides utility functions for handling document
    corpora. It supports reading and parsing files (*.tsv) containing text data
    such as passages or questions with schema fields like `id`, `text`, `title`, etc.
    The documents can be accessed via indexing, iteration, or length-based operations.

    Attributes:
        filepaths (List[str]): The paths to the files that contain the document corpus.
        Documents (List[dict]): A list of parsed documents from the provided file paths.
    """
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.documents = []
        for data_entry in self.filepaths:
            self.documents.extend(SearchCorpus.parse_document(data_entry))
    
    def __getitem__(self, index):
        return self.documents[index]

    def __len__(self):
        return len(self.documents)

    @classmethod
    def parse_document(cls, filepath, **kwargs):
        import pathlib
        file_ext = pathlib.PurePosixPath(filepath).suffix
        if file_ext == '.tsv':
            return SearchCorpus.parse_document_tsv(filepath, **kwargs)
        else:
            raise Exception("No other file type supported!")
    
    @classmethod
    def parse_document_tsv(cls, input_file, fields=None):
        """
        Utility function to read the standard tuple representation for both passages and questions.
        Args:
            input_file: the tsv file with either contexts or queries, tab-separated.
            fields: the schema of the file. It can be either ['id', 'text', 'title'] for contexts,
            ['id', 'text', 'relevant', 'answers'] or ['id', 'text', 'relevant'] for questions

        Returns: List[Dict[AnyStr: AnyStr]]
        the list of tuples

        """
        import csv
        passages = []
        if fields is None:
            num_args = 3
        else:
            num_args = len(fields)
        from tqdm import tqdm
        with open(input_file) as in_file:
            csv_reader = \
                csv.DictReader(in_file, fieldnames=fields, delimiter="\t") \
                    if fields is not None \
                    else csv.DictReader(in_file, delimiter="\t")
            next(csv_reader)
            for row in tqdm(csv_reader):
                assert len(row) in [2, 3, 4], f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
                itm = {'text': (row["title"] + ' '  if 'title' in row else '') + row["text"],
                                'id': row['id']}
                if 'title'in row:
                    itm['title'] = row['title']
                if 'relevant' in row:
                    itm['relevant'] = row['relevant']
                if 'answers' in row and row['answers'] is not None:
                    itm['answers'] = row['answers'].split("::")
                passages.append(itm)
        return passages

    def __iter__(self):
        return iter(self.documents)

    def prepare_for_ingestion(self, max_doc_length: int = 512, stride: int = 256, title_handling: str = "all"):
        # TODO - Add code using TextTiler
        return self.documents
