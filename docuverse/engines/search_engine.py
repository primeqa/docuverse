import yaml
import os
from docuverse.engines import SearchResult, SearchCorpus, SearchQueries


class SearchEngine(object):
    DEFAULT_CACHE_DIR = os.path.join(f"{os.getenv('HOME')}", ".local", "share", "elastic_ingestion")

    def __init__(self, retrieval_params, reranking_params=None, **kwargs):
        self.retrieval_params = retrieval_params
        self.reranking_params = reranking_params
        self.duplicate_removal = self.get_param(kwargs, 'duplicate_removal')
        if self.duplicate_removal is not None:
            if self.duplicate_removal == "rouge":
                from rouge_score.rouge_scorer import RougeScorer

                self.rouge_scorer = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            else:
                self.rouge_scorer = None
        self.cache_dir = self.get_param('cache_dir', self.DEFAULT_CACHE_DIR)
        self.cache_policy = self.get_param('cache_policy', 'always')

    @staticmethod
    def read_config(config_path):
        """
        Reads the configuration file at the specified path and returns the retrieved values for retrieval
        and reranking. The configuration file is consistent to the mf-coga config file (if that doesn't make sense
        don't worry about it :) ).
        The format of the yaml file is as follows:
          * The file should contain 'retrieval' and 'reranking' parameters; if no 'retrieval' is present, then the
          file is assumed to be flat.
          * The 'retrieval' parameter should have a value 'name' which will be used to decide the engine type.

        :param config_path: The path to the configuration file.
        :type config_path: str

        :return: A tuple containing the retrieved values for retrieval and reranking.
                 If the configuration file does not contain values for retrieval or reranking,
                 the corresponding value in the tuple will be None.
        :rtype: tuple

        :raises yaml.YAMLError: If there is an error while loading the configuration file.
        """
        try:
            vals = yaml.safe_load(config_path)
            return vals['retrieval'] if 'retrieval' in vals else vals, vals[
                'reranking'] if 'reranking' in vals else None

        except yaml.YAMLError as exc:
            raise exc

    @staticmethod
    def get_param(dict, key, default=None):
        return dict[key] if key in dict else default

    @staticmethod
    def create(config_path, **kwargs):
        retrieval_config, reranking_config = SearchEngine.read_config(config_path)

        if reranking_config is not None:
            reranker = SearchEngine._create_reranker(reranking_config)
        retriever = SearchEngine._create_retriever(retrieval_config)

    @staticmethod
    def _create_retriever(retrieval_config):
        """
        Create a retriever object based on the given retrieval configuration.

        Parameters:
        retrieval_config (dict): A dictionary containing the retrieval configuration.

        Raises:
        RuntimeError: If the docuverse_elastic package is not installed.

        Returns:
        engine: A retriever object.

        """
        name = SearchEngine.get_param(retrieval_config, 'name')
        try:
            from docuverse.engines.retrieval import elastic
        except ImportError as e:
            print(f"You need to install the docuverse_elastic package!")
            raise e

        if name.startswith('elastic'):
            if name == 'elastic_bm25':
                engine = elastic.ElasticBM25Engine(retrieval_config)
            elif name == 'elastic_dense':
                engine = elastic.ElasticDenseEngine(retrieval_config)
            elif name == 'elastic_elser':
                engine = elastic.ElasticElserEngine(retrieval_config)
            elif name == "elastic_hybrid":
                engine = elastic.ElasticHybridEngine(retrieval_config)
        elif name.startswith('primeqa'):
            pass
        elif name == 'chromadb':
            try:
                from docuverse.engines.retrieval.vectordb.chromadb import ChromaDBEngine
                engine = ChromaDBEngine(retrieval_config)
            except ImportError as e:
                print("You need to install docuverse_chomadb package.")
                raise e
        elif name == 'milvus':
            try:
                from docuverse.engines.retrieval.milvus import MilvusEngine
                engine = MilvusEngine(retrieval_config)
            except ImportError as e:
                print("You need to install docuverse_chomadb package.")
                raise e

    def remove_duplicates(self, results: SearchResult, duplicate_removal: str,
                          rouge_duplicate_threshold: float):
        res = results
        if duplicate_removal == "none":
            return res
        if len(res) == 0:
            return results
        ret = []
        if duplicate_removal == "exact":
            seen = {res[0]['_source']['text']: 1}
            ret = [res[0]]
            for r in res[1:]:
                text_ = r['_source']['text']
                if text_ not in seen:
                    seen[text_] = 1
                    ret.append(r)
        elif duplicate_removal == "rouge":
            for r in res[1:]:
                found = False
                text_ = r['_source']['text']
                for c in ret:
                    scr = self.rouge_scorer.score(c['_source']['text'], text_)
                    if scr['rougel'].fmeasure >= rouge_duplicate_threshold:
                        found = True
                        break
                if not found:
                    ret.append(r)
        return ret

    @classmethod
    def _create_reranker(cls, reranking_config):
        if reranking_config == None:
            return None

        from docuverse.engines.reranking.Reranker import Reranker
        reranker = Reranker(reranking_config)
        return reranker

    def ingest(self, corpus: SearchCorpus):
        pass

    def search(self, corpus: SearchQueries) -> SearchResult:
        pass

    def set_index(self, index=None):
        pass
