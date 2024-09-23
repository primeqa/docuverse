import os

from typing import Union, List
from docuverse.utils import get_param, get_config_dir

class EmbeddingFunction:
    def __init__(self, model_or_directory_name: str, batch_size: int, **kwargs):
        self.batch_size = batch_size

    @staticmethod
    def pull_from_dmf(name):
        try:
            from lakehouse.wrappers import LakehouseIceberg
            from lakehouse.assets import Model
        except ImportError as e:
            print(f"You need to install the DMF library:\n"
                  f"\tpip install git+ssh://git@github.ibm.com/arc/dmf-library.git")
            raise e
        lh = LakehouseIceberg(config="yaml", conf_location=get_config_dir("config/lh-conf.yaml"))
        model = Model(lh=lh)
        if get_param(os.environ, 'DMF_MODEL_CACHE', None) is None:
            os.environ['DMF_MODEL_CACHE'] = os.path.join(os.environ['HOME'], ".local/share/dmf")
        name = model.pull(model=name, namespace="retrieval_slate", table="model_shared")
        return name


    def __call__(self, texts:Union[List[str], str], **kwargs) -> \
        Union[Union[List[float], List[int]], List[Union[List[float], List[int]]]]:
        return self.encode(texts)

    def __del__(self):
        pass

    def create_model(self, model_or_directory_name:str=None, device: str='cpu'):
        pass

    def tokenizer(self):
        return self.model.tokenizer

    def start_pool(self):
        pass

    def stop_pool(self):
        pass

    def encode(self, texts:Union[List[str], str], _batch_size:int=-1, show_progress_bar=False, **kwargs):
        return []

    @property
    def vocab_size(self):
        return self.model.tokenizer.vocab_size

    @property
    def device(self):
        return self.model.model.device