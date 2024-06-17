from docuverse.engines import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig

config = DocUVerseConfig.get_stdargs_config()

engine = SearchEngine(config)

print(engine.get_retriever_info())

