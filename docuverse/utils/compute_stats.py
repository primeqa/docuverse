from docuverse import SearchEngine
from docuverse.engines.search_engine_config_params import DocUVerseConfig

config = DocUVerseConfig.get_stdargs_config()

engine = SearchEngine(config)

corpus = engine.read_data(config.input_passages, verbose=True)
