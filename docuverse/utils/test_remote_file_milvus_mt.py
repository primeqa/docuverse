from dataclasses import dataclass
from typing import Any

import click
import json
import pandas as pd
import os
from pathlib import Path

from accelerate.commands.config.default import description
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from docuverse.utils.milvus_server_management import create_milvus_server, MilvusServer
from docuverse.utils import read_config_file, get_param
from docuverse.utils.timer import timer

@dataclass
class MilvusServerConfig:
    attn: str
    collection_name: str
    model_name: str
    milvus_file_name: str

@click.command()
@click.option('--milvus-dir', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing milvus.db file')
@click.option('--queries-file', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='TSV file containing queries in "query" column')
@click.option('--model-name', type=str, default='ibm-granite/granite-embedding-english-r2',
              help='Name of the SentenceTransformer model to use')
@click.option('--collection-name', type=str, default='default_collection',
              help='Name of the Milvus collection to use')
@click.option('--config', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help='Configuration file')
@click.option('--workers', type=int, default=1,
              help="Number of workers to run in parallel the Milvus server")
@click.option('-o', '--output-file', type=click.Path(file_okay=True, dir_okay=False),
              help='Output JSONL file path', default='search_results.jsonl')
@click.option('--server-only', is_flag=True, help='Start server and keep it running indefinitely')
@click.option('-b', '--batch-size', type=int, default=256, help='Batch size for encoding queries')
def process_queries(milvus_dir, queries_file, model_name, collection_name, config, workers, output_file, server_only,
                    batch_size):
    """Run Milvus server and process queries from TSV file."""
    tm = timer("Milvus File Server Test")
    server_info = read_configuration(collection_name, config, milvus_dir, model_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tm.add_timing("init::milvus_server_init")
    if server_only:
        server = create_milvus_server(db_path=server_info.milvus_file_name, port=8765, use_api=True)
        print("Server started. Press Ctrl+C to stop...")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
            server.close()
    else:
        model = create_st_model(server_info)
        queries_df = pd.read_csv(queries_file, sep='\t')
        queries_text = queries_df['query'].tolist()
        queries = [l for l in model.encode(queries_text, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)]
        tm.add_timing("encode::sentence_transformer_encode")
        if workers > 1:
            from multiprocessing import Pool
            server = create_milvus_server(db_path=server_info.milvus_file_name, use_api=True, port=8765)
            with Pool(workers) as p:
                num_q = sum(p.starmap(run_queries, [(i, f"{output_file}_{i}", queries_text, queries, server_info, tm) for i in range(workers)]))
            server.close()
        else:
            num_q = run_queries(0, output_file, queries_text, queries, server_info, tm)

        tm.display_timing(tm.milliseconds_since_beginning(), keys={'queries': num_q}, sorted_by="%", reverse=True)
        print(f"Results written to {output_file}")

def read_queries(queries_file):
    queries_df = pd.read_csv(queries_file, sep='\t')
    if 'query' not in queries_df.columns:
        raise ValueError("TSV file must contain a 'query' column")
    return queries_df['query']

def run_queries(thread_number, output_file, queries_text, queries, server_info: MilvusServerConfig, tm: timer) -> int | None:
    server = create_milvus_server(db_path=server_info.milvus_file_name, port=8765, use_api=True)
    try:
        tm.add_timing("init::sentence_transformer_init")
        # Read queries from TSV file
        first_q = True
        # Process each query
        with open(output_file, 'w') as f:
            for qtext, query in tqdm(zip(queries_text, queries), total=len(queries),
                                     desc=f"Processing queries [{thread_number}]",
                                     leave=False, position=thread_number):
                results = server.search(collection_name=server_info.collection_name,
                                        query_vector=query,
                                        limit=5,
                                        output_fields=['text', 'id'])
                tm.add_timing("decode::milvus_search")
                result_entry = {
                    'query': qtext,
                    'results': results
                }
                f.write(json.dumps(result_entry) + '\n')
                tm.add_timing("decode::output_jsonl")
        return len(queries)
    finally:
        server.close()


def create_st_model(server_info: MilvusServerConfig) -> SentenceTransformer:
    model_args = {}
    if server_info.attn.find("flash") >= 0:
        model_args: dict[str, Any] = {"attn_implementation": server_info.attn}
        import torch
        model_args["torch_dtype"] = torch.bfloat16

    # Initialize the sentence transformer model
    model = SentenceTransformer(server_info.model_name, device="cuda",
                                model_kwargs=model_args,
                                trust_remote_code=True)
    return model


def read_configuration(collection_name, config, milvus_dir, model_name) -> MilvusServerConfig:
    cfg = read_config_file(config)['retriever']
    # Milvus does not accept '-' in the name, and docuverse replaces them with '_'
    collection_name = get_param(cfg, 'index_name', collection_name).replace("-", "_")
    milvus_dir = get_param(cfg, 'project_dir', milvus_dir)
    milvus_file_name = get_param(cfg, 'server', "")
    model_name = get_param(cfg, 'model_name', model_name)
    if milvus_file_name.find("file:") == -1:
        milvus_file_name = os.path.join(milvus_dir, 'milvus.db')
    else:
        milvus_file_name = milvus_file_name.replace("file:", "")
    attn = get_param(cfg, 'attn_implementation', 'sdpa')
    # Initialize and start Milvus server
    return MilvusServerConfig(attn, collection_name, model_name, milvus_file_name)


if __name__ == '__main__':
    process_queries()
