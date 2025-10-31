from dataclasses import dataclass
from typing import Any

import argparse
import json
import pandas as pd
import os
import yaml
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


def process_queries(args):
    """Run Milvus server and process queries from TSV file."""
    tm = timer("Milvus File Server Test")
    server_info = read_configuration(args.collection_name, args.config, args.milvus_dir, args.model_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_args():
    parser = argparse.ArgumentParser(description='Run Milvus server and process queries from TSV file.')
    parser.add_argument('--milvus-dir', '--milvus_dir', type=str, required=True,
                        help='Directory containing milvus.db file')
    parser.add_argument('--query-file',  '--queries-file', '--query_file', type=str, required=True,
                        help='TSV file containing queries in "query" column')
    parser.add_argument('--model-name', '--model_name', type=str,
                        default='ibm-granite/granite-embedding-english-r2',
                        help='Name of the SentenceTransformer model to use')
    parser.add_argument('--collection-name', '--collection_name', type=str,
                        default='default_collection',
                        help='Name of the Milvus collection to use')
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers to run in parallel the Milvus server')
    parser.add_argument('-o', '--output-file', '--output_file', type=str,
                        default='search_results.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--server-only', '--server_only', '-S', action='store_true',
                        help='Start server and keep it running indefinitely')
    parser.add_argument('-b', '--batch-size', '--batch_size', type=int, default=256,
                        help='Batch size for encoding queries')

    args = parser.parse_args()
    return args
    # process_queries(args)




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
    from docuverse.utils import read_config_file
    cfg = read_config_file(config)
    if 'retriever' in cfg:
        cfg = cfg.get('retriever')
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
    args = read_args()
    """Run Milvus server and process queries from TSV file."""
    tm = timer("Milvus File Server Test")
    server_info = read_configuration(args.collection_name, args.config, args.milvus_dir, args.model_name)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tm.add_timing("init::milvus_server_init")
    if args.server_only:
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
        queries_df = pd.read_csv(args.query_file, sep='\t')
        queries_text = queries_df['query'].tolist()
        queries = [l for l in model.encode(queries_text, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True)]
        tm.add_timing("encode::sentence_transformer_encode")
        if args.workers > 1:
            from multiprocessing import Pool
            server = create_milvus_server(db_path=server_info.milvus_file_name, use_api=True, port=8765)
            with Pool(args.workers) as p:
                num_q = sum(p.starmap(run_queries, [(i, f"{args.output_file}_{i}", queries_text, queries, server_info, tm) for i in range(args.workers)]))
            server.close()
        else:
            num_q = run_queries(0, args.output_file, queries_text, queries, server_info, tm)

        tm.display_timing(tm.milliseconds_since_beginning(), keys={'queries': num_q}, sorted_by="%", reverse=True)
        print(f"Results written to {args.output_file}")