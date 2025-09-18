from typing import Any

import click
import json
import pandas as pd
import os
from pathlib import Path

from accelerate.commands.config.default import description
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from docuverse.utils.milvus_server_management import create_milvus_server
from docuverse.utils import read_config_file, get_param


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
@click.option('-o', '--output-file', type=click.Path(file_okay=True, dir_okay=False),
              help='Output JSONL file path', default='search_results.jsonl')
def process_queries(milvus_dir, queries_file, model_name, collection_name, config, output_file):
    """Run Milvus server and process queries from TSV file."""

    cfg = read_config_file(config)['retriever']
    # Milvus does not accept '-' in the name, and docuverse replaces them with '_'
    collection_name = get_param(cfg, 'index_name', collection_name).replace("-","_")
    milvus_dir = get_param(cfg, 'project_dir', milvus_dir)
    milvus_file_name = get_param(cfg, 'server', "")
    model_name = get_param(cfg, 'model_name', model_name)
    if milvus_file_name.find("file:") == -1:
        milvus_file_name = os.path.join(milvus_dir, 'milvus.db')
    else:
        milvus_file_name = milvus_file_name.replace("file:", "")
    attn = get_param(cfg, 'attn_implementation', 'sdpa')
    # Initialize and start Milvus server
    server = create_milvus_server(db_path=milvus_file_name, port=8765, use_api=False)

    model_args = {}
    if attn.find("flash") >= 0:
        model_args: dict[str, Any] = {"attn_implementation": attn}
        import torch
        model_args["torch_dtype"] = torch.bfloat16

    # Initialize the sentence transformer model
    model = SentenceTransformer(model_name, device="cuda",
                                model_kwargs=model_args,
                                trust_remote_code=True)

    try:
        # Read queries from TSV file
        queries_df = pd.read_csv(queries_file, sep='\t')

        if 'query' not in queries_df.columns:
            raise ValueError("TSV file must contain a 'query' column")

        
        # Process each query
        with open(output_file, 'w') as f:
            for query in tqdm(queries_df['query'], desc="Processing queries", leave=False):
                query_embedding = model.encode(query, show_progress_bar=False)
                results = server.search(collection_name=collection_name,
                                        query_vector=query_embedding,
                                        limit=5,
                                        output_fields=['text'])
                result_entry = {
                    'query': query,
                    'results': results
                }
                f.write(json.dumps(result_entry) + '\n')
                # print(f"Query: {query}")
                # print(f"Results: {results}\n")

    finally:
        # Ensure server is stopped
        server.close()


if __name__ == '__main__':
    process_queries()
