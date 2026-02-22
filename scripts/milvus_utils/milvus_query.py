#!/usr/bin/env python3
"""
Interactive query tool for Milvus Lite database files.

Uses the DocUVerse SearchEngine/MilvusDenseEngine pipeline to load an embedding
model, encode questions, and run vector search against a .db file.

Usage:
    python milvus_query.py <database_file> --model <model_name>
    python milvus_query.py --config <config.yaml>
    python milvus_query.py <database_file> -m <model_name> -k 10
    python milvus_query.py <database_file> -m <model_name> -c <collection>

Examples:
    python milvus_query.py experiments/sap/sap.db \\
        -m ibm-granite/granite-embedding-125m-english

    python milvus_query.py --config experiments/sap/sap_milvus_dense.granite125.flat.file.yaml

    python milvus_query.py experiments/nq_new/nq.db \\
        -m ibm-granite/granite-embedding-30m-english -k 3
"""

import argparse
import sys
from pathlib import Path


def discover_collection(db_file):
    """Open the db briefly to list collections and auto-select if only one."""
    from pymilvus import MilvusClient
    client = MilvusClient(uri=db_file)
    collections = client.list_collections()
    client.close()
    return collections


def build_config(db_file, model_name, collection_name, top_k):
    """Build the minimal config dict for SearchEngine."""
    return {
        "db_engine": "milvus-dense",
        "model_name": model_name,
        "index_name": collection_name,
        "server": f"file:{db_file}",
        "top_k": top_k,
        "store_text_in_index": True,
        "no_cache": True,
    }


def print_results(result, top_k):
    """Pretty-print search results."""
    passages = result.retrieved_passages
    if not passages:
        print("  No results found.\n")
        return

    n = min(len(passages), top_k)
    for i in range(n):
        p = passages[i]
        score = getattr(p, "score", None)
        doc_id = getattr(p, "id", "?")
        title = getattr(p, "title", "")
        text = p.get_text() if hasattr(p, "get_text") else getattr(p, "text", "")

        print(f"  [{i+1}] Score: {score:.4f}  ID: {doc_id}")
        if title:
            print(f"      Title: {title}")
        # Show text preview (first 200 chars)
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"      Text:  {preview}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive query tool for Milvus .db files using DocUVerse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct mode: specify db file and model on command line
  python milvus_query.py experiments/sap/sap.db \\
      -m ibm-granite/granite-embedding-125m-english

  # Config mode: use an existing DocUVerse YAML config
  python milvus_query.py --config experiments/sap/sap_milvus_dense.granite125.flat.file.yaml

  # Config mode with overrides
  python milvus_query.py --config experiments/sap/sap_milvus_dense.granite125.flat.file.yaml -k 3
        """,
    )
    parser.add_argument("db_file", nargs="?", default=None,
                        help="Path to Milvus .db file (not needed with --config)")
    parser.add_argument("--config", default=None,
                        help="Path to a DocUVerse YAML config file")
    parser.add_argument("--model", "-m", default=None,
                        help="Embedding model name (e.g. ibm-granite/granite-embedding-30m-english)")
    parser.add_argument("--collection", "-c", default=None,
                        help="Collection name (auto-detected if only one exists)")
    parser.add_argument("--top_k", "-k", type=int, default=None,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--device", "-d", default=None,
                        help="CUDA device to use (e.g. 0, 1, 'cpu'). Default: all visible GPUs")
    args = parser.parse_args()

    if args.device is not None:
        import os
        if args.device.lower() == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Validate arguments
    if args.config is None and args.db_file is None:
        parser.error("either db_file or --config is required")
    if args.config is None and args.model is None:
        parser.error("--model is required when not using --config")

    from docuverse.engines.search_engine import SearchEngine
    from docuverse.engines.search_queries import SearchQueries
    from docuverse.engines.data_template import default_query_template
    import time

    if args.config:
        # Config file mode: let SearchEngine parse the YAML directly
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 1

        print(f"Loading config: {config_path}")
        print("Initializing search engine (loading model)...")
        t0 = time.time()
        engine = SearchEngine(str(config_path))
        print(f"Initialized in {time.time() - t0:.2f}s")

        # Apply CLI overrides
        if args.top_k is not None:
            engine.config.retriever_config.top_k = args.top_k
        if args.model is not None:
            print(f"Warning: --model ignored when using --config "
                  f"(model from config: {engine.config.retriever_config.model_name})")

        top_k = args.top_k if args.top_k is not None else engine.config.retriever_config.top_k
        collection_name = engine.config.retriever_config.index_name
        model_name = engine.config.retriever_config.model_name

    else:
        # Direct mode: build config from CLI args
        db_path = Path(args.db_file)
        if not db_path.exists():
            print(f"Error: file not found: {args.db_file}", file=sys.stderr)
            return 1

        db_file = str(db_path.resolve())
        top_k = args.top_k if args.top_k is not None else 5

        # Discover or validate collection name
        print(f"Opening database: {db_file}")
        t0 = time.time()
        collections = discover_collection(db_file)
        print(f"Database opened in {time.time() - t0:.2f}s")
        if not collections:
            print("Error: no collections found in database.", file=sys.stderr)
            return 1

        if args.collection:
            if args.collection not in collections:
                print(f"Error: collection '{args.collection}' not found. "
                      f"Available: {collections}", file=sys.stderr)
                return 1
            collection_name = args.collection
        elif len(collections) == 1:
            collection_name = collections[0]
        else:
            print(f"Multiple collections found: {collections}")
            print("Please specify one with --collection / -c", file=sys.stderr)
            return 1

        model_name = args.model

        # Build config and initialize SearchEngine
        print("\nInitializing search engine (loading model)...")
        config = build_config(db_file, model_name, collection_name, top_k)
        t0 = time.time()
        engine = SearchEngine(config)
        print(f"Initialized in {time.time() - t0:.2f}s")

    print(f"\nCollection: {collection_name}")
    print(f"Model: {model_name}")
    print(f"Top-k: {top_k}")
    print("\nReady. Enter a question (or 'quit' to exit):")

    # Interactive query loop
    query_num = 0
    while True:
        try:
            question_text = input("\nQuery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question_text:
            continue
        if question_text.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        query_num += 1
        query = SearchQueries.Query(
            template=default_query_template,
            text=question_text,
            id=f"q{query_num}",
            relevant=[],
        )

        try:
            result = engine.retriever.search(query)
            print()
            print_results(result, top_k)
        except Exception as e:
            print(f"  Search error: {e}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
