#!/usr/bin/env python3
"""
Interactive query tool for Milvus Lite and LanceDB database files.

Uses the DocUVerse SearchEngine pipeline to load an embedding model, encode
questions, and run vector search against a Milvus .db file or LanceDB directory.

Usage:
    python milvus_query.py <database_file> --model <model_name>
    python milvus_query.py --config <config.yaml>
    python milvus_query.py <database_file> -m <model_name> -k 10
    python milvus_query.py <lancedb_dir> -m <model_name> --engine lancedb

Examples:
    # Milvus .db file
    python milvus_query.py experiments/sap/sap.db \\
        -m ibm-granite/granite-embedding-125m-english

    # LanceDB directory
    python milvus_query.py /data/wiki_lance \\
        -m ibm-granite/granite-embedding-30m-english --engine lancedb

    # Config file mode
    python milvus_query.py --config experiments/sap/sap_milvus_dense.granite125.flat.file.yaml
"""

import argparse
import sys
from pathlib import Path


def discover_collection(client, collection_name=None):
    """Discover or validate the collection name using an existing client."""
    collections = client.list_collections()
    if not collections:
        print("Error: no collections found in database.", file=sys.stderr)
        sys.exit(1)

    if collection_name:
        if collection_name not in collections:
            print(f"Error: collection '{collection_name}' not found. "
                  f"Available: {collections}", file=sys.stderr)
            sys.exit(1)
        return collection_name
    elif len(collections) == 1:
        return collections[0]
    else:
        print(f"Multiple collections found: {collections}")
        print("Please specify one with --collection / -c", file=sys.stderr)
        sys.exit(1)


def detect_engine(path):
    """Detect whether a path is a Milvus .db file or a LanceDB directory."""
    p = Path(path)
    if p.is_file() and p.suffix == ".db":
        return "milvus"
    if p.is_dir():
        # LanceDB directories contain .lance files or _versions/
        lance_markers = list(p.glob("**/*.lance")) + list(p.glob("**/_versions"))
        if lance_markers:
            return "lancedb"
        # Also check if the path itself is a lancedb_data dir or its parent
        if p.name == "lancedb_data" or (p / "lancedb_data").is_dir():
            return "lancedb"
    return None


def discover_lancedb_table(db, table_name=None):
    """Discover or validate a table name in a LanceDB database."""
    try:
        resp = db.list_tables()
        tables = resp.tables if hasattr(resp, 'tables') else list(resp)
    except AttributeError:
        tables = list(db.table_names())

    if not tables:
        print("Error: no tables found in LanceDB database.", file=sys.stderr)
        sys.exit(1)

    if table_name:
        if table_name not in tables:
            print(f"Error: table '{table_name}' not found. "
                  f"Available: {tables}", file=sys.stderr)
            sys.exit(1)
        return table_name
    elif len(tables) == 1:
        return tables[0]
    else:
        print(f"Multiple tables found: {tables}")
        print("Please specify one with --collection / -c", file=sys.stderr)
        sys.exit(1)


def build_config(db_file, model_name, top_k, engine_type="milvus"):
    """Build the minimal config dict for SearchEngine.

    Uses a placeholder index_name; the real collection/table is discovered
    after the engine connects.
    """
    if engine_type == "lancedb":
        # LanceDBEngine by default connects to {project_dir}/lancedb_data.
        # If the user points to a directory that already has lancedb_data/
        # inside it, use project_dir directly. Otherwise, the path itself
        # is the LanceDB directory — pass it via 'server' to override.
        db_path = Path(db_file)
        config = {
            "db_engine": "lancedb",
            "model_name": model_name,
            "index_name": "_placeholder_",
            "top_k": top_k,
            "store_text_in_index": True,
            "no_cache": True,
        }
        if db_path.name == "lancedb_data":
            config["project_dir"] = str(db_path.parent)
        elif (db_path / "lancedb_data").is_dir():
            config["project_dir"] = str(db_path)
        else:
            # Path is the LanceDB directory itself — use direct path override
            config["project_dir"] = str(db_path.parent)
            config["server"] = str(db_path)
        return config
    else:
        return {
            "db_engine": "milvus-dense",
            "model_name": model_name,
            "index_name": "_placeholder_",
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
        description="Interactive query tool for Milvus/LanceDB databases using DocUVerse.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Milvus .db file (auto-detected)
  python milvus_query.py experiments/sap/sap.db \\
      -m ibm-granite/granite-embedding-125m-english

  # LanceDB directory (auto-detected or use --engine)
  python milvus_query.py /data/wiki_lance \\
      -m ibm-granite/granite-embedding-30m-english

  # Explicit engine type
  python milvus_query.py /data/wiki_lance \\
      -m ibm-granite/granite-embedding-30m-english --engine lancedb

  # Config mode: use an existing DocUVerse YAML config
  python milvus_query.py --config experiments/sap/sap_milvus_dense.granite125.flat.file.yaml
        """,
    )
    parser.add_argument("db_file", nargs="?", default=None,
                        help="Path to Milvus .db file or LanceDB directory (not needed with --config)")
    parser.add_argument("--config", default=None,
                        help="Path to a DocUVerse YAML config file")
    parser.add_argument("--model", "-m", default=None,
                        help="Embedding model name (e.g. ibm-granite/granite-embedding-30m-english)")
    parser.add_argument("--collection", "-c", default=None,
                        help="Collection/table name (auto-detected if only one exists)")
    parser.add_argument("--top_k", "-k", type=int, default=None,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--device", "-d", default=None,
                        help="CUDA device to use (e.g. 0, 1, 'cpu'). Default: all visible GPUs")
    parser.add_argument("--engine", "-e", default=None,
                        choices=["milvus", "lancedb"],
                        help="Engine type (auto-detected from path if not specified)")
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
        # Config file mode: read the YAML config
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 1

        print(f"Loading config: {config_path}")

        # Read and fix the config before passing to SearchEngine.
        # Some configs have a bare path for 'server' without the required
        # 'file:' prefix -- detect and fix this automatically.
        from docuverse.utils import read_config_file
        config_dict = read_config_file(str(config_path))
        retriever = config_dict.get("retriever", config_dict)
        server_val = retriever.get("server", "")
        if server_val and "file:" not in server_val and server_val.endswith(".db"):
            retriever["server"] = f"file:{server_val}"
            print(f"  (added 'file:' prefix to server path: {retriever['server']})")

        print("Initializing search engine (loading model)...")
        t0 = time.time()
        engine = SearchEngine(config_dict)
        print(f"Initialized in {time.time() - t0:.2f}s")

        # Verify the client connected successfully (Milvus only)
        if hasattr(engine.retriever, 'client') and engine.retriever.client is None:
            print("Error: failed to connect to Milvus database. "
                  "Check the 'server' field in your config.", file=sys.stderr)
            return 1

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
            print(f"Error: path not found: {args.db_file}", file=sys.stderr)
            return 1

        db_file = str(db_path.resolve())
        top_k = args.top_k if args.top_k is not None else 5
        model_name = args.model

        # Detect engine type
        engine_type = args.engine
        if engine_type is None:
            engine_type = detect_engine(db_file)
            if engine_type is None:
                parser.error(f"Cannot auto-detect engine type for '{args.db_file}'. "
                             f"Use --engine milvus or --engine lancedb.")
            print(f"Auto-detected engine: {engine_type}")

        # Initialize SearchEngine (loads model + connects to db)
        print(f"Opening database: {db_file}")
        print("Initializing search engine (loading model)...")
        config = build_config(db_file, model_name, top_k, engine_type=engine_type)
        t0 = time.time()
        engine = SearchEngine(config)
        print(f"Initialized in {time.time() - t0:.2f}s")

        # Discover/validate collection or table name
        if engine_type == "lancedb":
            collection_name = discover_lancedb_table(
                engine.retriever.db, args.collection
            )
        else:
            collection_name = discover_collection(
                engine.retriever.client, args.collection
            )
        engine.config.retriever_config.index_name = collection_name
        engine.retriever.config.index_name = collection_name

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
