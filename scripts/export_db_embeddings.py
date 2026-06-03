#!/usr/bin/env python3
"""Export embeddings from a vector database to a pickle file.

Reads embeddings and document IDs from a Milvus .db file, LanceDB directory,
or FAISS index directory and saves them as a pickle file compatible with the
Matryoshka training pipeline (scripts/matryoshka/).

The database format is auto-detected from the input path.

Output pickle format:
    {
        "embeddings": np.ndarray,   # (N, d) float32
        "ids": List[str],           # document IDs, length N
    }

Usage:
    python scripts/export_db_embeddings.py data.db -o corpus_embeddings.pkl
    python scripts/export_db_embeddings.py /data/lance_db -o corpus_embeddings.pkl
    python scripts/export_db_embeddings.py /data/faiss_out -o corpus_embeddings.pkl

Examples:
    # Milvus .db file (auto-detected)
    python scripts/export_db_embeddings.py my_collection.db -o embeddings.pkl

    # Specific collection in a multi-collection Milvus file
    python scripts/export_db_embeddings.py my_data.db -c my_collection -o embeddings.pkl

    # LanceDB directory
    python scripts/export_db_embeddings.py /data/lancedb_dir -o embeddings.pkl

    # FAISS directory
    python scripts/export_db_embeddings.py /data/faiss_dir -o embeddings.pkl

    # Limit to first 10000 documents
    python scripts/export_db_embeddings.py data.db -o embeddings.pkl --max-rows 10000

    # Custom batch size for large databases
    python scripts/export_db_embeddings.py data.db -o embeddings.pkl --batch-size 5000

    # Include text field alongside embeddings
    python scripts/export_db_embeddings.py data.db -o embeddings.pkl --include-text

    # Dry run: show schema and row count without exporting
    python scripts/export_db_embeddings.py data.db -o embeddings.pkl --dry-run
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# Reuse the reader infrastructure from db2db_copy
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.milvus_utils.db2db_copy import (
    detect_db_type,
    discover_faiss,
    discover_lancedb,
    discover_milvus,
    read_faiss_batches,
    read_lancedb_batches,
    read_milvus_batches,
)
from docuverse.utils.timer import timer


def export_embeddings(args):
    """Main export logic."""
    tm = timer("export_db_embeddings")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    # Auto-detect source format
    db_type = args.fromdb
    if not db_type:
        db_type = detect_db_type(str(input_path))
        if not db_type:
            print(
                f"Error: cannot auto-detect database format for '{args.input}'. "
                f"Use --fromdb to specify (milvus, lancedb, faiss).",
                file=sys.stderr,
            )
            return 1
        print(f"Auto-detected format: {db_type}")

    # Discover source
    print(f"Source: {input_path.resolve()} ({db_type})")

    if db_type == "milvus":
        source, collection, fields_info, total_rows, output_fields, vec_field, vec_dim = (
            discover_milvus(str(input_path), args.collection)
        )
    elif db_type == "lancedb":
        source, collection, fields_info, total_rows, output_fields, vec_field, vec_dim = (
            discover_lancedb(str(input_path), args.collection)
        )
    elif db_type == "faiss":
        faiss_index, faiss_id_map, faiss_meta, collection, fields_info, total_rows, output_fields, vec_field, vec_dim = (
            discover_faiss(str(input_path), args.collection)
        )
    else:
        print(f"Error: unsupported format: {db_type}", file=sys.stderr)
        return 1

    tm.add_timing("discover_source")

    # Find the ID field name
    id_field = "id"
    for f in fields_info:
        if f.get("is_primary") and not f.get("auto_id"):
            id_field = f["name"]
            break

    # Find text field if requested
    text_field = None
    if args.include_text:
        for f in fields_info:
            if f["name"] == "text" and not f["is_vector"]:
                text_field = "text"
                break
        if text_field is None:
            # Try other common names
            for candidate in ("text", "content", "passage", "document", "body"):
                for f in fields_info:
                    if f["name"] == candidate and not f["is_vector"]:
                        text_field = candidate
                        break
                if text_field:
                    break
        if text_field is None:
            print(
                "Warning: --include-text requested but no text field found in schema. "
                "Skipping text.",
                file=sys.stderr,
            )

    # Apply max-rows limit
    effective_rows = total_rows
    if args.max_rows and args.max_rows < total_rows:
        effective_rows = args.max_rows

    # Display plan
    print(f"\nCollection:     {collection}")
    print(f"Total rows:     {total_rows:,}")
    if effective_rows != total_rows:
        print(f"Exporting:      {effective_rows:,} (--max-rows)")
    print(f"Vector field:   {vec_field} (dim={vec_dim})")
    print(f"ID field:       {id_field}")
    if text_field:
        print(f"Text field:     {text_field}")
    print(f"Output:         {args.output}")

    est_bytes = effective_rows * vec_dim * 4
    print(f"Est. file size: {est_bytes / (1024**2):.1f} MB")

    print(f"\nSchema ({len(fields_info)} fields):")
    for f in fields_info:
        marker = ""
        if f.get("is_primary"):
            marker += " [PRIMARY]"
        if f.get("auto_id"):
            marker += " [auto_id]"
        if f["is_vector"]:
            marker += f" [dim={f.get('dim', '?')}]"
        print(f"  {f['name']:<25} {f['type_name']:<18}{marker}")

    if args.dry_run:
        print("\n[dry-run] No output written.")
        if db_type == "milvus":
            source.close()
        return 0

    # Create batch reader
    if db_type == "milvus":
        batch_gen = read_milvus_batches(
            source, collection, output_fields, args.batch_size, total_rows
        )
    elif db_type == "lancedb":
        batch_gen = read_lancedb_batches(
            source, collection, args.batch_size, total_rows
        )
    elif db_type == "faiss":
        batch_gen = read_faiss_batches(
            faiss_index, faiss_id_map, faiss_meta, vec_field,
            args.batch_size, total_rows
        )

    # Read all embeddings and IDs
    all_embeddings = []
    all_ids = []
    all_texts = [] if text_field else None
    exported = 0

    pbar = tqdm(total=effective_rows, desc="Exporting embeddings", unit="rows")
    for raw_batch, batch_len, _ in batch_gen:
        if exported >= effective_rows:
            break

        for row in raw_batch:
            if exported >= effective_rows:
                break

            # Extract embedding vector
            vec = row.get(vec_field)
            if vec is None:
                continue

            if isinstance(vec, list):
                vec = np.array(vec, dtype=np.float32)
            elif isinstance(vec, np.ndarray):
                vec = vec.astype(np.float32)
            else:
                vec = np.array(vec, dtype=np.float32)

            all_embeddings.append(vec)

            # Extract ID
            doc_id = row.get(id_field, str(exported))
            all_ids.append(str(doc_id))

            # Extract text if requested
            if text_field and all_texts is not None:
                all_texts.append(str(row.get(text_field, "")))

            exported += 1

        pbar.update(min(batch_len, effective_rows - (exported - batch_len)))

    pbar.close()

    if db_type == "milvus":
        source.close()

    tm.add_timing("read_batches")

    if not all_embeddings:
        print("Error: no embeddings found in source.", file=sys.stderr)
        return 1

    # Stack into numpy array
    embeddings = np.stack(all_embeddings, axis=0)
    del all_embeddings

    tm.add_timing("stack_embeddings")

    # Build output dict
    output_data = {
        "embeddings": embeddings,
        "ids": all_ids,
    }
    if all_texts is not None:
        output_data["texts"] = all_texts

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"\nSaving to {args.output}...")
    with open(args.output, "wb") as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    tm.add_timing("save_pickle")

    file_size = Path(args.output).stat().st_size
    print(f"\nExport complete:")
    print(f"  Embeddings:  {embeddings.shape[0]:,} x {embeddings.shape[1]} ({embeddings.dtype})")
    print(f"  IDs:         {len(all_ids):,}")
    if all_texts is not None:
        print(f"  Texts:       {len(all_texts):,}")
    print(f"  File size:   {file_size / (1024**2):.1f} MB")

    # Quick sanity check: print norm stats
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Norm stats:  min={norms.min():.4f}  max={norms.max():.4f}  "
          f"mean={norms.mean():.4f}  std={norms.std():.4f}")
    if abs(norms.mean() - 1.0) < 0.05:
        print(f"  Embeddings appear L2-normalized.")
    else:
        print(f"  Embeddings are NOT L2-normalized (mean norm={norms.mean():.4f}).")
        if args.normalize:
            print(f"  Normalizing embeddings (--normalize)...")
            norms_safe = np.maximum(norms, 1e-8)[:, np.newaxis]
            embeddings = embeddings / norms_safe
            output_data["embeddings"] = embeddings
            with open(args.output, "wb") as f:
                pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            tm.add_timing("normalize_and_resave")
            print(f"  Re-saved with normalized embeddings.")

    tm.add_timing("verify_norms")

    # Display timing summary
    total_ms = tm.milliseconds_since_beginning()
    print()
    timer.display_timing(
        totalms=total_ms,
        keys={"rows": exported, "dims": exported * vec_dim},
    )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Export embeddings from a vector database to a pickle file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Milvus .db file
  python scripts/export_db_embeddings.py data.db -o corpus_embeddings.pkl

  # LanceDB directory
  python scripts/export_db_embeddings.py /data/lance_db -o corpus_embeddings.pkl

  # FAISS directory
  python scripts/export_db_embeddings.py /data/faiss_dir -o corpus_embeddings.pkl

  # Limit export and include text
  python scripts/export_db_embeddings.py data.db -o embs.pkl --max-rows 50000 --include-text

  # Use with matryoshka training
  python scripts/export_db_embeddings.py data.db -o embs.pkl
  python -m scripts.matryoshka.train --method adaptor --embeddings_cache embs.pkl
""",
    )

    parser.add_argument(
        "input",
        help="Source database path (Milvus .db file, LanceDB dir, or FAISS dir)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--fromdb",
        default=None,
        choices=["milvus", "lancedb", "faiss"],
        help="Source format (auto-detected if omitted)",
    )
    parser.add_argument(
        "--collection", "-c",
        default=None,
        help="Collection/table name (auto-detected if only one exists)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=2000,
        help="Rows per read batch (default: 2000)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to export (default: all)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings if they aren't already",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include document text in the pickle (adds 'texts' key)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show schema and row count without exporting",
    )

    args = parser.parse_args()
    return export_embeddings(args)


if __name__ == "__main__":
    sys.exit(main())
