#!/usr/bin/env python3
"""
Download HuggingFace datasets and save them in JSONL format.

This script downloads a dataset from HuggingFace Hub and saves each split
(train, dev, test, validation, etc.) as a separate JSONL file.

Usage:
    python download_hf_dataset_to_jsonl.py <dataset_name> [--output_dir DIR] [--splits SPLITS]

Examples:
    # Download all available splits
    python download_hf_dataset_to_jsonl.py squad

    # Download specific splits
    python download_hf_dataset_to_jsonl.py glue --subset mrpc --splits train,validation

    # Specify output directory
    python download_hf_dataset_to_jsonl.py squad --output_dir data/squad
"""

import argparse
import bz2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import orjson
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
    from huggingface_hub import dataset_info
except ImportError:
    print("Error: 'datasets' and 'huggingface_hub' libraries are required. Please install them:")
    print("  pip install datasets huggingface_hub")
    exit(1)


def get_subsets(dataset_name: str) -> List[str]:
    """Fetch available subsets/configs for a dataset from HuggingFace Hub."""
    info = dataset_info(dataset_name)
    subsets = [c["config_name"] for c in info.card_data.get("configs", [])] if info.card_data else []
    if not subsets and info.config_names:
        subsets = info.config_names
    return subsets


def download_subset_worker(
    dataset_name: str,
    subset: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
) -> str:
    """Download a single subset into output_dir/<subset>/. Returns a status message."""
    subset_dir = str(Path(output_dir) / subset)
    try:
        download_and_save_dataset(dataset_name, output_dir=subset_dir, subset=subset, splits=splits)
        return f"  ✓ {subset}"
    except Exception as e:
        return f"  ✗ {subset}: {e}"


def save_split_to_jsonl(dataset, output_path: Path, split_name: str):
    """
    Save a dataset split to a JSONL file.

    Args:
        dataset: HuggingFace dataset split
        output_path: Path to save the JSONL file
        split_name: Name of the split (for logging)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_examples = len(dataset)
    print(f"  Saving {num_examples} examples to {output_path}")

    with bz2.open(output_path, 'wb') as f:
        for example in dataset:
            f.write(orjson.dumps(example) + b'\n')

    print(f"  ✓ Saved {split_name} split ({num_examples} examples)")


def download_and_save_dataset(
    dataset_name: str,
    output_dir: Optional[str] = None,
    subset: Optional[str] = None,
    splits: Optional[List[str]] = None,
):
    """
    Download a HuggingFace dataset and save splits to JSONL files.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        output_dir: Directory to save JSONL files (default: ./data/<dataset_name>)
        subset: Dataset subset/config name (e.g., 'mrpc' for GLUE)
        splits: List of specific splits to download (default: all available)
    """
    # Set default output directory
    if output_dir is None:
        base_name = f"{dataset_name}_{subset}" if subset else dataset_name
        base_name = base_name.replace('/', '_')
        output_dir = f"data/{base_name}"

    output_path = Path(output_dir)

    print(f"\nDownloading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")
    print(f"  Output directory: {output_path}")
    print()

    # Load the dataset
    try:
        if subset:
            dataset = load_dataset(dataset_name, subset, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Get available splits
    available_splits = list(dataset.keys())
    print(f"Available splits: {', '.join(available_splits)}")

    # Determine which splits to save
    if splits:
        splits_to_save = [s for s in splits if s in available_splits]
        if not splits_to_save:
            print(f"Warning: None of the requested splits {splits} are available")
            return
        missing_splits = [s for s in splits if s not in available_splits]
        if missing_splits:
            print(f"Warning: Splits not found: {', '.join(missing_splits)}")
    else:
        splits_to_save = available_splits

    print(f"\nSaving splits: {', '.join(splits_to_save)}")
    print()

    # Common split name mappings
    split_name_map = {
        'validation': 'dev',
        'val': 'dev',
    }

    # Save each split
    for split in splits_to_save:
        # Use mapped name if available (e.g., 'validation' -> 'dev')
        output_name = split_name_map.get(split, split)
        output_file = output_path / f"{output_name}.jsonl.bz2"

        print(f"Processing {split} split:")
        save_split_to_jsonl(dataset[split], output_file, split)
        print()

    print(f"✓ Dataset saved successfully to {output_path}")
    print(f"\nFiles created:")
    for split in splits_to_save:
        output_name = split_name_map.get(split, split)
        output_file = output_path / f"{output_name}.jsonl.bz2"
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  - {output_file} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace datasets and save them in JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all available splits
  python download_hf_dataset_to_jsonl.py squad

  # Download specific splits
  python download_hf_dataset_to_jsonl.py squad --splits train,validation

  # Download with subset (for datasets like GLUE)
  python download_hf_dataset_to_jsonl.py glue --subset mrpc

  # Specify output directory
  python download_hf_dataset_to_jsonl.py squad --output_dir data/squad

  # List available subsets for a dataset
  python download_hf_dataset_to_jsonl.py miracl/miracl --list-subsets

  # Download all subsets in parallel (each in its own subdirectory)
  python download_hf_dataset_to_jsonl.py miracl/miracl --all-subsets --output_dir benchmark/miracl --workers 8

  # Rename validation split to dev
  python download_hf_dataset_to_jsonl.py squad
  # (automatically maps 'validation' -> 'dev.jsonl')

Common datasets:
  - squad, squad_v2
  - glue (with --subset: cola, sst2, mrpc, qqp, mnli, qnli, rte, wnli)
  - super_glue (with --subset: boolq, cb, copa, multirc, record, rte, wic, wsc)
  - natural_questions
  - trivia_qa
  - ms_marco
        """
    )

    parser.add_argument(
        'dataset_name',
        type=str,
        help='Name of the HuggingFace dataset (e.g., squad, glue, natural_questions)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for JSONL files (default: data/<dataset_name>)'
    )

    parser.add_argument(
        '--subset',
        type=str,
        default=None,
        help='Dataset subset/config name (e.g., mrpc for GLUE)'
    )

    parser.add_argument(
        '--splits',
        type=str,
        default=None,
        help='Comma-separated list of splits to download (default: all available). '
             'Common splits: train, validation, test, dev'
    )

    parser.add_argument(
        '--list-subsets',
        action='store_true',
        help='List available subsets/configs for the dataset and exit'
    )

    parser.add_argument(
        '--all-subsets',
        action='store_true',
        help='Download all subsets, each into its own subdirectory under output_dir'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers for --all-subsets (default: 4)'
    )

    args = parser.parse_args()

    # Parse splits argument
    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(',')]

    if args.list_subsets or args.all_subsets:
        try:
            subsets = get_subsets(args.dataset_name)
        except Exception as e:
            print(f"Error fetching dataset info: {e}")
            exit(1)
        if not subsets:
            print(f"No subsets found for {args.dataset_name} (default config only)")
            exit(0)

        if args.list_subsets:
            print(f"Available subsets for {args.dataset_name} ({len(subsets)}):")
            for s in subsets:
                print(f"  {s}")
            exit(0)

        # --all-subsets: download each subset in parallel
        output_dir = args.output_dir or f"data/{args.dataset_name.replace('/', '_')}"
        print(f"\nDownloading all {len(subsets)} subsets of {args.dataset_name}")
        print(f"  Output directory: {output_dir}")
        print(f"  Workers: {args.workers}\n")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_subset_worker, args.dataset_name, s, output_dir, splits): s
                for s in subsets
            }
            for future in as_completed(futures):
                print(future.result())

        print(f"\n✓ All subsets saved to {output_dir}")
        exit(0)

    download_and_save_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        subset=args.subset,
        splits=splits,
    )


if __name__ == '__main__':
    main()
