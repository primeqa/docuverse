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
import json
import os
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library not found. Please install it:")
    print("  pip install datasets")
    exit(1)


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

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            # Convert the example to a JSON string and write it
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')

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
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
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
        output_file = output_path / f"{output_name}.jsonl"

        print(f"Processing {split} split:")
        save_split_to_jsonl(dataset[split], output_file, split)
        print()

    print(f"✓ Dataset saved successfully to {output_path}")
    print(f"\nFiles created:")
    for split in splits_to_save:
        output_name = split_name_map.get(split, split)
        output_file = output_path / f"{output_name}.jsonl"
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

    args = parser.parse_args()

    # Parse splits argument
    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(',')]

    download_and_save_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        subset=args.subset,
        splits=splits,
    )


if __name__ == '__main__':
    main()
