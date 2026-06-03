#!/usr/bin/env python3
"""
Prune a SentenceTransformer model's vocabulary based on token frequency statistics.

Given a frequency count file (from token_stats_from_jsonl.py or
interpolate_token_stats.py) and a SentenceTransformer model directory, this
script:

  1. Selects the most frequent tokens to keep (via --keep or --remove)
  2. Rebuilds the tokenizer with a pruned vocabulary and filtered BPE merges
  3. Resizes the model's embedding matrix, preserving the token→vector mapping
  4. Writes the pruned model to --output_dir

Special/added tokens are always preserved regardless of frequency.

Usage:
    python prune_vocabulary.py \\
        --model /path/to/sentence-transformer \\
        --counts /path/to/token_stats.json \\
        --keep 100000 \\
        --output_dir /path/to/pruned-model

    python prune_vocabulary.py \\
        --model /path/to/sentence-transformer \\
        --counts /path/to/token_stats.json \\
        --remove 50000 \\
        --output_dir /path/to/pruned-model
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

try:
    from docuverse.utils import save_command_line
except ImportError:
    save_command_line = None

import torch


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_counts(path: str) -> list[dict]:
    """Load token frequency histogram from JSON.

    Supports both raw count files (``count`` field) and interpolated files
    (``interpolated_prob`` field).  Returns the histogram list sorted by
    frequency descending.
    """
    import orjson

    data = orjson.loads(Path(path).read_bytes())

    if "histogram" in data:
        histogram = data["histogram"]
    else:
        for key in data:
            if key.endswith("_most_frequent"):
                histogram = data[key]
                break
        else:
            raise KeyError(f"No 'histogram' or '*_most_frequent' key found in {path}")

    # Ensure descending sort regardless of input ordering
    if histogram and "count" in histogram[0]:
        histogram.sort(key=lambda e: e["count"], reverse=True)
    elif histogram and "interpolated_prob" in histogram[0]:
        histogram.sort(key=lambda e: float(e["interpolated_prob"]), reverse=True)

    return histogram


# ---------------------------------------------------------------------------
# Token selection
# ---------------------------------------------------------------------------

def determine_tokens_to_keep(
    histogram: list[dict],
    regular_vocab_ids: set[int],
    added_token_ids: set[int],
    keep_count: int | None,
    remove_count: int | None,
    orig_vocab_size: int,
) -> set[int]:
    """Return the set of token IDs to keep.

    *keep_count* / *remove_count* refer to the **total** vocabulary size
    (regular + added tokens).  Added/special tokens are always preserved.
    """
    num_added = len(added_token_ids)
    num_regular = len(regular_vocab_ids)

    target_total = keep_count if keep_count is not None else orig_vocab_size - remove_count

    target_regular = target_total - num_added
    if target_regular < 0:
        sys.exit(
            f"ERROR: target vocab size ({target_total}) is smaller than the "
            f"number of added/special tokens ({num_added})"
        )
    if target_regular > num_regular:
        sys.exit(
            f"ERROR: target would keep {target_regular} regular tokens but "
            f"only {num_regular} exist"
        )
    if target_regular == num_regular:
        print("  WARNING: target keeps all regular tokens — nothing to prune")

    # Rank regular tokens by frequency (histogram is already sorted descending)
    ranked_ids: list[int] = []
    seen: set[int] = set()
    for entry in histogram:
        tid = entry["token_id"]
        if tid in regular_vocab_ids and tid not in seen:
            ranked_ids.append(tid)
            seen.add(tid)

    # Tokens present in vocab but absent from histogram are lowest-priority
    missing = regular_vocab_ids - seen
    if missing:
        print(f"  {len(missing):,} regular tokens absent from count file (removed first)")
    ranked_ids.extend(sorted(missing))

    kept_regular = set(ranked_ids[:target_regular])
    return kept_regular | added_token_ids


# ---------------------------------------------------------------------------
# ID remapping
# ---------------------------------------------------------------------------

def build_id_mapping(
    kept_ids: set[int],
    added_token_ids: set[int],
) -> dict[int, int]:
    """Build an old_id → new_id mapping.

    Regular tokens get contiguous IDs ``0 .. K-1`` (sorted by original ID).
    Added tokens follow at ``K .. K+A-1`` (sorted by original ID).
    """
    kept_regular = sorted(tid for tid in kept_ids if tid not in added_token_ids)
    kept_added = sorted(tid for tid in kept_ids if tid in added_token_ids)

    old_to_new: dict[int, int] = {}
    for new_id, old_id in enumerate(kept_regular):
        old_to_new[old_id] = new_id

    offset = len(kept_regular)
    for i, old_id in enumerate(kept_added):
        old_to_new[old_id] = offset + i

    return old_to_new


# ---------------------------------------------------------------------------
# Tokenizer pruning
# ---------------------------------------------------------------------------

def prune_tokenizer_json(
    tokenizer_data: dict,
    kept_ids: set[int],
    old_to_new: dict[int, int],
) -> dict:
    """Return a pruned deep-copy of *tokenizer_data*.

    Updates ``model.vocab``, ``model.merges``, and ``added_tokens``.
    """
    pruned = copy.deepcopy(tokenizer_data)
    model = pruned["model"]
    old_vocab: dict[str, int] = model["vocab"]

    # Build set of kept token strings (for merge filtering)
    id_to_str = {v: k for k, v in old_vocab.items()}
    kept_strs: set[str] = {id_to_str[tid] for tid in kept_ids if tid in id_to_str}
    for tok in pruned.get("added_tokens", []):
        if tok["id"] in kept_ids:
            kept_strs.add(tok["content"])

    # 1. Prune vocab
    new_vocab = {
        tok_str: old_to_new[old_id]
        for tok_str, old_id in old_vocab.items()
        if old_id in old_to_new
    }
    model["vocab"] = new_vocab
    print(f"  Vocab: {len(old_vocab):,} → {len(new_vocab):,}")

    # 2. Prune BPE merges — keep only merges whose inputs AND result are all
    #    in the kept vocabulary
    old_merges = model.get("merges", [])
    new_merges = [
        merge for merge in old_merges
        if merge[0] in kept_strs
        and merge[1] in kept_strs
        and (merge[0] + merge[1]) in kept_strs
    ]
    model["merges"] = new_merges
    print(f"  Merges: {len(old_merges):,} → {len(new_merges):,}")

    # 3. Update added_tokens IDs
    for tok in pruned.get("added_tokens", []):
        if tok["id"] in old_to_new:
            tok["id"] = old_to_new[tok["id"]]

    # 4. Remap token IDs embedded in post_processor and padding sections
    _remap_tokenizer_ids(pruned, old_to_new)

    return pruned


def _remap_tokenizer_ids(tokenizer_data: dict, old_to_new: dict[int, int]):
    """Remap hardcoded token IDs in post_processor and padding sections."""
    # post_processor → special_tokens → <name> → ids: [old_id, ...]
    post_proc = tokenizer_data.get("post_processor", {})
    for _name, spec in post_proc.get("special_tokens", {}).items():
        if "ids" in spec:
            spec["ids"] = [
                old_to_new.get(tid, tid) for tid in spec["ids"]
            ]

    # padding → pad_id
    padding = tokenizer_data.get("padding")
    if padding and "pad_id" in padding:
        old_pad = padding["pad_id"]
        if old_pad in old_to_new:
            padding["pad_id"] = old_to_new[old_pad]


def update_tokenizer_config(
    tok_config: dict,
    old_to_new: dict[int, int],
) -> dict:
    """Return a copy of tokenizer_config.json with remapped added_tokens_decoder."""
    tok_config = copy.deepcopy(tok_config)
    old_decoder = tok_config.get("added_tokens_decoder")
    if old_decoder:
        new_decoder = {}
        for old_id_str, info in old_decoder.items():
            old_id = int(old_id_str)
            if old_id in old_to_new:
                new_decoder[str(old_to_new[old_id])] = info
        tok_config["added_tokens_decoder"] = new_decoder
    return tok_config


# ---------------------------------------------------------------------------
# Model-weight pruning
# ---------------------------------------------------------------------------

def prune_model_weights(
    model_dir: Path,
    output_dir: Path,
    old_to_new: dict[int, int],
    new_vocab_size: int,
    orig_vocab_size: int,
):
    """Load model weights, resize all vocab-dimension tensors, and save."""
    safetensors_path = model_dir / "model.safetensors"
    pytorch_path = model_dir / "pytorch_model.bin"

    metadata = None
    if safetensors_path.exists():
        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="pt") as f:
            metadata = f.metadata()
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        save_format = "safetensors"
    elif pytorch_path.exists():
        state_dict = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        save_format = "pytorch"
    else:
        sys.exit(f"ERROR: no model.safetensors or pytorch_model.bin in {model_dir}")

    # Build the old→new index tensor once (for efficient advanced indexing)
    # new_indices[new_id] = old_id
    new_indices = torch.empty(new_vocab_size, dtype=torch.long)
    for old_id, new_id in old_to_new.items():
        new_indices[new_id] = old_id

    # Resize every tensor that has a dimension matching orig_vocab_size
    resized = 0
    for key in list(state_dict.keys()):
        tensor = state_dict[key]
        for dim_idx in range(tensor.dim()):
            if tensor.shape[dim_idx] == orig_vocab_size:
                state_dict[key] = torch.index_select(tensor, dim_idx, new_indices)
                print(f"  {key}: {list(tensor.shape)} → {list(state_dict[key].shape)}")
                resized += 1
                break  # at most one vocab dimension per tensor

    if resized == 0:
        print("  WARNING: no tensors with vocab dimension found — weights unchanged")

    if save_format == "safetensors":
        from safetensors.torch import save_file
        save_file(state_dict, str(output_dir / "model.safetensors"), metadata=metadata)
    else:
        torch.save(state_dict, str(output_dir / "pytorch_model.bin"))


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under *path* (recursive, follows symlinks)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _count_params(model_path: Path) -> int:
    """Count total parameters in a safetensors or pytorch_model.bin file."""
    safetensors_path = model_path / "model.safetensors"
    pytorch_path = model_path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors import safe_open
        total = 0
        with safe_open(str(safetensors_path), framework="pt") as f:
            for k in f.keys():
                total += f.get_tensor(k).numel()
        return total
    elif pytorch_path.exists():
        state_dict = torch.load(str(pytorch_path), map_location="cpu", weights_only=True)
        return sum(t.numel() for t in state_dict.values())
    return 0


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_output(output_dir: Path, test_text: str = "Hello world"):
    """Quick sanity check: load the pruned tokenizer and do a round-trip."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(output_dir))
    ids = tok.encode(test_text, add_special_tokens=True)
    decoded = tok.decode(ids, skip_special_tokens=True)
    print(f"  Encode '{test_text}' → {ids}")
    print(f"  Decode → '{decoded}'")
    print(f"  Tokenizer vocab size: {tok.vocab_size}")

    with open(output_dir / "config.json") as f:
        cfg = json.load(f)
    print(f"  Config vocab_size:    {cfg['vocab_size']}")

    if tok.vocab_size + len(tok.added_tokens_encoder) != cfg["vocab_size"]:
        print("  WARNING: tokenizer total size does not match config vocab_size")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the SentenceTransformer model directory",
    )
    parser.add_argument(
        "--counts", required=True,
        help="Path to a token-frequency JSON file (histogram with count or "
             "interpolated_prob fields)",
    )
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--keep", type=int,
        help="Target total vocabulary size (regular + special tokens)",
    )
    size_group.add_argument(
        "--remove", type=int,
        help="Number of tokens to remove from the vocabulary",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write the pruned model into (must not exist unless "
             "--force is given)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite --output_dir if it already exists",
    )
    args = parser.parse_args()
    if save_command_line is not None:
        save_command_line(sys.argv)

    model_dir = Path(args.model)
    output_dir = Path(args.output_dir)

    if not model_dir.is_dir():
        sys.exit(f"ERROR: {model_dir} is not a directory")
    if output_dir.exists() and not args.force:
        sys.exit(f"ERROR: {output_dir} already exists (use --force to overwrite)")

    # ------------------------------------------------------------------
    # Load model metadata
    # ------------------------------------------------------------------
    print(f"Model: {model_dir}")
    with open(model_dir / "tokenizer.json") as f:
        tokenizer_data = json.load(f)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    orig_vocab_size = config["vocab_size"]
    regular_vocab: dict[str, int] = tokenizer_data["model"]["vocab"]
    added_tokens: list[dict] = tokenizer_data.get("added_tokens", [])
    added_token_ids: set[int] = {t["id"] for t in added_tokens}
    regular_vocab_ids: set[int] = set(regular_vocab.values())

    print(
        f"  Original vocab: {orig_vocab_size:,} "
        f"({len(regular_vocab):,} regular + {len(added_tokens)} added/special)"
    )

    # ------------------------------------------------------------------
    # Load frequency counts
    # ------------------------------------------------------------------
    print(f"Counts: {args.counts}")
    histogram = load_counts(args.counts)
    print(f"  Histogram entries: {len(histogram):,}")

    # ------------------------------------------------------------------
    # Select tokens to keep
    # ------------------------------------------------------------------
    print("Selecting tokens…")
    kept_ids = determine_tokens_to_keep(
        histogram,
        regular_vocab_ids,
        added_token_ids,
        keep_count=args.keep,
        remove_count=args.remove,
        orig_vocab_size=orig_vocab_size,
    )
    new_vocab_size = len(kept_ids)
    kept_regular = new_vocab_size - len(added_token_ids)
    removed = orig_vocab_size - new_vocab_size
    print(
        f"  Keeping {new_vocab_size:,} "
        f"({kept_regular:,} regular + {len(added_token_ids)} added), "
        f"removing {removed:,}"
    )

    old_to_new = build_id_mapping(kept_ids, added_token_ids)

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    # Prune tokenizer
    # ------------------------------------------------------------------
    print("Pruning tokenizer…")
    pruned_tokenizer = prune_tokenizer_json(tokenizer_data, kept_ids, old_to_new)
    with open(output_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(pruned_tokenizer, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Update config.json
    # ------------------------------------------------------------------
    config["vocab_size"] = new_vocab_size
    # Remap special token IDs referenced in config
    _TOKEN_ID_KEYS = (
        "bos_token_id", "eos_token_id", "pad_token_id",
        "cls_token_id", "sep_token_id", "mask_token_id",
        "decoder_start_token_id",
    )
    for key in _TOKEN_ID_KEYS:
        if key in config and config[key] is not None:
            old_id = config[key]
            if old_id in old_to_new:
                config[key] = old_to_new[old_id]
            else:
                print(f"  WARNING: {key}={old_id} not in kept tokens")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config.json: vocab_size → {new_vocab_size:,}")

    # ------------------------------------------------------------------
    # Update tokenizer_config.json
    # ------------------------------------------------------------------
    tok_config_path = model_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        with open(tok_config_path) as f:
            tok_config = json.load(f)
        tok_config = update_tokenizer_config(tok_config, old_to_new)
        with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tok_config, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Prune model weights
    # ------------------------------------------------------------------
    print("Pruning model weights…")
    prune_model_weights(
        model_dir, output_dir, old_to_new, new_vocab_size, orig_vocab_size,
    )

    # ------------------------------------------------------------------
    # Copy remaining files unchanged
    # ------------------------------------------------------------------
    print("Copying remaining files…")
    skip = {
        "model.safetensors", "pytorch_model.bin",
        "tokenizer.json", "config.json", "tokenizer_config.json",
    }
    for item in sorted(model_dir.iterdir()):
        if item.name in skip:
            continue
        dest = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    print("Verifying…")
    verify_output(output_dir)

    # ------------------------------------------------------------------
    # Report sizes
    # ------------------------------------------------------------------
    old_size = _dir_size_bytes(model_dir)
    new_size = _dir_size_bytes(output_dir)
    old_params = _count_params(model_dir)
    new_params = _count_params(output_dir)
    print(f"\nDone. Pruned model written to {output_dir}")
    print(f"  Vocab:      {orig_vocab_size:,} → {new_vocab_size:,} tokens ({removed:,} removed)")
    print(f"  Params:     {_fmt_params(old_params)} → {_fmt_params(new_params)} ({_fmt_params(old_params - new_params)} removed)")
    print(f"  Disk size:  {_fmt_size(old_size)} → {_fmt_size(new_size)} ({_fmt_size(old_size - new_size)} saved)")


if __name__ == "__main__":
    main()
