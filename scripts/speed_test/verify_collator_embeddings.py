#!/usr/bin/env python3
"""
Verify that GPUEmbedder (padded) and CollatorGPUEmbedder (flash-attn flattened)
produce numerically equivalent embeddings for the same inputs.

Runs the same texts through both embedders and reports per-embedding cosine
similarity and absolute L2 differences so you can judge whether any divergence
is within floating-point noise or a real correctness bug.

Usage:
    # Generated texts, default model:
    python verify_collator_embeddings.py --num_samples 200 --batch_size 16

    # From a JSONL file:
    python verify_collator_embeddings.py \
        --input_file data.jsonl --field_path text \
        --num_samples 500 --batch_size 32

    # Specific model with longer sequences:
    python verify_collator_embeddings.py \
        --local_model_name ibm-granite/granite-embedding-125m-english \
        --input_file corpus.jsonl.bz2 --field_path document.text \
        --max_num_tokens 512 --num_samples 1000 --batch_size 8

    # From a YAML config (same format as run_benchmark.sh):
    python verify_collator_embeddings.py --config speed_configs/en_martin.yaml

    # YAML config with CLI overrides (CLI wins):
    python verify_collator_embeddings.py --config bench.yaml --num_samples 50

    # File-of-files (one path per line, same as benchmark --fof):
    python verify_collator_embeddings.py --fof my_files.txt --field_path text

    # With output file:
    python verify_collator_embeddings.py \
        --input_file data.jsonl --num_samples 200 \
        --output_file verify_results.json
"""

import argparse
import json
import os
import random
import sys
import numpy as np

# Add parent directory so we can import from the benchmark script and docuverse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from benchmark_embedding_timing import (
    GPUEmbedder,
    CollatorGPUEmbedder,
    generate_sample_texts,
    load_texts_from_file,
)
from docuverse.utils.jsonl_utils import read_jsonl_file


# YAML keys whose CLI counterpart is a store_true boolean flag.
# Used so that a YAML value of `true` / `false` is converted to the right type.
_BOOL_ARGS = {"trust_remote_code", "verbose"}

# YAML keys that map to a different argparse dest name.
_KEY_ALIASES = {
    "batch_sizes": "batch_size",   # benchmark YAML uses plural; we accept both
}


def _load_yaml_defaults(config_path: str) -> dict:
    """Load a YAML config file and return a dict of argparse-compatible defaults.

    Only keys whose names match a verify_collator_embeddings argument (or a
    known alias) are kept.  Unknown keys are silently ignored so the same YAML
    files used by run_benchmark.sh can be passed here without errors.
    """
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required for --config support (pip install pyyaml)",
              file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        print(f"ERROR: YAML root in {config_path} must be a mapping (key: value pairs)",
              file=sys.stderr)
        sys.exit(1)

    defaults = {}
    for key, val in cfg.items():
        dest = _KEY_ALIASES.get(key, key)
        if dest in _BOOL_ARGS:
            defaults[dest] = bool(val)
        elif dest == "models":
            # benchmark YAMLs may carry a model list; take the first entry as
            # local_model_name only when local_model_name isn't set explicitly.
            if isinstance(val, list) and val:
                defaults.setdefault("local_model_name", str(val[0]))
        elif dest == "input_file":
            # YAML may give a single string or a list
            if isinstance(val, list):
                defaults[dest] = [str(v) for v in val]
            else:
                defaults[dest] = [str(val)]
        elif val is not None:
            defaults[dest] = val

    return defaults


def _find_config_arg(argv):
    """Return the value of --config from argv without running argparse."""
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between matching rows of a and b."""
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1) / (norm_a.squeeze(1) * norm_b.squeeze(1) + 1e-12)


def compare_embeddings(std_embs: np.ndarray, col_embs: np.ndarray,
                       tol_cos: float, tol_l2: float) -> dict:
    """Compute comparison statistics between two embedding matrices."""
    cos_sims = cosine_similarity_matrix(std_embs, col_embs)
    l2_diffs = np.linalg.norm(std_embs - col_embs, axis=1)
    abs_diffs = np.abs(std_embs - col_embs)

    failures_cos = int(np.sum(cos_sims < tol_cos))
    failures_l2 = int(np.sum(l2_diffs > tol_l2))

    return {
        "n": len(cos_sims),
        "cos_sim": {
            "mean": float(np.mean(cos_sims)),
            "min":  float(np.min(cos_sims)),
            "max":  float(np.max(cos_sims)),
            "std":  float(np.std(cos_sims)),
            "failures_below": failures_cos,
            "threshold": tol_cos,
        },
        "l2_diff": {
            "mean": float(np.mean(l2_diffs)),
            "max":  float(np.max(l2_diffs)),
            "std":  float(np.std(l2_diffs)),
            "failures_above": failures_l2,
            "threshold": tol_l2,
        },
        "abs_diff": {
            "mean": float(np.mean(abs_diffs)),
            "max":  float(np.max(abs_diffs)),
        },
        "passed": failures_cos == 0 and failures_l2 == 0,
    }


def print_stats(stats: dict, batch_label: str = ""):
    prefix = f"  [{batch_label}] " if batch_label else "  "
    cos = stats["cos_sim"]
    l2  = stats["l2_diff"]
    ab  = stats["abs_diff"]
    print(f"{prefix}n={stats['n']}  "
          f"cos_sim: mean={cos['mean']:.6f}  min={cos['min']:.6f}  "
          f"max={cos['max']:.6f}  std={cos['std']:.2e}  "
          f"failures<{cos['threshold']}={cos['failures_below']}")
    print(f"{prefix}         "
          f"l2_diff: mean={l2['mean']:.4e}  max={l2['max']:.4e}  "
          f"std={l2['std']:.2e}  "
          f"failures>{l2['threshold']}={l2['failures_above']}")
    print(f"{prefix}         "
          f"abs_diff: mean={ab['mean']:.4e}  max={ab['max']:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify GPUEmbedder vs CollatorGPUEmbedder produce equivalent embeddings"
    )
    # Config file (must be declared first so --help shows it)
    parser.add_argument("--config", type=str, default=None, metavar="YAML",
                        help="YAML config file with default argument values "
                             "(same format as run_benchmark.sh / speed_configs/*.yaml). "
                             "CLI flags override YAML values.")

    # Model args (mirror benchmark_embedding_timing.py)
    parser.add_argument("--local_model_name", type=str,
                        default="ibm-granite/granite-embedding-125m-english",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detected if not specified)")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Model weight dtype (default: bf16)")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Allow loading custom model code from the Hub")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        choices=["flash_attention_2", "sdpa", "eager", "default"],
                        help="Attention implementation for the standard GPUEmbedder. "
                             "CollatorGPUEmbedder always uses flash_attention_2.")
    parser.add_argument("--max_num_tokens", type=int, default=512,
                        help="Max tokens per text (truncation limit, default: 512)")

    # Data args
    parser.add_argument("--input_file", type=str, nargs="+", default=None,
                        help="JSONL or JSONL.bz2 file(s) to load texts from")
    parser.add_argument("--fof", type=str, default=None,
                        help="File-of-files: one input file path per line "
                             "(same as --fof in benchmark_embedding_timing.py). "
                             "Appended to any --input_file paths.")
    parser.add_argument("--field_path", type=str, default=None,
                        help="Dot-separated path to the text field in JSONL records. "
                             "Auto-detected if not given.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max records to read from file before sampling")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of texts to randomly subsample for verification "
                             "(default: use all loaded texts)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for sampling")

    # Verification args
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size used for both embedders (default: 16)")
    parser.add_argument("--tol_cos", type=float, default=1.0 - 1e-3,
                        help="Minimum acceptable cosine similarity (default: 0.999)")
    parser.add_argument("--tol_l2", type=float, default=1e-2,
                        help="Maximum acceptable L2 distance (default: 0.01)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save detailed results as JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-batch statistics")

    # ── Load YAML defaults before parsing so CLI flags can override them ──
    config_path = _find_config_arg(sys.argv[1:])
    if config_path:
        yaml_defaults = _load_yaml_defaults(config_path)
        # Only override args that the YAML provides AND the user didn't supply
        # explicitly on the CLI.  We detect "explicitly supplied" by checking
        # whether the flag appears in sys.argv (crude but reliable for our
        # known argument set).
        cli_flags = set(sys.argv[1:])
        filtered = {}
        for dest, val in yaml_defaults.items():
            flag = f"--{dest}"
            flag_alt = f"--{dest.replace('_', '-')}"
            if flag not in cli_flags and flag_alt not in cli_flags:
                filtered[dest] = val
        if filtered:
            parser.set_defaults(**filtered)

    args = parser.parse_args()

    # Expand --fof into args.input_file
    if args.fof:
        with open(args.fof) as fof_f:
            fof_paths = [ln.strip() for ln in fof_f
                         if ln.strip() and not ln.startswith("#")]
        args.input_file = (args.input_file or []) + fof_paths

    print("=" * 80)
    print("COLLATOR EMBEDDING VERIFICATION")
    print("=" * 80)
    print(f"  model:          {args.local_model_name}")
    print(f"  dtype:          {args.dtype}")
    print(f"  batch_size:     {args.batch_size}")
    print(f"  num_samples:    {args.num_samples if args.num_samples is not None else 'all'}")
    print(f"  max_num_tokens: {args.max_num_tokens}")
    print(f"  tol_cos:        {args.tol_cos}")
    print(f"  tol_l2:         {args.tol_l2}")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # Load texts                                                           #
    # ------------------------------------------------------------------ #
    if args.input_file:
        all_texts = []
        for f in args.input_file:
            all_texts.extend(
                load_texts_from_file(f, field_path=args.field_path,
                                     max_samples=args.max_samples)
            )
    else:
        n_gen = args.num_samples if args.num_samples is not None else 200
        print(f"\nNo --input_file given — generating {n_gen} sample texts.")
        all_texts = generate_sample_texts(n_gen)

    if args.num_samples is not None and len(all_texts) > args.num_samples:
        random.seed(args.random_seed)
        all_texts = random.sample(all_texts, args.num_samples)
        print(f"  Randomly sampled {len(all_texts)} texts (seed={args.random_seed}).")
    print(f"\nUsing {len(all_texts)} texts for verification.")

    # ------------------------------------------------------------------ #
    # Load standard GPUEmbedder                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 40)
    print("Loading GPUEmbedder (standard padded)...")
    std_embedder = GPUEmbedder(
        model_name=args.local_model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    std_embedder.set_max_num_tokens(args.max_num_tokens)

    # ------------------------------------------------------------------ #
    # Load CollatorGPUEmbedder (same weights, different attn path)         #
    # ------------------------------------------------------------------ #
    print("\n" + "-" * 40)
    print("Loading CollatorGPUEmbedder (flash-attn flattened)...")
    try:
        col_embedder = CollatorGPUEmbedder(
            model_name=args.local_model_name,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        col_embedder.set_max_num_tokens(args.max_num_tokens)
    except Exception as e:
        print(f"\n\033[91m\033[1mFailed to load CollatorGPUEmbedder: {e}\033[0m")
        print("Is flash_attn installed and does the model support flash_attention_2?")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Truncate texts that exceed max_num_tokens (consistent pre-processing)#
    # ------------------------------------------------------------------ #
    _tok = std_embedder.tokenizer
    _orig_max = _tok.model_max_length
    _tok.model_max_length = 10 ** 7
    texts = list(all_texts)
    truncated = 0
    for i, text in enumerate(texts):
        ids = _tok(text, add_special_tokens=False,
                   padding=False, truncation=False)["input_ids"]
        if len(ids) > args.max_num_tokens:
            texts[i] = _tok.decode(ids[:args.max_num_tokens], skip_special_tokens=True)
            truncated += 1
    _tok.model_max_length = _orig_max
    if truncated:
        print(f"\n  Pre-truncated {truncated}/{len(texts)} texts to {args.max_num_tokens} tokens.")

    # ------------------------------------------------------------------ #
    # Run both embedders batch-by-batch and collect embeddings             #
    # ------------------------------------------------------------------ #
    print(f"\n" + "-" * 40)
    print(f"Embedding {len(texts)} texts with batch_size={args.batch_size}...")

    std_all = []
    col_all = []
    batch_stats = []

    batch_ranges = range(0, len(texts), args.batch_size)
    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    for b_idx, start in enumerate(batch_ranges):
        batch = texts[start:start + args.batch_size]

        std_emb = std_embedder.embed(batch)
        col_emb = col_embedder.embed(batch)

        std_all.append(std_emb)
        col_all.append(col_emb)

        if args.verbose:
            bs = compare_embeddings(std_emb, col_emb, args.tol_cos, args.tol_l2)
            print_stats(bs, batch_label=f"batch {b_idx+1}/{total_batches}")
        else:
            # Print a progress dot every 10 batches
            if (b_idx + 1) % 10 == 0 or b_idx == 0 or (b_idx + 1) == total_batches:
                print(f"  batch {b_idx+1}/{total_batches}...", flush=True)

        # Keep per-batch stats for the JSON output
        if args.output_file:
            bs = compare_embeddings(std_emb, col_emb, args.tol_cos, args.tol_l2)
            bs["batch_idx"] = b_idx
            bs["batch_start"] = start
            batch_stats.append(bs)

    std_all = np.vstack(std_all)
    col_all = np.vstack(col_all)

    # ------------------------------------------------------------------ #
    # Global statistics                                                    #
    # ------------------------------------------------------------------ #
    global_stats = compare_embeddings(std_all, col_all, args.tol_cos, args.tol_l2)

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS (all samples)")
    print("=" * 80)
    print_stats(global_stats)

    # Worst-case samples
    cos_sims = cosine_similarity_matrix(std_all, col_all)
    worst_idx = int(np.argmin(cos_sims))
    print(f"\n  Worst-case sample (idx {worst_idx}):")
    print(f"    text preview: {texts[worst_idx][:120]!r}")
    print(f"    cos_sim:      {cos_sims[worst_idx]:.8f}")
    print(f"    l2_diff:      {np.linalg.norm(std_all[worst_idx] - col_all[worst_idx]):.6e}")

    print("\n" + "=" * 80)
    if global_stats["passed"]:
        print("\033[92m\033[1mPASSED — outputs are equivalent within tolerance.\033[0m")
    else:
        cos = global_stats["cos_sim"]
        l2  = global_stats["l2_diff"]
        print("\033[91m\033[1mFAILED — outputs differ beyond tolerance:\033[0m")
        if cos["failures_below"]:
            print(f"  {cos['failures_below']} samples with cos_sim < {cos['threshold']} "
                  f"(min observed: {cos['min']:.6f})")
        if l2["failures_above"]:
            print(f"  {l2['failures_above']} samples with l2_diff > {l2['threshold']} "
                  f"(max observed: {l2['max']:.4e})")
    print("=" * 80)

    # ------------------------------------------------------------------ #
    # JSON output                                                          #
    # ------------------------------------------------------------------ #
    if args.output_file:
        output = {
            "model": args.local_model_name,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "num_samples": len(texts),
            "max_num_tokens": args.max_num_tokens,
            "tol_cos": args.tol_cos,
            "tol_l2": args.tol_l2,
            "global_stats": global_stats,
            "batch_stats": batch_stats,
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_file}")

    return 0 if global_stats["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
