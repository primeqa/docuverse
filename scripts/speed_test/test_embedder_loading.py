#!/usr/bin/env python3
"""Quick smoke test for GPUEmbedder attention fallback across model types."""

import sys
import torch

sys.path.insert(0, "scripts/speed_test")
from benchmark_embedding_timing import GPUEmbedder

MODELS = [
    ("Snowflake/snowflake-arctic-embed-m-v2.0", True),
    ("models/granite_ml_r2_candidates/MergedML0.8En0.2_KDStage3TempRGranite41CodeReasonTeacher", True),
]

for model_name, trust_remote in MODELS:
    short = model_name.rsplit("/", 1)[-1]
    print(f"\n=== {short} ===")
    try:
        emb = GPUEmbedder(model_name, dtype="bf16", trust_remote_code=trust_remote)
        result = emb.embed(["hello world"])
        print(f"OK: shape={result.shape}")
        del emb
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")