#!/bin/bash
# Example script to run embedding timing benchmark

# Set your RITS API key (required for API benchmarking)
# export RITS_API_KEY="your-api-key-here"

# Example 1: Run with generated sample texts (default)
# This will test both API and GPU with batch sizes: 1, 8, 16, 32
python benchmark_embedding_timing.py \
    --model_name ibm/slate-125m-english-rtrvr-v2 \
    --endpoint https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2 \
    --local_model_name ibm-granite/granite-embedding-125m-english \
    --num_samples 100 \
    --batch_sizes 1,8,16,32

# Example 2: Run with JSONL file (auto-detect text field)
# python benchmark_embedding_timing.py \
#     --input_file data/my_texts.jsonl \
#     --batch_sizes 1,8,16,32

# Example 3: Run with JSONL.bz2 file and custom field path
# python benchmark_embedding_timing.py \
#     --input_file /proj/rh-inf-scaling/aashka/rag_rl/data/evals/nq-fixed/nq-dev-500-fixed.jsonl \
#     --field_path question \
#     --max_samples 200 \
#     --batch_sizes 1,16,32

# Example 4: Run with nested field path
# python benchmark_embedding_timing.py \
#     --input_file data.jsonl \
#     --field_path document.text \
#     --batch_sizes 1,8,16

# Example 5: GPU only with JSONL file
# python benchmark_embedding_timing.py \
#     --input_file data.jsonl \
#     --field_path text \
#     --skip_api \
#     --batch_sizes 1,16,32,64

# Example 6: Extract all documents from array field
# python benchmark_embedding_timing.py \
#     --input_file data.jsonl \
#     --field_path "documents[*].text" \
#     --batch_sizes 1,16,32

# Example 7: Extract only first document from each line
# python benchmark_embedding_timing.py \
#     --input_file data.jsonl \
#     --field_path "documents[0].text" \
#     --batch_sizes 1,16,32

# Example 8: Random sampling from large file
# python benchmark_embedding_timing.py \
#     --input_file large_dataset.jsonl \
#     --field_path text \
#     --num_samples 200 \
#     --random_seed 42 \
#     --batch_sizes 1,16,32

# Example 9: Read first 1000 lines, then randomly sample 200
# python benchmark_embedding_timing.py \
#     --input_file huge_dataset.jsonl \
#     --field_path text \
#     --max_samples 1000 \
#     --num_samples 200 \
#     --batch_sizes 1,16,32

# Example 10: Generate plots with results
# python benchmark_embedding_timing.py \
#     --input_file data.jsonl \
#     --field_path text \
#     --num_samples 200 \
#     --batch_sizes 1,8,16,32 \
#     --plot \
#     --plot_dir my_benchmark_plots \
#     --plot_format png

# Other useful options:
# --skip_api                    # Skip API benchmarking
# --skip_gpu                    # Skip GPU benchmarking
# --device cuda                 # Explicitly set device (cuda/cpu)
# --num_samples 500             # Number of texts to use (generates or randomly samples)
# --random_seed 123             # Seed for random sampling (default: 42)
# --batch_sizes 1,4,8,16,32,64  # Test more batch sizes
# --max_samples 1000            # Limit lines read from input file (sequential)
