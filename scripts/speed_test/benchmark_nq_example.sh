#!/bin/bash
# Example: Benchmark embedding generation using the NQ dev-500 dataset

# Ensure RITS API key is set
if [ -z "$RITS_API_KEY" ]; then
    echo "Error: RITS_API_KEY environment variable not set"
    echo "Please set it with: export RITS_API_KEY='your-key-here'"
    exit 1
fi

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gma_rag_rl

# Path to NQ dataset (adjust if needed)
NQ_FILE="/proj/rh-inf-scaling/aashka/rag_rl/data/evals/nq-fixed/nq-dev-500-fixed.jsonl"

# Check if file exists
if [ ! -f "$NQ_FILE" ]; then
    echo "Warning: NQ file not found at $NQ_FILE"
    echo "Please update the NQ_FILE variable in this script"
    echo ""
    echo "Falling back to generated sample texts..."

    # Run with generated texts instead
    python benchmark_embedding_timing.py \
        --num_samples 100 \
        --batch_sizes 1,8,16,32

    exit 0
fi

# Run benchmark with NQ questions
echo "Benchmarking with NQ dev-500 questions..."
echo "==========================================="
echo ""

python benchmark_embedding_timing.py \
    --input_file "$NQ_FILE" \
    --field_path question \
    --max_samples 200 \
    --batch_sizes 1,8,16,32 \
    --model_name ibm/slate-125m-english-rtrvr-v2 \
    --endpoint https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/slate-125m-english-rtrvr-v2 \
    --local_model_name ibm-granite/granite-embedding-125m-english

echo ""
echo "Benchmark complete!"
