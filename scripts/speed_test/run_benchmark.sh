#!/usr/bin/env bash
# Run benchmark_embedding_timing.py with sensible defaults.
#
# Usage:
#   ./run_benchmark.sh --local_model_name ibm-granite/granite-embedding-125m-english [options]
#   ./run_benchmark.sh --local_model_name mymodel --fof my_files.txt --batch_sizes 256,512,1024
#
# All benchmark_embedding_timing.py flags are passed through; the defaults below
# are applied for any flag not supplied on the command line.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="${SCRIPT_DIR}/benchmark_embedding_timing.py"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LOCAL_MODEL="ibm-granite/granite-embedding-125m-english"
DEFAULT_FIELD_PATH="text"
DEFAULT_MAX_SAMPLES="32768"
DEFAULT_WARMUP_BATCHES="10"
DEFAULT_MAX_NUM_TOKENS="1024"
DEFAULT_BATCH_SIZES="1024"

# ── parse our own flags so we can derive the output filename ──────────────────
local_model="${DEFAULT_LOCAL_MODEL}"
field_path="${DEFAULT_FIELD_PATH}"
max_samples="${DEFAULT_MAX_SAMPLES}"
warmup_batches="${DEFAULT_WARMUP_BATCHES}"
max_num_tokens="${DEFAULT_MAX_NUM_TOKENS}"
batch_sizes="${DEFAULT_BATCH_SIZES}"
skip_api="--skip_api"
torch_compile="--torch_compile"
trust_remote_code="--trust_remote_code"
output_file=""
extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local_model_name)   local_model="$2";    shift 2 ;;
        --field_path)         field_path="$2";     shift 2 ;;
        --max_samples)        max_samples="$2";    shift 2 ;;
        --warmup_batches)     warmup_batches="$2"; shift 2 ;;
        --max_num_tokens)     max_num_tokens="$2";  shift 2 ;;
        --batch_sizes)        batch_sizes="$2";    shift 2 ;;
        --output_file)        output_file="$2";    shift 2 ;;
        --skip_api)           skip_api="--skip_api"; shift ;;
        --no_skip_api)        skip_api="";           shift ;;
        --torch_compile)      torch_compile="--torch_compile"; shift ;;
        --no_torch_compile)   torch_compile="";               shift ;;
        --trust_remote_code)  trust_remote_code="--trust_remote_code"; shift ;;
        --no_trust_remote_code) trust_remote_code="";                  shift ;;
        *)                    extra_args+=("$1");  shift ;;
    esac
done

# ── derive output filename from model name if not supplied ────────────────────
if [[ -z "${output_file}" ]]; then
    model_slug="${local_model//\//_}"   # replace / with _
    model_slug="${model_slug//-/_}"     # replace - with _
    output_file="latency/benchmark_${model_slug}.json"
fi

# ── assemble and run ──────────────────────────────────────────────────────────
cmd=(
    python "${BENCHMARK_PY}"
    --local_model_name "${local_model}"
    --field_path        "${field_path}"
    --max_samples       "${max_samples}"
    --warmup_batches    "${warmup_batches}"
    --max_num_tokens     "${max_num_tokens}"
    --batch_sizes       "${batch_sizes}"
    --output_file       "${output_file}"
)

[[ -n "${skip_api}"          ]] && cmd+=("${skip_api}")
[[ -n "${torch_compile}"     ]] && cmd+=("${torch_compile}")
[[ -n "${trust_remote_code}" ]] && cmd+=("${trust_remote_code}")
[[ ${#extra_args[@]} -gt 0 ]] && cmd+=("${extra_args[@]}")

echo "Running: ${cmd[*]}"
echo "Output : ${output_file}"
echo ""
exec "${cmd[@]}"
