#!/usr/bin/env bash
# Run benchmark_embedding_timing.py with sensible defaults.
#
# Usage:
#   ./run_benchmark.sh --local_model_name ibm-granite/granite-embedding-125m-english [options]
#   ./run_benchmark.sh --local_model_name mymodel --fof my_files.txt --batch_sizes 256,512,1024
#   ./run_benchmark.sh --config bench.yaml          # load defaults from YAML
#   ./run_benchmark.sh --config bench.yaml --max_samples 1024   # YAML + CLI overrides
#
# All benchmark_embedding_timing.py flags are passed through; the defaults below
# are applied for any flag not supplied on the command line or in --config YAML.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_yaml_config.sh"
BENCHMARK_PY="${SCRIPT_DIR}/benchmark_embedding_timing.py"

# ── defaults ──────────────────────────────────────────────────────────────────
local_model_name="ibm-granite/granite-embedding-125m-english"
field_path="text"
max_samples="32768"
warmup_batches="10"
max_num_tokens="1024"
batch_sizes="1024"
output_file=""
output_dir="latency"
fof=""
skip_api=true
torch_compile=true
trust_remote_code=true
dry_run=false

# Pre-pass: load YAML defaults if --config provided (CLI flags below override)
config_file=$(find_config_arg "$@")
if [[ -n "$config_file" ]]; then
    eval "$(load_yaml_config "$config_file")" || exit 1
fi

extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)               shift 2 ;;  # already consumed in pre-pass
        --local_model_name)     local_model_name="$2";  shift 2 ;;
        --field_path)           field_path="$2";        shift 2 ;;
        --max_samples)          max_samples="$2";       shift 2 ;;
        --warmup_batches)       warmup_batches="$2";    shift 2 ;;
        --max_num_tokens)       max_num_tokens="$2";    shift 2 ;;
        --batch_sizes)          batch_sizes="$2";       shift 2 ;;
        --output_file)          output_file="$2";       shift 2 ;;
        --output_dir)           output_dir="$2";        shift 2 ;;
        --fof)                  fof="$2";               shift 2 ;;
        --skip_api)             skip_api=true;          shift ;;
        --no_skip_api)          skip_api=false;         shift ;;
        --torch_compile)        torch_compile=true;     shift ;;
        --no_torch_compile)     torch_compile=false;    shift ;;
        --trust_remote_code)    trust_remote_code=true; shift ;;
        --no_trust_remote_code) trust_remote_code=false; shift ;;
        --dry-run)              dry_run=true;           shift ;;
        *)                      extra_args+=("$1");     shift ;;
    esac
done

# ── derive output filename from model name if not supplied ────────────────────
if [[ -z "${output_file}" ]]; then
    model_slug="${local_model_name//\//_}"   # replace / with _
    model_slug="${model_slug//-/_}"          # replace - with _
    output_file="${output_dir}/benchmark_${model_slug}.json"
elif [[ "${output_file}" != /* ]]; then
    output_file="${output_dir}/${output_file}"
fi

# ── assemble and run ──────────────────────────────────────────────────────────
cmd=(
    python "${BENCHMARK_PY}"
    --local_model_name "${local_model_name}"
    --field_path        "${field_path}"
    --max_samples       "${max_samples}"
    --warmup_batches    "${warmup_batches}"
    --max_num_tokens    "${max_num_tokens}"
    --batch_sizes       "${batch_sizes}"
    --output_file       "${output_file}"
)

[[ -n "$fof" ]]                      && cmd+=("--fof" "${fof}")
[[ "$skip_api" == "true" ]]          && cmd+=("--skip_api")
[[ "$torch_compile" == "true" ]]     && cmd+=("--torch_compile")
[[ "$trust_remote_code" == "true" ]] && cmd+=("--trust_remote_code")
[[ ${#extra_args[@]} -gt 0 ]]        && cmd+=("${extra_args[@]}")

echo "Running: ${cmd[*]}"
echo "Output : ${output_file}"
echo ""
[[ "$dry_run" == "true" ]] && exit 0
exec "${cmd[@]}"
