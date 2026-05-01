#!/usr/bin/env bash
# Run benchmark_embedding_timing.py with sensible defaults.
#
# Usage:
#   ./run_benchmark.sh --local_model_name ibm-granite/granite-embedding-125m-english [options]
#   ./run_benchmark.sh --local_model_name mymodel --fof my_files.txt --batch_sizes 256,512,1024
#   ./run_benchmark.sh --models_file new_models.dat            # run each model in file
#   ./run_benchmark.sh --config bench.yaml                     # load defaults from YAML
#   ./run_benchmark.sh --config bench.yaml --max_samples 1024  # YAML + CLI overrides
#   ./run_benchmark.sh --config bench.yaml                     # if YAML has "models:" list,
#                                                              # runs each model in that list
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
models_file=""
models=()   # may be populated by YAML "models:" list
skip_api=true
torch_compile=true
trust_remote_code=true
dry_run=false
force=false

# Pre-pass: load YAML defaults if --config provided (CLI flags below override)
config_file=$(find_config_arg "$@")
if [[ -n "$config_file" ]]; then
    eval "$(load_yaml_config "$config_file")" || exit 1
fi

echo $field_path

extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)               shift 2 ;;  # already consumed in pre-pass
        --local_model_name)     local_model_name="$2";  shift 2 ;;
        --models_file)          models_file="$2";       shift 2 ;;
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
        --force)                force=true;             shift ;;
        *)                      extra_args+=("$1");     shift ;;
    esac
done

# ── collect model list ────────────────────────────────────────────────────────
# Priority: --models_file CLI > "models:" array from YAML > single --local_model_name
model_list=()
if [[ -n "$models_file" ]]; then
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        _line="${_line%%#*}"                              # strip inline comments
        _line="${_line#"${_line%%[![:space:]]*}"}"        # ltrim whitespace
        _line="${_line%"${_line##*[![:space:]]}"}"        # rtrim whitespace
        [[ -n "$_line" ]] && model_list+=("$_line")
    done < "$models_file"
elif [[ ${#models[@]} -gt 0 ]]; then
    model_list=("${models[@]}")
fi

# ── derive output filename for a model ───────────────────────────────────────
_model_output_file() {
    local slug="${1//\//_}"
    slug="${slug//-/_}"
    echo "${output_dir}/benchmark_${slug}.json"
}

# ── run benchmark for one model ───────────────────────────────────────────────
_run_one() {
    local model="$1" out_file="$2"
    local cmd=(
        python "${BENCHMARK_PY}"
        --local_model_name "${model}"
        --field_path        "${field_path}"
        --max_samples       "${max_samples}"
        --warmup_batches    "${warmup_batches}"
        --max_num_tokens    "${max_num_tokens}"
        --batch_sizes       "${batch_sizes}"
        --output_file       "${out_file}"
    )
    [[ -n "$fof" ]]                      && cmd+=("--fof" "${fof}")
    [[ "$skip_api" == "true" ]]          && cmd+=("--skip_api")
    [[ "$torch_compile" == "true" ]]     && cmd+=("--torch_compile")
    [[ "$trust_remote_code" == "true" ]] && cmd+=("--trust_remote_code")
    [[ ${#extra_args[@]} -gt 0 ]]        && cmd+=("${extra_args[@]}")

    if [[ "$force" != "true" && -f "$out_file" ]]; then
        echo "Skipping: ${out_file} already exists (use --force to re-run)"
        return 0
    fi
    echo "Running: ${cmd[*]}"
    echo "Output : ${out_file}"
    echo ""
    [[ "$dry_run" == "true" ]] && return 0
    "${cmd[@]}"
}

# ── assemble and run ──────────────────────────────────────────────────────────
if [[ ${#model_list[@]} -gt 0 ]]; then
    echo "Running benchmark for ${#model_list[@]} model(s)..."
    echo ""
    for _model in "${model_list[@]}"; do
        _run_one "$_model" "$(_model_output_file "$_model")"
    done
else
    # Single model — honour explicit --output_file if given
    if [[ -z "${output_file}" ]]; then
        output_file="$(_model_output_file "$local_model_name")"
    elif [[ "${output_file}" != /* ]]; then
        output_file="${output_dir}/${output_file}"
    fi
    _run_one "$local_model_name" "$output_file"
fi
