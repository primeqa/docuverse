#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_yaml_config.sh"

# defaults
conda_env=docu
max_num_tokens=512
fof=benchmark/miracl/miracle.fof
field_path=text
batch_sizes=1024
max_samples=32768  # The max number of samples to run
output_dir="latency"
torch_compile=true
dry_run=0
models=()

# Pre-pass: load YAML defaults if --config provided (CLI flags below override)
config_file=$(find_config_arg "$@")
if [[ -n "$config_file" ]]; then
    eval "$(load_yaml_config "$config_file")" || exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) shift 2 ;;  # already consumed in pre-pass
        --conda_env) conda_env="$2"; shift 2 ;;
        --max_num_tokens) max_num_tokens="$2"; shift 2 ;;
        --fof) fof="$2"; shift 2 ;;
        --field_path) field_path="$2"; shift 2 ;;
        --batch_sizes) batch_sizes="$2"; shift 2 ;;
        --max_samples) max_samples="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --torch_compile) torch_compile=true; shift; ;;
        --no_torch_compile|--no_torch_conmpile) torch_compile=false; shift; ;;
        --dry-run|--dry_run) dry_run=1; shift; ;;
        --help) echo "Usage: $0 [OPTIONS] MODEL [MODEL ...]"
                echo "Options:"
                echo "  --config FILE         YAML config providing defaults (CLI flags override)"
                echo "  --conda_env NAME      Conda environment (default: docu)"
                echo "  --max_num_tokens N    Max tokens (default: 512)"
                echo "  --fof FILE            File of files (default: benchmark/miracl/miracle.fof)"
                echo "  --field_path PATH     Field path (default: text)"
                echo "  --batch_sizes N       Batch sizes (default: 1024)"
                echo "  --max_samples N       Max samples (default: 32768)"
                echo "  --output_dir DIR      Output directory (default: latency)"
                echo "  --torch_compile       Enable torch.compile (default)"
                echo "  --no_torch_compile    Disable torch.compile"
                echo "  --dry-run             Print the bsub commands without submitting"
                exit 0 ;;
        --) shift; break ;;
        --*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

# CLI positional models override the YAML `models:` list
if [[ $# -gt 0 ]]; then
    models=("$@")
fi

if [[ ${#models[@]} -eq 0 ]]; then
    echo "No models specified (pass as positional args or via 'models:' in YAML)" >&2
    exit 1
fi

if [[ "$torch_compile" == "true" ]]; then
    torch_flag="--torch_compile"
else
    torch_flag="--no_torch_conmpile"
fi

for MODEL in "${models[@]}"; do
    inner_cmd="
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${conda_env}
cd /u/raduf/sandbox2/docuverse
./scripts/speed_test/run_benchmark.sh \
--local_model_name ${MODEL} \
--field_path ${field_path} \
--max_num_tokens ${max_num_tokens} \
--batch_sizes ${batch_sizes} \
--max_samples ${max_samples} \
--output_dir ${output_dir} \
${torch_flag} \
--fof ${fof}
#  %JDone
"
    if [[ $dry_run -eq 1 ]]; then
        echo "[dry-run] bsub -o /u/raduf/.lsbatch/%J.out -e /u/raduf/.lsbatch/%J.err -gpu \"num=1:J_exclusive=yes\" -M 150G -G grp_preemptable -q preemptable bash -c '${inner_cmd}'"
    else
        bsub -o /u/raduf/.lsbatch/%J.out -e /u/raduf/.lsbatch/%J.err -gpu "num=1:J_exclusive=yes" -M 150G -G grp_preemptable -q preemptable bash -c "${inner_cmd}"
    fi
done
