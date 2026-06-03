#!/usr/bin/env bash
# Wrapper around prune_vocabulary.py that activates the 'docu5' conda environment.
#
# Usage:
#   prune_vocabulary.sh [--conda_dir DIR] [prune_vocabulary.py args...]
#
# Options:
#   --conda_dir DIR   Root directory of the conda/miniforge installation
#                     (default: /home/raduf/miniforge3)

set -euo pipefail

CONDA_DIR="/home/raduf/miniforge3"
CONDA_ENV="docu5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/prune_vocabulary.py"

# Pull out --conda_dir if present, pass everything else through to Python
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda_dir)
            CONDA_DIR="$2"
            shift 2
            ;;
        --conda_dir=*)
            CONDA_DIR="${1#*=}"
            shift
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

# Activate conda environment
CONDA_SH="${CONDA_DIR}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda init script not found at ${CONDA_SH}" >&2
    echo "       Use --conda_dir to specify the correct miniforge/anaconda directory." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

exec python "$SCRIPT" "${PASSTHROUGH[@]}"
