#!/bin/bash

checkError() {
  # Exitcode has to be saved first or else shift clobbers it
  EXIT_CODE=$?
  ERR_TITLE=$1
  ERR_DESCP=$2
  ERR_CMD=$3
  FILE_TO_CHECK=$4
  CMD_TO_RUN=$5

  if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: $ERR_TITLE $ERR_DESCP"

    if [ "$JSON_LOG_FILE" != "" ]; then
      current_time=$(date --rfc-3339=seconds)
      current_time=$(printf %q "$current_time")

      if [ "$ERR_TITLE" != "" ]; then
        ERR_TITLE=$(printf %q "$ERR_TITLE")
      fi
      if [ "$ERR_DESCP" != "" ]; then
        ERR_DESCP=$(printf %q "$ERR_DESCP")
      fi

      echo "{\"timestamp\":\"$current_time\" " >>$JSON_LOG_FILE
      echo " \"error\": \"$ERR_TITLE\" " >>$JSON_LOG_FILE
      echo " \"description\":\"$ERR_DESCP\" " >>$JSON_LOG_FILE
      echo " \"command\":\"$ERR_CMD\"}" >>$JSON_LOG_FILE
      echo " \"code\":\"$EXIT_CODE\"}" >>$JSON_LOG_FILE
    fi

    if [ "$FILE_TO_CHECK" != "" ]; then
      if [ ! -e $FILE_TO_CHECK ]; then
        echo "The file $FILE_TO_CHECK does not exist - want to run the command '$CMD_TO_RUN' ([Y]es/[n]o)?"
        while true; do
          read -t 2 response
          if [ "$response" == "y" -o "$response" = "Y" -o "$response" = "yes" -o "$response" = "Yes" -o "$response" = "" ]; then
            eval $CMD_TO_RUN
            break
          elif [ "$response" = "n" -o "$response" = "no" ]; then
            break
          else
            echo "Please answer with 'y' or 'yes' or 'n' or 'no'"
          fi
        done
      fi
    fi
    exit $EXIT_CODE
  fi
}

# run a command and check its return code
runCmd() {
  echo "================================================================"
  printf "Running \e[38;5;87m %s \e[0m" "$1"
  echo
  # echo Running "\e[38;5;87m" "$1" "\e[0m"
  time eval $1
  EXIT_CODE=$?

  if [ $EXIT_CODE -ne 0 ]; then
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    echo "failed running $1"
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  fi
  echo "================================================================"

  return $EXIT_CODE
}

checkFile() {
  if [ ! -e $1 ]; then
    echo "The file $1 does not exist!"
    exit 11
  fi
}

function join_by { local IFS="$1"; shift; echo "$*"; }

echo "$0 $*" >>run_cmmds

# ---- Defaults ----
STEPS="064 128 256 512 768"
NUM_THREADS=10
MODELS=""
MAX_DOC_LENGTHS=""
STRIDE=""
CONFIGS=()
USE_BSUB=0
BSUB_QUEUE=""
BSUB_REQUIREMENTS=""
BSUB_EXTRA=""
CONDA_ENV="docu"
GPU_ID="0"
EXCLUDE_FILE=""

# ---- Parse arguments ----
usage() {
  echo "Usage: $0 [OPTIONS] [config1 config2 ...]"
  echo
  echo "Options:"
  echo "  --model \"m1 m2 ...\"           Model name(s) to sweep (space-separated)"
  echo "  --steps \"d1 d2 ...\"           Matryoshka dims to sweep (default: \"$STEPS\")"
  echo "  --configs \"c1 c2 ...\"         Config files (alternative to positional args)"
  echo "  --num-preprocessor-threads N Number of preprocessing threads (default: $NUM_THREADS)"
  echo "  --max-doc-length \"N1 N2 ...\"  Max document length(s) to sweep (space-separated)"
  echo "  --stride N                    Stride for document chunking"
  echo "  --bsub                        Submit jobs via bsub instead of running locally"
  echo "  --queue Q                     bsub queue name (implies --bsub)"
  echo "  --bsub-requirements R         bsub resource requirements string, e.g. \"rusage[mem=16000,ngpus_physical=1]\""
  echo "  --bsub-extra \"ARGS\"           Extra arguments passed verbatim to bsub"
  echo "  --conda-env NAME              Conda environment to activate (default: $CONDA_ENV)"
  echo "  --gpu ID                      CUDA_VISIBLE_DEVICES for local runs (default: $GPU_ID, ignored with --bsub)"
  echo "  --exclude FILE                File with combinations to skip (tab-separated):"
  echo "                                  <model> <max_doc_length> <config> <dim>"
  echo "                                Lines starting with # are ignored. Use * to match any value."
  echo "  -h, --help                    Show this help message"
  exit 0
}

# collect_values: consume all following args until the next flag (starts with -)
# Usage: collect_values "$@"; set -- "${_remaining[@]}"
# Results are in _collected (array)
collect_values() {
  _collected=()
  _remaining=()
  local found=0
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == -* && $found -gt 0 ]]; then
      _remaining=("$@")
      return
    fi
    _collected+=("$1")
    found=$((found + 1))
    shift
  done
  _remaining=()
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      shift; collect_values "$@"; set -- "${_remaining[@]}"
      MODELS="${_collected[*]}" ;;
    --steps)
      shift; collect_values "$@"; set -- "${_remaining[@]}"
      STEPS="${_collected[*]}" ;;
    --configs)
      shift; collect_values "$@"; set -- "${_remaining[@]}"
      CONFIGS+=("${_collected[@]}") ;;
    --max-doc-length)
      shift; collect_values "$@"; set -- "${_remaining[@]}"
      MAX_DOC_LENGTHS="${_collected[*]}" ;;
    --num-preprocessor-threads | --num_preprocessor_threads)
      NUM_THREADS="$2"; shift 2 ;;
    --stride)
      STRIDE="$2"; shift 2 ;;
    --bsub)
      USE_BSUB=1; shift ;;
    --queue)
      BSUB_QUEUE="$2"; USE_BSUB=1; shift 2 ;;
    --bsub-requirements | --bsub-req)
      BSUB_REQUIREMENTS="$2"; USE_BSUB=1; shift 2 ;;
    --bsub-extra)
      BSUB_EXTRA="$2"; USE_BSUB=1; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    --gpu)
      GPU_ID="$2"; shift 2 ;;
    --exclude)
      EXCLUDE_FILE="$2"; shift 2 ;;
    -h|--help)
      usage ;;
    -*)
      echo "Unknown option: $1"; usage ;;
    *)
      CONFIGS+=("$1"); shift ;;
  esac
done

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "Error: no config files specified."
  usage
fi

# ---- Resolve model name(s) ----
# If --model not given on CLI, extract from the first config file
if [ -z "$MODELS" ]; then
  first_config="${CONFIGS[0]}"
  if [ -f "$first_config" ]; then
    MODELS=$(grep -E '^\s*model_name\s*:' "$first_config" | head -1 \
             | sed -E 's/^[^:]+:\s*"?([^"]+)"?\s*$/\1/')
    [ -n "$MODELS" ] && echo "Model (from config): $MODELS"
  fi
fi

# Build SHORT_MODEL: strip org prefix, remove filler words, clean up hyphens
shorten_model() {
  local m="${1##*/}"                                       # strip everything before /
  echo "$m" | sed -E \
    -e 's/[-_]?(embedding|embed)(s)?[-_]?/-/gi' \
    -e 's/[-_]?(english|multilingual)[-_]?/-/gi' \
    -e 's/--+/-/g' \
    -e 's/^-//' \
    -e 's/-$//'
}

# ---- Print sweep summary ----
echo "Configs:       ${CONFIGS[*]}"
echo "Steps:         $STEPS"
echo "Threads:       $NUM_THREADS"
[ -n "$MODELS" ]          && echo "Models:        $MODELS"
[ -n "$MAX_DOC_LENGTHS" ] && echo "MaxDocLengths: $MAX_DOC_LENGTHS"
[ -n "$STRIDE" ]          && echo "Stride:        $STRIDE"
[ -n "$EXCLUDE_FILE" ]    && echo "Exclude file:  $EXCLUDE_FILE"
if [ "$USE_BSUB" -eq 1 ]; then
  echo "Submit:        bsub"
  [ -n "$BSUB_QUEUE" ]        && echo "Queue:         $BSUB_QUEUE"
  [ -n "$BSUB_REQUIREMENTS" ] && echo "Requirements:  $BSUB_REQUIREMENTS"
  [ -n "$BSUB_EXTRA" ]        && echo "Extra args:    $BSUB_EXTRA"
  echo "Conda env:     $CONDA_ENV"
fi

# Count total combinations
n_models=$(echo $MODELS | wc -w)
n_steps=$(echo $STEPS | wc -w)
n_configs=${#CONFIGS[@]}
n_doclen=$(echo $MAX_DOC_LENGTHS | wc -w)
[ "$n_models" -eq 0 ]  && n_models=1
[ "$n_doclen" -eq 0 ]  && n_doclen=1
total=$((n_configs * n_models * n_doclen * n_steps))
echo "Total runs:    $total  ($n_configs configs x $n_models models x $n_doclen doc-lengths x $n_steps steps)"
echo

if [ "$USE_BSUB" -eq 1 ]; then
  read -r -p "Proceed with submitting $total bsub jobs? [Y/n] " response
  case "$response" in
    [nN]|[nN][oO])
      echo "Aborted."
      exit 0 ;;
  esac
fi

# ---- Load exclusion list ----
declare -A EXCLUDE_SET
n_excluded=0
if [ -n "$EXCLUDE_FILE" ]; then
  if [ ! -f "$EXCLUDE_FILE" ]; then
    echo "Error: exclude file not found: $EXCLUDE_FILE"
    exit 1
  fi
  while IFS=$'\t' read -r ex_model ex_mdl ex_config ex_dim; do
    # Skip empty lines and comments
    [ -z "$ex_model" ] && continue
    [[ "$ex_model" == \#* ]] && continue
    EXCLUDE_SET["${ex_model}|${ex_mdl}|${ex_config}|${ex_dim}"]=1
    n_excluded=$((n_excluded + 1))
  done < "$EXCLUDE_FILE"
  echo "Loaded $n_excluded exclusion(s) from $EXCLUDE_FILE"
fi

# ---- If no models/doc-lengths given, use a single empty placeholder ----
[ -z "$MODELS" ]          && MODELS="_none_"
[ -z "$MAX_DOC_LENGTHS" ] && MAX_DOC_LENGTHS="_none_"

# is_excluded: check if a combination should be skipped
# Supports * as wildcard in any field of the exclude file
is_excluded() {
  local e_model="$1" e_mdl="$2" e_config="$3" e_dim="$4"
  # Check exact match first
  if [ "${EXCLUDE_SET["${e_model}|${e_mdl}|${e_config}|${e_dim}"]+_}" ]; then
    return 0
  fi
  # Check wildcard matches
  for key in "${!EXCLUDE_SET[@]}"; do
    IFS='|' read -r pat_model pat_mdl pat_config pat_dim <<< "$key"
    if { [ "$pat_model" = "*" ]  || [ "$pat_model" = "$e_model" ]; } && \
       { [ "$pat_mdl" = "*" ]    || [ "$pat_mdl" = "$e_mdl" ]; } && \
       { [ "$pat_config" = "*" ] || [ "$pat_config" = "$e_config" ]; } && \
       { [ "$pat_dim" = "*" ]    || [ "$pat_dim" = "$e_dim" ]; }; then
      return 0
    fi
  done
  return 1
}

run_idx=0
skip_count=0
for model in $MODELS; do
  SHORT_MODEL=""
  if [ "$model" != "_none_" ]; then
    SHORT_MODEL=$(shorten_model "$model")
  fi

  for mdl in $MAX_DOC_LENGTHS; do
    for config in "${CONFIGS[@]}"; do
      for i in $STEPS; do
        run_idx=$((run_idx + 1))

        # Check exclusion list
        if [ "$n_excluded" -gt 0 ] && is_excluded "$model" "$mdl" "$config" "$i"; then
          skip_count=$((skip_count + 1))
          echo "[$run_idx/$total] Skipping (excluded): model=$model mdl=$mdl config=$config dim=$i"
          continue
        fi

        CMD=(python docuverse/utils/ingest_and_test.py)
        CMD+=(--num_preprocessor_threads "$NUM_THREADS")
        [ "$model" != "_none_" ] && CMD+=(--model_name "$model")
        [ "$mdl" != "_none_" ]   && CMD+=(--max_doc_length "$mdl")
        [ -n "$STRIDE" ]         && CMD+=(--stride "$STRIDE")
        CMD+=(--config "$config")
        CMD+=(--matryoshka_dim "$i")

        # Build short name: model[-mdlN]-dimN
        short_suffix=""
        if [ -n "$SHORT_MODEL" ]; then
          short_suffix="${SHORT_MODEL}"
          [ "$mdl" != "_none_" ] && short_suffix="${short_suffix}-mdl${mdl}"
          short_suffix="${short_suffix}-dim${i}"
          CMD+=(--short_model_name "$short_suffix")
        fi

        echo "[$run_idx/$total]"
        if [ "$USE_BSUB" -eq 1 ]; then
          # Build bsub arguments
          BSUB_CMD=(bsub)
          # Job name: config basename + short suffix or dim
          cfg_base=$(basename "$config" | sed 's/\.\(yaml\|yml\|json\)$//')
          if [ -n "$short_suffix" ]; then
            job_name="${cfg_base}_${short_suffix}"
          else
            job_name="${cfg_base}_dim${i}"
          fi
          BSUB_CMD+=(-J "$job_name")
          # Build log file name encoding model, doc-length, and dim
          log_parts=("%J"
          )
          [ -n "$SHORT_MODEL" ]    && log_parts+=("$SHORT_MODEL")
          [ "$mdl" != "_none_" ]   && log_parts+=("mdl${mdl}")
          log_parts+=("dim${i}")
          log_tag=$(IFS=-; echo "${log_parts[*]}")
          BSUB_CMD+=(-o "/u/raduf/tmp/${cfg_base}_${log_tag}.out" -e "/u/raduf/tmp/${cfg_base}_${log_tag}.err")
          [ -n "$BSUB_QUEUE" ]        && BSUB_CMD+=(-q "$BSUB_QUEUE")
          [ -n "$BSUB_REQUIREMENTS" ] && BSUB_CMD+=(-R "$BSUB_REQUIREMENTS")
          [ -n "$BSUB_EXTRA" ]        && read -ra _bsub_extra_arr <<< "$BSUB_EXTRA" && BSUB_CMD+=("${_bsub_extra_arr[@]}")
          
          RANDOM_VALUE=$RANDOM
          TMP_DB_PATH="/tmp/unifsearch_${RANDOM_VALUE}.db"
          CMD+=("--server file:$TMP_DB_PATH")

          echo "  Submitting: ${BSUB_CMD[*]} ... ${CMD[*]}"
          "${BSUB_CMD[@]}" bash -c "
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
cd $(pwd)
${CMD[*]}
rm $TMP_DB_PATH
"
        else
          runCmd "CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD[*]}"
        fi
      done
    done
  done
done

if [ "$skip_count" -gt 0 ]; then
  echo "Skipped $skip_count/$total run(s) due to exclusions."
fi
