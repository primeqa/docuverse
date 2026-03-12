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
  echo "  -h, --help                    Show this help message"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODELS="$2"; shift 2 ;;
    --steps)
      STEPS="$2"; shift 2 ;;
    --configs)
      read -ra CONFIGS <<< "$2"; shift 2 ;;
    --num-preprocessor-threads | --num_preprocessor_threads)
      NUM_THREADS="$2"; shift 2 ;;
    --max-doc-length)
      MAX_DOC_LENGTHS="$2"; shift 2 ;;
    --stride)
      STRIDE="$2"; shift 2 ;;
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

# ---- If no models/doc-lengths given, use a single empty placeholder ----
[ -z "$MODELS" ]          && MODELS="_none_"
[ -z "$MAX_DOC_LENGTHS" ] && MAX_DOC_LENGTHS="_none_"

run_idx=0
for model in $MODELS; do
  SHORT_MODEL=""
  if [ "$model" != "_none_" ]; then
    SHORT_MODEL=$(shorten_model "$model")
  fi

  for mdl in $MAX_DOC_LENGTHS; do
    for config in "${CONFIGS[@]}"; do
      for i in $STEPS; do
        run_idx=$((run_idx + 1))

        CMD=(CUDA_VISIBLE_DEVICES=1 python docuverse/utils/ingest_and_test.py)
        CMD+=(--num_preprocessor_threads "$NUM_THREADS")
        [ "$model" != "_none_" ] && CMD+=(--model_name "$model")
        [ "$mdl" != "_none_" ]   && CMD+=(--max_doc_length "$mdl")
        [ -n "$STRIDE" ]         && CMD+=(--stride "$STRIDE")
        CMD+=(--config "$config")
        CMD+=(--matryoshka_dim "$i")

        # Build short name: model[-mdlN]-dimN
        if [ -n "$SHORT_MODEL" ]; then
          short_suffix="${SHORT_MODEL}"
          [ "$mdl" != "_none_" ] && short_suffix="${short_suffix}-mdl${mdl}"
          short_suffix="${short_suffix}-dim${i}"
          CMD+=(--short_model_name "$short_suffix")
        fi

        echo "[$run_idx/$total]"
        runCmd "${CMD[*]}"
      done
    done
  done
done
