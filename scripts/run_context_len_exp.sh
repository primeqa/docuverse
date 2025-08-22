#!/bin/bash

# set -x
# shellcheck disable=SC1090

rerank() {
  k=$1
  printString "Reranking $k"
  output=`basename $config | perl -pe 's/.yaml/.'$k'.json/'`
  runCmd "python docuverse/utils/ingest_and_test.py --config $config --actions re --output_file output/$output --reranker_top_k $k  $eval"
  return
}
echo "$0 $*" >>run_cmmds

source ~/bin/common.sh

ranks="20,100"
actions="re"

tasks="2wikimqa courtlistener_HTML courtlistener_Plain_Text gov_report legal_case_reports multifieldqa passage_retrieval qasper_abstract qasper_title qmsum stackoverflow summ_screen_fd"

weights="128 256 512 1024 2048 4096 8192 16384 32768"
base_config="experiments/unified_search/ibmsw_milvus_dense.granite-149m.w512.flash_attn.test.short.yaml"

SHORT_FLAGS="-o r:a:t:w:b:v"
lng_flags=(
  "ranks:"
  "actions:"
  "weights:"
  "tasks:"
  "base_config:"
  "verbose"
)
LONG_FLAGS="--long "$(join_by "," ${lng_flags[*]})
TEMP=$(getopt $SHORT_FLAGS $LONG_FLAGS -n 'run_exp.sh' -- "$@")

eval set -- "$TEMP"
quit=0
while true; do
  echo "Processing flag $1, with opt arg $2"
  case "$1" in
  -r | --ranks) ranks=$2; shift 2;;
  -a | --actions) actions=$2; shift 2;;
  -w | --weights) weights=$2; shift 2;;
  -t | --tasks) tasks=$2; shift 2;;
  -b | --base_config) base_config=$2; shift 2;;
  -h)    echo "$SHORT_FLAGS $LONG_FLAGS"; quit=1; break;;
  -d)    debug=1; shift; break;;
  -v | --verbose)    set -x; shift; break;;
  --)    shift; break;;
  *)     break ;;
  esac
done

if [[ $quit == 1 ]]; then
  exit
fi

config=$1
eval=$2

IFS=', ' read -r -a weights <<< "$weights";
IFS=', ' read -r -a tasks <<< "$*"

if [[ $eval != "" ]]; then
  eval="--input_queries $eval"
fi
#printString "No reranking"
#output=`basename $config | perl -pe 's/.yaml/.retrieve.json/'`
#runCmd "python docuverse/utils/ingest_and_test.py --config $config --actions $actions --output_file output/$output --reranker_engine none $eval"

#for i in $(echo $ranks | tr ',' ' '); do
#  rerank $i
#done
# rerank 100
expt_dir="experiments/locov1"

echo "Tasks: $tasks"

for task in "${tasks[@]}"; do
  for weight in "${weights[@]}"; do
    filename=$task
    runCmd "python docuverse/utils/ingest_and_test.py --config $filename --max_doc_length $weight"
    checkError "Failed to run with task $task, weight $weight" "run command" "python docuverse/utils/ingest_and_test.py --config $filename" "" ""
  done
done


#printString "Reranking 100"
#output=`basename $config | perl -pe 's/.yaml/.100.json/'`
#runCmd "python  docuverse/utils/ingest_and_test.py --config $config --actions re --output_file output/$output --reranker_top_k 100 --no-cache $eval"

