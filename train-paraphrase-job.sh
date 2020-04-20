#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner project experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${dataset} dataset/

modeldir="$HOME/$model"
mkdir -p "$modeldir"
mkdir -p "/shared/tensorboard/${project}/${experiment}/${owner}/${model}"

# run paraphrase training script
genienlp train-paraphrase \
  --train_data_file dataset/train.tsv \
  --eval_data_file dataset/dev.tsv \
  --output_dir "$modeldir" \
  --tensorboard_dir "/shared/tensorboard/${project}/${experiment}/${owner}/${model}" \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --overwrite_output_dir \
  --logging_steps 200 \
  --save_steps 200 \
  --max_steps -1 \
  --save_total_limit 1 \
  "$@"

aws s3 sync $modeldir/ s3://almond-research/${owner}/models/${project}/${experiment}/${model}



