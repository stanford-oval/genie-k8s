#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner task_name project experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${dataset} dataset/

modeldir="$HOME/models/$model"
mkdir -p "$modeldir"
rm -fr "$modeldir/dataset"
mkdir "$modeldir/dataset"
rm -fr "$modeldir/cache"
mkdir -p "$modeldir/cache"
ln -s "$HOME/dataset" "$modeldir/dataset/${task_name}"
ln -s $modeldir /home/genie-toolkit/current
mkdir -p "/shared/tensorboard/${project}/${experiment}/${owner}/${model}"

genienlp train \
  --data "$modeldir/dataset" \
  --embeddings ${DECANLP_EMBEDDINGS} \
  --save "$modeldir" \
  --tensorboard_dir "/shared/tensorboard/${project}/${experiment}/${owner}/${model}" \
  --cache "$modeldir/cache" \
  --train_tasks "${task_name}" \
  --preserve_case \
  --save_every 1000 \
  --log_every 100 \
  --val_every 1000 \
  --exist_ok \
  "$@"

rm -fr "$modeldir/cache"
rm -fr "$modeldir/dataset"
aws s3 sync $modeldir/ s3://almond-research/${owner}/models/${project}/${experiment}/${model}
