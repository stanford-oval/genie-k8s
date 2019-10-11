#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner task_name experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${owner}/dataset/${experiment}/${dataset} dataset/

modeldir="$HOME/models/$model"
mkdir -p "$modeldir"
rm -fr "$modeldir/dataset"
mkdir "$modeldir/dataset"
rm -fr "$modeldir/cache"
mkdir -p "$modeldir/cache"
ln -s "$HOME/dataset" "$modeldir/dataset/${task_name}"
ln -s $modeldir /home/genie-toolkit/current

decanlp train \
  --data "$modeldir/dataset" \
  --embeddings ${DECANLP_EMBEDDINGS} \
  --save "$modeldir" \
  --cache "$modeldir/cache" \
  --train_tasks "${task_name}" \
  --preserve_case \
  --save_every 2000 \
  --log_every 500 \
  --val_every 1000 \
  --exist_ok \
  "$@"

rm -fr "$modeldir/cache"
aws s3 sync $modeldir/ s3://almond-research/${owner}/models/${experiment}/${model}
