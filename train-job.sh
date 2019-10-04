#!/bin/bash

set -e
set -x

owner=$1
experiment=$2
dataset=$3
model=$4
shift
shift
shift
shift

aws s3 sync s3://almond-research/${owner}/dataset/${experiment}${dataset} dataset/

modeldir="$HOME/models/$model"
mkdir -p "$modeldir"
rm -fr "$modeldir/dataset"
mkdir "$modeldir/dataset"
rm -fr "$modeldir/cache"
mkdir -p "$modeldir/cache"
ln -s /opt/dataset/${owner}/contextual/$dataset "$modeldir/dataset/contextual_almond"
ln -s $modeldir /home/genie-toolkit/current

decanlp train \
  --data "$modeldir/dataset" \
  --embeddings ${DECANLP_EMBEDDINGS} \
  --save "$modeldir" \
  --cache "$modeldir/cache" \
  --train_tasks contextual_almond  \
  --train_iterations 100000 \
  --preserve_case \
  --save_every 2000 \
  --log_every 500 \
  --val_every 1000 \
  --exist_ok \
  "$@"

rm -fr "$modeldir/cache"
aws s3 sync $modeldir/ s3://almond-research/${owner}/models/${experiment}${model}
