#!/bin/bash

set -e
set -x

experiment=$1
dataset=$2
model=$3
shift
shift
shift

aws s3 sync s3://almond-research/gcampax/dataset/${experiment}${dataset} dataset/

modeldir="$HOME/models/$model"
mkdir -p "$modeldir"
rm -fr "$modeldir/dataset"
mkdir "$modeldir/dataset"
rm -fr "$modeldir/cache"
mkdir -p "$modeldir/cache"
ln -s /opt/dataset/gcampax/contextual/$dataset "$modeldir/dataset/contextual_almond"
ln -s $modeldir /home/genie-toolkit/current

decanlp train \
  --data "$modeldir/dataset" \
  --embeddings /usr/local/share/decanlp/embeddings \
  --save "$modeldir" \
  --cache "$modeldir/cache" \
  --train_tasks contextual_almond  \
  --train_iterations 100000 \
  --preserve_case \
  --save_every 2000 \
  --log_every 500 \
  --val_every 1000 \
  --exist_ok \
  --thingpedia /opt/dataset/thingpedia-8strict.json \
  "$@"

rm -fr "$modeldir/cache"
aws s3 sync $modeldir/ s3://almond-research/gcampax/models/${experiment}${model}
