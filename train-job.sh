#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner task_name experiment dataset model load_from" "$@"
shift $n

set -e
set -x

modeldir="$HOME/models/$model"
mkdir -p "$modeldir"

if ! test  ${load_from} = 'None' ; then
	aws s3 sync ${load_from}/ "$modeldir"/ --exclude "iteration_*.pth" --exclude "*eval/*"  --exclude "*.log"
fi

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset} dataset/

rm -fr "$modeldir/dataset"
mkdir "$modeldir/dataset"
rm -fr "$modeldir/cache"
mkdir -p "$modeldir/cache"
ln -s "$HOME/dataset" "$modeldir/dataset/${task_name}"
ln -s $modeldir /home/genie-toolkit/current
mkdir -p "/shared/tensorboard/${experiment}/${owner}/${model}"

on_error () {
  # on failure ship everything to s3
  aws s3 sync $modeldir/ s3://almond-research/${owner}/models/${experiment}/${model}/failed_train/
}
trap on_error ERR


genienlp train \
  --data "$modeldir/dataset" \
  --embeddings ${GENIENLP_EMBEDDINGS} \
  --save "$modeldir" \
  --tensorboard_dir "/shared/tensorboard/${experiment}/${owner}/${model}" \
  --cache "$modeldir/cache" \
  --train_tasks "${task_name}" \
  --preserve_case \
  --save_every 1000 \
  --log_every 100 \
  --val_every 1000 \
  --exist_ok \
  --skip_cache \
  "$@" 
  
rm -fr "$modeldir/cache"
rm -fr "$modeldir/dataset"
aws s3 sync ${modeldir}/ s3://almond-research/${owner}/models/${experiment}/${model}
