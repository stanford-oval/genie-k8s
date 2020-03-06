#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner experiment dataset model workdir task_name" "$@"
shift $n

set -e
set -x

 on_error () {
 	# on failure ship everything to s3
 	aws s3 sync . s3://almond-research/${owner}/models/${experiment}/${model}/failed_eval/
 }
 trap on_error ERR

pwd
aws s3 sync s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset} dataset/ --exclude "*eval/*"


mkdir -p ${experiment}/models
aws s3 sync s3://almond-research/${owner}/models/${experiment}/${model}/ ${experiment}/models/${model}/ --exclude "iteration_*.pth"

workingdir="$HOME/${workdir}"
mkdir -p ${workingdir}/cache
mkdir -p ${workingdir}/eval_dir


ln -s "$HOME/dataset" "$HOME/dataset/${task_name}"

ls -R

genienlp predict \
  --data "$HOME/dataset" \
  --embeddings ${GENIENLP_EMBEDDINGS} \
  --cache "${workingdir}/cache" \
  --path "$HOME/${experiment}/models/${model}/" \
  --eval_dir ${workingdir}/eval_dir \
  --skip_cache \
  --tasks ${task_name} \
  "$@"

ls -R
aws s3 sync ${workingdir}/eval_dir s3://almond-research/${owner}/models/${experiment}/${model}/${dataset}/eval/
