#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "train_or_gen owner dataset_owner task_name experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset} dataset/

modeldir="$HOME/models/$model"
mkdir -p "$modeldir"
rm -fr "$modeldir/dataset"
mkdir "$modeldir/dataset"
rm -fr "$modeldir/cache"
mkdir -p "$modeldir/cache"
ln -s "$HOME/dataset" "$modeldir/dataset/${task_name}"
ln -s $modeldir /home/genie-toolkit/current
mkdir -p "/shared/tensorboard/${experiment}/${owner}/${model}"


if [ "$train_or_gen" = "train" ]
then
  # run paraphrase training script
  genienlp train-paraphrase \
    --train_data_file dataset/para_freeform_train.txt \
    --eval_data_file dataset/para_freeform_dev.txt \
    --output_dir "$modeldir" \
    --tensorboard_dir "/shared/tensorboard/${experiment}/${owner}/${model}" \
    --model_type gpt2 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir \
    --logging_steps 1000 \
    --save_steps 1000 \
    --max_steps 40000 \
    --save_total_limit 1 \
    "$@"
  aws s3 sync $modeldir/ s3://almond-research/${owner}/models/${experiment}/${model}
else
  # run paraphrase generation script
  workingdir="$HOME/${workdir}"
  mkdir -p ${workingdir}/eval_dir
  aws s3 sync s3://almond-research/${owner}/models/${experiment}/${model} $modeldir/
  genienlp run-paraphrase \
    --model_type gpt2 \
    --model_name_or_path "$modeldir" \
    --input_file dataset/test.tsv \
    --input_column 1 \
    --output_file ${workingdir}/eval_dir/output.tsv \
    "$@"
  aws s3 sync ${workingdir}/eval_dir s3://almond-research/${owner}/models/${experiment}/${model}/eval/
fi



rm -fr "$modeldir/cache"
rm -fr "$modeldir/dataset"
