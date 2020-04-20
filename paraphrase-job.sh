#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner project experiment \
            input_dataset output_dataset filtering_model paraphrasing_model_full_path skip_generation" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${input_dataset} input_dataset/
aws s3 sync --exclude '*/dataset/*' --exclude '*/cache/*' --exclude 'iteration_*.pth' --exclude '*_optim.pth' s3://almond-research/${owner}/models/${project}/${experiment}/${filtering_model} filtering_model/
aws s3 sync s3://almond-research/${paraphrasing_model_full_path} paraphraser/

mkdir -p output_dataset
cp -r input_dataset/* output_dataset

on_error () {
  # on failure upload the outputs to S3
  aws s3 sync output_dataset s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${output_dataset}
}
trap on_error ERR

if [ "$skip_generation" = true ] ; then
  echo "Skipping generation. Will filter the existing generations."
  if ! test -f input_dataset/unfiltered.tsv ; then
    exit 1
  fi
else
  # run paraphrase generation script
  genienlp run-paraphrase \
    --model_name_or_path paraphraser \
    --input_file input_dataset/train.tsv \
    --input_column 1 \
    --thingtalk_column 2 \
    --output_file output_dataset/paraphrased.tsv \
    "$@"

  # join the original file and the paraphrasing output
  export num_new_queries=$((`wc -l output_dataset/paraphrased.tsv | cut -d " " -f1` / `wc -l input_dataset/train.tsv | cut -d " " -f1`))
  python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
    input_dataset/train.tsv \
    output_dataset/unfiltered.tsv \
    --query_file output_dataset/paraphrased.tsv \
    --num_new_queries ${num_new_queries} \
    --transformation replace_queries \
    --remove_duplicates \
    --remove_with_heuristics
fi

# get parser output for paraphrased utterances
mkdir -p almond/
cp output_dataset/unfiltered.tsv almond/eval.tsv
genienlp predict \
  --data ./ \
  --path filtering_model \
  --eval_dir ./eval_dir \
  --evaluate valid \
  --task almond \
  --overwrite \
  --silent \
  --main_metric_only \
  --skip_cache \
  --val_batch_size 2000

# remove paraphrases that do not preserve the meaning according to the parser
python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
  output_dataset/unfiltered.tsv \
  output_dataset/filtered.tsv \
  --thingtalk_gold_file ./eval_dir/valid/almond.tsv \
  --transformation remove_wrong_thingtalk \
  --remove_duplicates

cp ./eval_dir/valid/almond.results.json ./output_dataset/
# append paraphrases to the end of the original training file
cat output_dataset/train.tsv > output_dataset/train2.tsv
cat output_dataset/filtered.tsv >> output_dataset/train2.tsv
python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
  output_dataset/train2.tsv \
  output_dataset/train.tsv \
  --remove_duplicates
rm output_dataset/train2.tsv

# upload the new dataset to S3
aws s3 sync output_dataset s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${output_dataset}



