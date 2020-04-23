#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner project experiment \
            input_dataset output_dataset filtering_model paraphrasing_model_full_path skip_generation is_dialogue" "$@"
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

export paraphrasing_arguments=$@

make_input_ready(){
    if [ "$is_dialogue" = true ] ; then
    # add previous agent utterance; this additional context helps the paraphraser
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/dialog_to_tsv.py \
      input_dataset/user/train.tsv \
      --dialog_file input_dataset/synthetic.txt \
      output_dataset/with_context.tsv
    else
      : # nothing to be done
    fi
}

run_paraphrase(){
  echo $paraphrasing_arguments
  # run paraphrase generation
  if [ "$is_dialogue" = true ] ; then
    genienlp run-paraphrase \
      --model_name_or_path paraphraser/ \
      --input_file output_dataset/with_context.tsv \
      --output_file output_dataset/paraphrased.tsv \
      --input_column 0 \
      --prompt_column 1 \
      --thingtalk_column 2 \
      --gold_column 3 \
      $paraphrasing_arguments
  else
    genienlp run-paraphrase \
      --model_name_or_path paraphraser \
      --input_file input_dataset/train.tsv \
      --output_file output_dataset/paraphrased.tsv \
      --input_column 1 \
      --thingtalk_column 2 \
      $paraphrasing_arguments
  fi
}


join(){
  # join the original file and the paraphrasing output
  if [ "$is_dialogue" = true ] ; then
    export num_new_queries=$((`wc -l output_dataset/paraphrased.tsv | cut -d " " -f1` / `wc -l input_dataset/user/train.tsv | cut -d " " -f1`))
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
      input_dataset/user/train.tsv \
      output_dataset/unfiltered.tsv \
      --query_file output_dataset/paraphrased.tsv \
      --num_new_queries ${num_new_queries} \
      --transformation replace_queries \
      --remove_duplicates \
      --output_columns 0 1 2 3 \
      --utterance_column 2 \
      --thingtalk_column 3
    else
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
}

run_parser(){
  # get parser output for paraphrased utterances
  if [ "$is_dialogue" = true ] ; then
    mkdir -p filtering_dataset/almond/user/
    cp output_dataset/unfiltered.tsv filtering_dataset/almond/user/eval.tsv
    genienlp predict \
      --data ./filtering_dataset \
      --path filtering_model \
      --eval_dir ./eval_dir \
      --evaluate valid \
      --task almond_dialogue_nlu \
      --overwrite \
      --silent \
      --main_metric_only \
      --skip_cache \
      --val_batch_size 600
  else
    mkdir -p filtering_dataset/almond/
    cp output_dataset/unfiltered.tsv filtering_dataset/almond/eval.tsv
    genienlp predict \
      --data ./filtering_dataset \
      --path filtering_model \
      --eval_dir ./eval_dir \
      --evaluate valid \
      --task almond \
      --overwrite \
      --silent \
      --main_metric_only \
      --skip_cache \
      --val_batch_size 2000
  fi
}

filter(){
  # remove paraphrases that do not preserve the meaning according to the parser
  if [ "$is_dialogue" = true ] ; then
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
      output_dataset/unfiltered.tsv \
      output_dataset/filtered.tsv \
      --thingtalk_gold_file eval_dir/valid/almond_dialogue_nlu.tsv \
      --transformation remove_wrong_thingtalk \
      --remove_duplicates \
      --output_columns 0 1 2 3 \
      --utterance_column 2 \
      --thingtalk_column 3
  else
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
      output_dataset/unfiltered.tsv \
      output_dataset/filtered.tsv \
      --thingtalk_gold_file ./eval_dir/valid/almond.tsv \
      --transformation remove_wrong_thingtalk \
      --remove_duplicates
  fi
}

append_to_original(){
  # append paraphrases to the end of the original training file and remove duplicates
  if [ "$is_dialogue" = true ] ; then
    cp ./eval_dir/valid/almond_dialogue_nlu.results.json ./output_dataset/
    cp output_dataset/user/train.tsv output_dataset/user/train2.tsv
    cat output_dataset/filtered.tsv >> output_dataset/user/train2.tsv
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
      output_dataset/user/train2.tsv \
      output_dataset/user/train.tsv \
      --remove_duplicates \
      --utterance_column 2
    rm output_dataset/user/train2.tsv
  else
    cp ./eval_dir/valid/almond.results.json ./output_dataset/
    cp output_dataset/train.tsv output_dataset/train2.tsv
    cat output_dataset/filtered.tsv >> output_dataset/train2.tsv
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
      output_dataset/train2.tsv \
      output_dataset/train.tsv \
      --remove_duplicates
    rm output_dataset/train2.tsv
  fi
}


if [ "$skip_generation" = true ] ; then
  echo "Skipping generation. Will filter the existing generations."
  if ! test -f input_dataset/unfiltered.tsv ; then
    exit 1
  fi
else
  make_input_ready
  run_paraphrase
  join
fi

run_parser
filter
append_to_original

# upload the new dataset to S3
aws s3 sync output_dataset s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${output_dataset}



