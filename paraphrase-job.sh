#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner dataset_owner project experiment \
            input_dataset output_dataset filtering_model paraphrasing_model_full_path skip_generation task_name" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${input_dataset} input_dataset/
aws s3 sync --exclude '*/dataset/*' --exclude '*/cache/*' --exclude 'iteration_*.pth' --exclude '*_optim.pth' s3://almond-research/${owner}/models/${project}/${experiment}/${filtering_model} filtering_model/
aws s3 sync s3://almond-research/${paraphrasing_model_full_path} paraphraser/

mkdir -p output_dataset
cp -r input_dataset/* output_dataset


export paraphrasing_arguments=$@
if [ "$task_name" = "almond_dialogue_nlu" ] ; then
  export is_dialogue=true
  export input_dir="input_dataset/user"
  export output_dir="output_dataset/user"
  export filtering_dir="almond/user"
  export filtering_batch_size="600"
elif [ "$task_name" = "almond" ] ; then
  export input_dir="input_dataset"
  export output_dir="output_dataset"
  export filtering_dir="almond"
  export filtering_batch_size="2000"
else
  exit 1
fi

make_input_ready(){
    # remove duplicates before paraphrasing to avoid wasting effort
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
      ${input_dir}/train.tsv \
      ${input_dir}/train_no_duplicate.tsv \
      --remove_duplicates \
      --task ${task_name}
      cp ${input_dir}/train_no_duplicate.tsv ${input_dir}/train.tsv

    if [ "$is_dialogue" = true ] ; then
    # add previous agent utterance; this additional context helps the paraphraser
    python3 /opt/genienlp/genienlp/data_manipulation_scripts/dialog_to_tsv.py \
      ${input_dir}/train.tsv \
      --dialog_file input_dataset/synthetic.txt \
      ${output_dir}/with_context.tsv
    else
      : # nothing to be done
    fi
}

run_paraphrase(){
  echo $paraphrasing_arguments
  # run paraphrase generation
  if [ "$is_dialogue" = true ] ; then
    genienlp run-paraphrase \
      --model_name_or_path paraphraser \
      --input_file ${output_dir}/with_context.tsv \
      --output_file ${output_dir}/paraphrased.tsv \
      --input_column 0 \
      --prompt_column 1 \
      --thingtalk_column 2 \
      --gold_column 3 \
      $paraphrasing_arguments
  else
    genienlp run-paraphrase \
      --model_name_or_path paraphraser \
      --input_file ${input_dir}/train.tsv \
      --output_file ${output_dir}/paraphrased.tsv \
      --input_column 1 \
      --thingtalk_column 2 \
      $paraphrasing_arguments
  fi
}

join(){
  # join the original file and the paraphrasing output
  export num_new_queries=$((`wc -l ${output_dir}/paraphrased.tsv | cut -d " " -f1` / `wc -l ${input_dir}/train.tsv | cut -d " " -f1`))
  python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
    ${input_dir}/train.tsv \
    ${output_dir}/unfiltered.tsv \
    --query_file ${output_dir}/paraphrased.tsv \
    --num_new_queries ${num_new_queries} \
    --transformation replace_queries \
    --remove_duplicates \
    --remove_with_heuristics \
    --task ${task_name}
}

run_parser(){
  # get parser output for paraphrased utterances
  mkdir -p filtering_dataset/${filtering_dir}
  cp ${output_dir}/unfiltered.tsv filtering_dataset/${filtering_dir}/eval.tsv
  genienlp predict \
    --data ./filtering_dataset \
    --path filtering_model \
    --eval_dir ./eval_dir \
    --evaluate valid \
    --task ${task_name} \
    --overwrite \
    --silent \
    --main_metric_only \
    --skip_cache \
    --val_batch_size ${filtering_batch_size}
}

filter(){
  # remove paraphrases that do not preserve the meaning according to the parser
  python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
    ${output_dir}/unfiltered.tsv \
    ${output_dir}/filtered.tsv \
    --thingtalk_gold_file eval_dir/valid/${task_name}.tsv \
    --transformation remove_wrong_thingtalk \
    --remove_duplicates \
    --task ${task_name}
}

append_to_original(){
  # append paraphrases to the end of the original training file and remove duplicates
  cp ./eval_dir/valid/${task_name}.results.json ${output_dir}/
  cp ${input_dir}/train.tsv ${output_dir}/temp.tsv
  cat ${output_dir}/filtered.tsv >> ${output_dir}/temp.tsv
  python3 /opt/genienlp/genienlp/data_manipulation_scripts/transform_dataset.py \
    ${output_dir}/temp.tsv \
    ${output_dir}/train.tsv \
    --remove_duplicates \
    --task ${task_name}
  rm ${output_dir}/temp.tsv
}


if [ "$skip_generation" = true ] ; then
  echo "Skipping generation. Will filter the existing generations."
  if ! test -f ${input_dir}/unfiltered.tsv ; then
    exit 1
  fi
else
  make_input_ready
  run_paraphrase
  join

  # paraphrasing was successful, so upload the intermediate files
  aws s3 sync output_dataset s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${output_dataset}
fi

run_parser
filter
append_to_original

# upload the new dataset to S3
aws s3 sync output_dataset s3://almond-research/${dataset_owner}/dataset/${project}/${experiment}/${output_dataset}



