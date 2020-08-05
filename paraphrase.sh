#!/bin/bash

. config
. lib.sh


parse_args "$0" "skip_generation=false skip_filtering=false keep_original_duplicates=false ignore_context=false experiment input_dataset output_dataset filtering_model=null paraphrasing_model_full_path=None" "$@"
shift $n
check_config "S3_BUCKET OWNER DATASET_OWNER IMAGE PROJECT TRAIN_TASK_NAME"


JOB_NAME=${OWNER}-paraphrase-${experiment}-${output_dataset}
cmdline="--s3_bucket ${S3_BUCKET} --owner ${OWNER} --dataset_owner ${DATASET_OWNER} --project ${PROJECT} --experiment $experiment \
         --input_dataset $input_dataset --output_dataset $output_dataset --filtering_model $filtering_model \
         --paraphrasing_model_full_path $paraphrasing_model_full_path --skip_generation $skip_generation \
         --skip_filtering $skip_filtering --keep_original_duplicates $keep_original_duplicates \
         --task_name ${TRAIN_TASK_NAME} --ignore_context ${ignore_context} -- "$(requote "$@")

set -e
set -x
replace_config paraphrase.yaml.in > paraphrase.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f paraphrase.yaml
