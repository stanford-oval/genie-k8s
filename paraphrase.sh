#!/bin/bash

. config
. lib.sh


parse_args "$0" "skip_generation=false experiment input_dataset output_dataset filtering_model paraphrasing_model_full_path=None \
            is_dialogue=false" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE PROJECT"


JOB_NAME=${OWNER}-paraphrase-${output_dataset}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --project ${PROJECT} --experiment $experiment \
         --input_dataset $input_dataset --output_dataset $output_dataset --filtering_model $filtering_model \
         --paraphrasing_model_full_path $paraphrasing_model_full_path --skip_generation $skip_generation \
         --is_dialogue $is_dialogue -- "$(requote "$@")

set -e
set -x
replace_config paraphrase.yaml.in > paraphrase.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f paraphrase.yaml
