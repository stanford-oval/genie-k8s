#!/bin/bash

. config
. lib.sh

parse_args "$0" "train_or_gen experiment dataset model" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE train_task_name"


JOB_NAME=${OWNER}-paraphrase-${experiment}-${model}
cmdline="--train_or_gen ${train_or_gen} --owner ${OWNER} --dataset_owner ${DATASET_OWNER} --task_name ${train_task_name} --experiment $experiment --dataset $dataset --model $model -- "$(requote "$@")

set -e
set -x
replace_config paraphrase.yaml.in > paraphrase.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f paraphrase.yaml
