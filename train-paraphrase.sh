#!/bin/bash

. config
. lib.sh


parse_args "$0" "experiment=paraphrase dataset model" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE PROJECT TRAIN_TASK_NAME"


JOB_NAME=${OWNER}-paraphrase-${model}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --task_name ${TRAIN_TASK_NAME} --project ${PROJECT} --experiment $experiment --dataset $dataset --model $model -- "$(requote "$@")

set -e
set -x
replace_config paraphrase.yaml.in > paraphrase.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f paraphrase.yaml
