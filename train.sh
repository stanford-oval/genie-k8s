#!/bin/bash

. config
. lib.sh


parse_args "$0" "experiment dataset model load_from=None" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE PROJECT TRAIN_TASK_NAME"

JOB_NAME=${OWNER}-train-${experiment}-${model}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --task_name \"${TRAIN_TASK_NAME}\" --project ${PROJECT} --experiment $experiment --dataset $dataset --model $model --load_from \"${load_from}\" -- "$(requote "$@")

set -e
set -x
replace_config train.yaml.in > train.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f train.yaml
