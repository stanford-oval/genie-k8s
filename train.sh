#!/bin/bash

. config
. lib.sh

check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE PROJECT"
parse_args "$0" "experiment dataset dataset_owner=${DATASET_OWNER} model task=${TRAIN_TASK_NAME} load_from=None" "$@"
shift $n

JOB_NAME=${OWNER}-train-${experiment}-${model}
cmdline="--owner ${OWNER} --dataset_owner ${dataset_owner} --task_name \"${task}\" --project ${PROJECT} --experiment $experiment --dataset $dataset --model $model --load_from \"${load_from}\" -- "$(requote "$@")

set -e
set -x
replace_config train.yaml.in > train.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f train.yaml
