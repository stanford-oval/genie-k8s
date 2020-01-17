#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment dataset model" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE train_task_name"

JOB_NAME=${OWNER}-train-${experiment}-${model}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --task_name ${train_task_name} --experiment $experiment --dataset $dataset --model $model -- "$(requote "$@")

set -e
set -x
replace_config train.yaml.in > train.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f train.yaml
