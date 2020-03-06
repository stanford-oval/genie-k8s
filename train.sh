#!/bin/bash

. config
. lib.sh


parse_args "$0" "experiment dataset model load_from" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE PROJECT TRAIN_TASK_NAME"

load_from=$(check_aws $load_from)

if test  ${load_from} = 'None'; then
	echo "** Wrong aws link"
	echo "** The pre-trained model won't be loaded**"
fi

JOB_NAME=${OWNER}-train-${experiment}-${model}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --task_name \"${TRAIN_TASK_NAME}\" --project ${PROJECT} --experiment $experiment --dataset $dataset --model $model --load_from $load_from -- "$(requote "$@")

set -e
set -x
replace_config train.yaml.in > train.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f train.yaml
