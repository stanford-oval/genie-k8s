#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment model" "$@"
shift $n
check_config "IAM_ROLE OWNER IMAGE PROJECT"

JOB_NAME=${OWNER}-evaluate-${experiment}-${model}
cmdline="--owner ${OWNER} --project ${PROJECT} --experiment ${experiment} --model ${model} -- "$(requote "$@")

set -e
set -x
replace_config evaluate.yaml.in > evaluate.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f evaluate.yaml
