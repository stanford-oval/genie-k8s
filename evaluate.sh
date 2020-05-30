#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment model model_owner eval_set=eval" "$@"
shift $n
check_config "IAM_ROLE OWNER IMAGE PROJECT"

JOB_NAME=${OWNER}-evaluate-${experiment}-${model_owner}-${model}
cmdline="--owner ${OWNER} --project ${PROJECT} --experiment ${experiment} --model ${model} --model_owner ${model_owner} --eval_set ${eval_set} -- "$(requote "$@")

set -e
set -x
replace_config evaluate.yaml.in > evaluate.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f evaluate.yaml
