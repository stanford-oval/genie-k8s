#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment eval_set model" "$@"
shift $n
check_config "IAM_ROLE OWNER IMAGE"

JOB_NAME=${OWNER}-evaluate-${experiment}-${model}
cmdline="--owner ${OWNER} --experiment $experiment --model $model --workdir ${workdir} --eval_set ${eval_set} -- "$(requote "$@")

set -e
set -x
replace_config evaluate.yaml.in > evaluate.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f evaluate.yaml
