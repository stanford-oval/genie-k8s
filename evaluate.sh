#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment model model_owner eval_set=eval eval_version=v2" "$@"
shift $n
check_config "S3_BUCKET OWNER IMAGE PROJECT"

JOB_NAME=${OWNER}-evaluate-${experiment}-${model_owner}-${model}
cmdline="--s3_bucket ${S3_BUCKET} --owner ${OWNER} --project ${PROJECT} --experiment ${experiment} --model ${model} --model_owner ${model_owner} --eval_set ${eval_set} --eval_version ${eval_version} -- "$(requote "$@")

set -e
set -x
replace_config evaluate.yaml.in > evaluate.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f evaluate.yaml
