#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment dataset" "$@"
shift $n
check_config "IAM_ROLE OWNER IMAGE project"

JOB_NAME=${OWNER}-gen-dataset-${experiment}-${dataset}
cmdline="--owner ${OWNER} --project $project --experiment $experiment --dataset $dataset -- "$(requote "$@")

set -e
set -x
replace_config generate-dataset.yaml.in > generate-dataset.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f generate-dataset.yaml
