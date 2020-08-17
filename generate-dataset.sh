#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment dataset parallel=6" "$@"
shift $n
check_config "S3_BUCKET OWNER IMAGE PROJECT"

mod_experiment=$(echo ${experiment} | tr '_' '-' | tr A-Z a-z)

JOB_NAME=${OWNER}-gen-dataset-${mod_experiment}-${dataset}
cmdline="--s3_bucket ${S3_BUCKET} --owner ${OWNER} --project ${PROJECT} --experiment $experiment --dataset $dataset --parallel $parallel -- "$(requote "$@")

set -e
set -x
replace_config generate-dataset.yaml.in > generate-dataset.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f generate-dataset.yaml
