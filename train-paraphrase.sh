#!/bin/bash

. config
. lib.sh


parse_args "$0" "experiment=paraphrase dataset model" "$@"
shift $n
check_config "S3_BUCKET OWNER DATASET_OWNER IMAGE PROJECT"


JOB_NAME=${OWNER}-paraphrase-${model}
cmdline="--s3_bucket ${S3_BUCKET} --owner ${OWNER} --dataset_owner ${DATASET_OWNER} --project ${PROJECT} --experiment $experiment --dataset $dataset --model $model -- "$(requote "$@")

set -e
set -x
replace_config train-paraphrase.yaml.in > train-paraphrase.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f train-paraphrase.yaml
