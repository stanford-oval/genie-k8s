#!/bin/bash

. config
. lib.sh

parse_args "$0" "experiment dataset model" "$@"
shift $n

JOB_NAME=${OWNER}-train-${experiment}-${dataset}-${model}
cmdline="--owner ${owner} --experiment $experiment --dataset $dataset --model $model "$(requote "$@")

set -e
set -x
replace_config train.yaml.in > train.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f train.yaml
