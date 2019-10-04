#!/bin/bash

. config
. lib.sh

set -e
set -x

experiment=$1
dataset=$2
model=$3

JOB_NAME=${OWNER}-evaluate-${experiment}-${dataset}-${model}

replace_config evaluate.yaml.in > evaluate.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f evaluate.yaml
