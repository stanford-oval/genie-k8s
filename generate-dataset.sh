#!/bin/bash

. config
. lib.sh

set -e
set -x

experiment=$1
dataset=$2

JOB_NAME=${OWNER}-generate-dataset-${experiment}-${dataset}

replace_config generate-dataset.yaml.in > generate-dataset.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f generate-dataset.yaml
