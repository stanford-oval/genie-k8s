#!/bin/bash

. config
. lib.sh

set -e
set -x

experiment=$1
dataset=$2
model=$3

JOB_NAME=${OWNER}-train-${experiment}-${dataset}-${model}

replace_config train.yaml.in > train.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f train.yaml
