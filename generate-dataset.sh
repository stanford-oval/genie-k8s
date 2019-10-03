#!/bin/bash

. config

set -e
set -x

experiment=$1
dataset=$2

JOB_NAME=${OWNER}-generate-dataset-${experiment}-${dataset}

sed \
  -e "s|@@JOB_NAME@@|${JOB_NAME}|g" \
  -e "s|@@OWNER@@|${OWNER}|g" \
  -e "s|@@IAM_ROLE@@|${IAM_ROLE}|g" \
  -e "s|@@IMAGE@@|${IMAGE}|g" \
  -e "s|@@experiment@@|${experiment}|g" \
  -e "s|@@dataset@@|${dataset}|g" \
  -e "s|@@dataset@@|${dataset}|g" \
  generate-dataset.yaml.in > generate-dataset.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f generate-dataset.yaml
