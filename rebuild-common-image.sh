#!/bin/bash

. lib.sh
. config
check_config "COMMON_IMAGE"

az acr login --name stanfordoval.azurecr.io

set -e
set -x

docker pull nvidia/cuda:10.2-runtime-ubi8
docker build -t ${COMMON_IMAGE} -f Dockerfile.common .
docker push ${COMMON_IMAGE}
