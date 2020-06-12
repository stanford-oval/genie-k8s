#!/bin/bash

. lib.sh
. config
check_config "IMAGE COMMON_IMAGE genie_version thingtalk_version genienlp_version"
export AWS_PROFILE

aws ecr get-login --no-include-email | bash

set -e
set -x

docker pull nvidia/cuda:10.2-runtime-ubi8
docker build -t ${COMMON_IMAGE} -f Dockerfile.common .
docker push ${COMMON_IMAGE}
