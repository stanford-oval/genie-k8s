#!/bin/bash

. lib.sh
. config
check_config "IMAGE COMMON_IMAGE genie_version thingtalk_version genienlp_version"
export AWS_PROFILE

aws ecr get-login --no-include-email | bash

set -e
set -x

docker pull ${COMMON_IMAGE}
docker build -t ${IMAGE} \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg GENIENLP_VERSION=${genienlp_version} \
  --build-arg THINGTALK_VERSION=${thingtalk_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  .
docker push ${IMAGE}
