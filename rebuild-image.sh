#!/bin/bash

. lib.sh
. config
check_config "IMAGE COMMON_IMAGE genie_version thingtalk_version decanlp_version"

set -e
set -x

#docker build -t ${COMMON_IMAGE} -f Dockerfile.common .
#docker push ${COMMON_IMAGE}

docker build -t ${IMAGE} \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg DECANLP_VERSION=${decanlp_version} \
  --build-arg THINGTALK_VERSION=${thingtalk_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  .
docker push ${IMAGE}
