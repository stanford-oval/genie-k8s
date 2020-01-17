#!/bin/bash

. lib.sh
. config
check_config "IMAGE COMMON_IMAGE genie_version thingtalk_version"

set -e
set -x

#podman build -t ${COMMON_IMAGE} \
#  --build-arg THINGPEDIA_DEVELOPER_KEY=${THINGPEDIA_DEVELOPER_KEY} \
#  -f Dockerfile.common .
#podman push ${COMMON_IMAGE}

podman build -t ${IMAGE} \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg THINGTALK_VERSION=${thingtalk_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  .
podman push ${IMAGE}
