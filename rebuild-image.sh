#!/bin/bash

. lib.sh
. config
check_config "IMAGE genie_version thingtalk_version"

set -e
set -x

podman build -t ${IMAGE} \
  --build-arg THINGPEDIA_DEVELOPER_KEY=${THINGPEDIA_DEVELOPER_KEY} \
  --build-arg THINGTALK_VERSION=${thingtalk_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  .
podman push ${IMAGE}
