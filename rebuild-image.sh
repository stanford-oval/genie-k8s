#!/bin/bash

. config

set -e
set -x

podman build -t ${IMAGE} --build-arg THINGPEDIA_DEVELOPER_KEY=${THINGPEDIA_DEVELOPER_KEY} .
podman push ${IMAGE}
