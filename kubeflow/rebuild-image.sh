#!/bin/bash

export AWS_PROFILE

aws ecr get-login --no-include-email | bash

cp ../lib.sh .

set -ex

COMMON_IMAGE=$1
IMAGE=$2
JUPYTER_IMAGE=$3

genienlp_version=${genienlp_version:-master}
thingtalk_version=${thingtalk_version:-master}
genie_version=${genie_version:-master}
bootleg_version=${bootleg_version:-master}
add_bootleg=${add_bootleg:-false}


docker pull ${COMMON_IMAGE}
docker build -t ${IMAGE} \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg GENIENLP_VERSION=${genienlp_version} \
  --build-arg THINGTALK_VERSION=${thingtalk_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  --build-arg BOOTLEG_VERSION=${bootleg_version} \
  --build-arg ADD_BOOTLEG=${add_bootleg} \
  .
docker push ${IMAGE}

if test -n "${JUPYTER_IMAGE}" ; then
  docker build -t ${JUPYTER_IMAGE} \
    --build-arg BASE_IMAGE=${IMAGE} \
    -f Dockerfile.jupyter \
    .
  docker push ${JUPYTER_IMAGE}
fi

rm lib.sh
