#!/bin/bash

export AWS_PROFILE

aws ecr get-login --no-include-email | bash

. ../config

cp ../lib.sh .
. lib.sh

check_config "IMAGE COMMON_IMAGE genie_version thingtalk_version bootleg_version genienlp_version "

set -ex

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

docker system prune -f