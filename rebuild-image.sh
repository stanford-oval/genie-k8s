#!/bin/bash

. lib.sh
. config

check_config "COMMON_IMAGE genie_version genienlp_version"

parse_args "$0" "image jupyter_image=None add_apex=false" "$@"
shift $n

az acr login --name stanfordoval.azurecr.io

set -ex

genienlp_version=${genienlp_version:-master}
genie_version=${genie_version:-master}
add_apex=${add_apex:-false}

docker pull ${COMMON_IMAGE}
docker build -t ${image} \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg GENIENLP_VERSION=${genienlp_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  --build-arg ADD_APEX=${add_apex} \
  .
docker push ${image}

if test "${jupyter_image}" != None ; then
  docker build -t ${jupyter_image} \
    --build-arg BASE_IMAGE=${image} \
    -f Dockerfile.jupyter \
    .
  docker push ${jupyter_image}
fi
