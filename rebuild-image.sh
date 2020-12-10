#!/bin/bash

. lib.sh
. config

check_config "COMMON_IMAGE genie_version thingtalk_version bootleg_version genienlp_version"
export AWS_PROFILE

parse_args "$0" "image jupyter_image=None add_bootleg=true" "$@"
shift $n

aws ecr get-login --no-include-email | bash

set -ex

genienlp_version=${genienlp_version:-master}
thingtalk_version=${thingtalk_version:-master}
genie_version=${genie_version:-master}
bootleg_version=${bootleg_version:-master}
add_bootleg=${add_bootleg:-true}

docker pull ${COMMON_IMAGE}
docker build -t ${image} \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg GENIENLP_VERSION=${genienlp_version} \
  --build-arg THINGTALK_VERSION=${thingtalk_version} \
  --build-arg GENIE_VERSION=${genie_version} \
  --build-arg BOOTLEG_VERSION=${bootleg_version} \
  --build-arg ADD_BOOTLEG=${add_bootleg} \
  .
docker push ${image}

if test "${jupyter_image}" != None ; then
  docker build -t ${jupyter_image} \
    --build-arg BASE_IMAGE=${image} \
    -f Dockerfile.jupyter \
    .
  docker push ${jupyter_image}
fi

rm lib.sh
