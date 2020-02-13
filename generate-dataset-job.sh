#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner project experiment dataset" "$@"
shift $n

set -e
set -x

mkdir workdir
cd workdir

#on_error () {
	# on failure ship everything to s3
#	aws s3 sync ./ s3://almond-research/${owner}/workdir/${project}/
#}
#trap on_error ERR

pwd
aws s3 sync s3://almond-research/${owner}/workdir/${project}/ .
#mkdir -p tmp
#aws s3 sync s3://almond-research/${owner}/tmp/ tmp/

export GENIE_TOKENIZER_ADDRESS=tokenizer.default.svc.cluster.local:8888
export TZ=America/Los_Angeles
make -j geniedir=/opt/genie-toolkit experiment=${experiment} owner=${owner} "$@" datadir

#aws s3 sync tmp/ s3://almond-research/${owner}/tmp/
aws s3 sync datadir/ s3://almond-research/${owner}/dataset/${project}/${experiment}/${dataset}/
