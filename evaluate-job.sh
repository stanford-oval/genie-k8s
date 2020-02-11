#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner project experiment dataset model" "$@"
shift $n

set -e
set -x

on_error () {
	# on failure ship everything to s3
	aws s3 sync . s3://almond-research/${owner}/${workdir}
}
trap on_error ERR

pwd
aws s3 sync s3://almond-research/${owner}/workdir/${project} .
mkdir -p ${experiment}/models
aws s3 sync s3://almond-research/${owner}/models/${project}/${experiment}/${model}/ ${experiment}/models/${model}/

ls -al
mkdir -p tmp
export GENIE_TOKENIZER_ADDRESS=tokenizer.default.svc.cluster.local:8888
export TZ=America/Los_Angeles
make geniedir=/opt/genie-toolkit "project=${project}" "experiment=${experiment}" "owner=${owner}" "model=${model}" "$@" evaluate
#cat model/*.results > ${experiment}-${dataset}-${model}.results
#aws s3 cp ${experiment}-${dataset}-${model}.results s3://almond-research/${owner}/${workdir}/
make syncup
