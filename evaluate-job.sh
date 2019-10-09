#!/bin/bash

. lib.sh

parse_args "$0" "owner experiment dataset model workdir" "$@"
shift $n

set -e
set -x

on_error () {
	# on failure ship everything to s3
	aws s3 sync . s3://almond-research/${owner}/${workdir}
}
trap on_error ERR

pwd
aws s3 sync s3://almond-research/${owner}/${workdir} .
aws s3 sync s3://almond-research/${owner}/models/${experiment}${model} model/

ls -al
mkdir tmp
make experiment=${experiment} owner=${owner} "$@" evaluate
cat model/*.results > ${experiment}-${dataset}-${model}.results
aws s3 cp ${experiment}-${dataset}-${model}.results s3://almond-research/${owner}/${workdir}/
