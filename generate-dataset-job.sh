#!/bin/bash

set -e
set -x

owner=$1
experiment=$2
dataset=$3
shift
shift
shift

on_error () {
	# on failure ship everything to s3
	aws s3 sync . s3://almond-research/${owner}/workdir
}
trap on_error ERR

pwd
aws s3 sync s3://almond-research/${owner}/workdir .

ls -al
mkdir tmp
make experiment=${experiment} owner=${owner} "$@" datadir
aws s3 sync datadir/ s3://almond-research/${owner}/dataset/${experiment}${dataset}
