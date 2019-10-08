#!/bin/bash

. lib.sh

parse_args "$0" "owner experiment dataset model" "$@"
shift $n

set -e
set -x

mkdir workdir
cd workdir

on_error () {
	# on failure ship everything to s3
	aws s3 sync ./ s3://almond-research/${owner}/workdir/
}
trap on_error ERR

pwd
aws s3 sync s3://almond-research/${owner}/workdir/ .
mkdir tmp
aws s3 sync s3://almond-research/${owner}/tmp/ tmp/

make experiment=${experiment} owner=${owner} "$@" datadir

aws s3 sync tmp/ s3://almond-research/${owner}/tmp/
aws s3 sync datadir/ s3://almond-research/${owner}/dataset/${experiment}${dataset}/
