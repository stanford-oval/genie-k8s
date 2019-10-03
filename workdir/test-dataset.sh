#!/bin/bash

set -e
set -o pipefail

dataset=$1
dataset_basename=$(basename $dataset)
models=$2

for model in `cat $models` ; do
	if ! test -f $model/$dataset_basename.results ; then
		./test-model.sh $model $dataset > $model/$dataset_basename.results.tmp
		mv $model/$dataset_basename.results.tmp $model/$dataset_basename.results
	fi
	cat $model/$dataset_basename.results
done
