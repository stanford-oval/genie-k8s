#!/bin/bash

set -e

for experiment in main ; do
	for dataset in $(cat $experiment-test-sets) ; do
		./test-dataset.sh $dataset $experiment-models
	done
done
