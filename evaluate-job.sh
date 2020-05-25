#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "owner project experiment model" "$@"
shift $n

set -e
set -x

# on_error () {
 	# on failure ship everything to s3
# 	aws s3 sync . s3://almond-research/${owner}/models/${project}/${experiment}/${model}/failed_eval/
# }
# trap on_error ERR

pwd
aws s3 sync --include '${project}/' s3://almond-research/${owner}/workdir/${project} .
mkdir -p ${experiment}/models
aws s3 sync --exclude 'iteration_*.pth' --exclude '*_optim.pth' s3://almond-research/${owner}/models/${project}/${experiment}/${model}/ ${experiment}/models/${model}/

ls -al
mkdir -p tmp
export GENIE_TOKENIZER_ADDRESS=tokenizer.default.svc.cluster.local:8888
export TZ=America/Los_Angeles
make geniedir=/opt/genie-toolkit experiment=$experiment owner=$owner $experiment/eval/$model.results
cat $experiment/eval/$model.results
aws s3 sync $experiment/eval/ s3://almond-research/${owner}/workdir/${project}/$experiment/eval/
