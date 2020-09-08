#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "s3_bucket owner project experiment model model_owner eval_set eval_version" "$@"
shift $n

set -e
set -x

# on_error () {
 	# on failure ship everything to s3
# 	aws s3 sync . s3://${s3_bucket}/${owner}/models/${project}/${experiment}/${model}/failed_eval/
# }
# trap on_error ERR

pwd
aws s3 sync s3://${s3_bucket}/${owner}/workdir/${project} .
# mkdir -p ${experiment}/models
# aws s3 sync --exclude 'iteration_*.pth' --exclude '*_optim.pth' s3://${s3_bucket}/${owner}/models/${project}/${experiment}/${model}/ ${experiment}/models/${model}/

ls -al
mkdir -p tmp
export GENIE_TOKENIZER_ADDRESS=tokenizer.default.svc.cluster.local:8888
export TZ=America/Los_Angeles
make geniedir=/opt/genie-toolkit thingpedia_cli=thingpedia experiment=$experiment eval_set=${eval_set} eval_version=${eval_version} model=${model_owner}/${model} evaluate
make geniedir=/opt/genie-toolkit thingpedia_cli=thingpedia experiment=$experiment eval_set=${eval_set} eval_version=${eval_version} model=${model_owner}/${model} evaluate-upload
