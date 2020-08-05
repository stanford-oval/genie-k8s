#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "s3_bucket owner project experiment model model_owner eval_set eval_version" "$@"
shift $n

set -e
set -x

# on_error () {
 	# on failure ship everything to s3
# 	aws s3 sync . s3://${s3_bucket}/models/${project}/${experiment}/${model}/failed_eval/
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
if [ "$eval_version" = "None" ] ; then
	make geniedir=/opt/genie-toolkit thingpedia_cli=thingpedia experiment=$experiment eval_set=${eval_set} ${experiment}/${eval_set}/${model_owner}/${model}.{nlu,dialogue}.results
	for f in $experiment/${eval_set}/${model_owner}/${model}.{nlu,dialogue}.{results,debug} ; do
		aws s3 cp $f s3://${s3_bucket}/${owner}/workdir/${project}/${experiment}/${eval_set}/${model_owner}/
	done
else
	echo "evaluation sets are versioned"
	make geniedir=/opt/genie-toolkit thingpedia_cli=thingpedia experiment=$experiment eval_set=${eval_set} eval_version=${eval_version} ${experiment}/${eval_set}/${eval_version}/${model_owner}/${model}.dialogue.results
	for f in $experiment/${eval_set}/${eval_version}/${model_owner}/${model}.{nlu,dialogue}.{results,debug} ; do
		aws s3 cp $f s3://${s3_bucket}/${owner}/workdir/${project}/${experiment}/${eval_set}/${eval_version}/${model_owner}/
	done
fi
