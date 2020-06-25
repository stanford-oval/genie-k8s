#!/bin/bash

. config
. lib.sh


parse_args "$0" "experiment input_splits tgt_lang=None dlg_side=user process_translations=false num_gpus=1 ignore_context=false " "$@"
shift $n
check_config "S3_BUCKET OWNER DATASET_OWNER IMAGE PROJECT TRAIN_TASK_NAME"

GPU_NUM=$num_gpus
GPU_TYPE="p3.$((2*$num_gpus))xlarge"

JOB_NAME=${OWNER}-translate-${experiment}
cmdline="--s3_bucket ${S3_BUCKET} --owner ${OWNER} --dataset_owner ${DATASET_OWNER} --project ${PROJECT} --experiment $experiment \
         --input_splits $input_splits  --dlg_side $dlg_side \
         --process_translations $process_translations --tgt_lang $tgt_lang \
         --task_name ${TRAIN_TASK_NAME} --ignore_context ${ignore_context} -- "$(requote "$@")

set -e
set -x
replace_config translate.yaml.in > translate.yaml

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -f translate.yaml
