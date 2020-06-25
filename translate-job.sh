#!/bin/bash

. /opt/genie-toolkit/lib.sh

parse_args "$0" "s3_bucket owner dataset_owner project experiment tgt_lang input_splits dlg_side process_translations input_splits ignore_context task_name" "$@"
shift $n

if [ "$ignore_context" = true ] ; then
  echo "ignoring context"
fi

set -e
set -x

export translation_arguments="$@"

run_translation(){
  # copy workdir makefiles over
  aws s3 cp --recursive s3://${s3_bucket}/${owner}/workdir/${project} ./ --exclude "*" \
   --include '*.mk' --include '*.config' --include '*/schema.tt' --include 'Makefile' --include '*.py' \
    --include "*${experiment}/${dlg_side}/en/*" --include '*dlg-shared-parameter-datasets*' --include "*dataset-dialogs/${experiment}/${dlg_side}/*"
  # generate english ref file if does not exists
  IFS='+'; read -ra input_splits_array <<<"$input_splits" ; IFS=' '
  for f in "${input_splits_array[@]}" ; do
    if ! test -s ${experiment}/${dlg_side}/en/aug-en/unquoted-qpis-marian/$f.tsv; then
      make -B geniedir=/opt/genie-toolkit all_names=$f experiment_dialog=${experiment}/${dlg_side} process_data_dialogs
      aws s3 sync ./ s3://geniehai/${owner}/workdir/${project}/ --exclude '*' --include "*${experiment}/${dlg_side}*"
    fi
    genienlp run-paraphrase \
      --id_column 0 --input_column 1 --gold_column 1 \
      --input_file ${experiment}/${dlg_side}/en/aug-en/unquoted-qpis-marian/$f.tsv \
      --output_file ${experiment}/${dlg_side}/marian/it/unquoted-qpis-translated/$f.tsv \
      --model_name_or_path Helsinki-NLP/opus-mt-en-${tgt_lang} \
      --output_example_ids_too \
      --replace_qp \
      --skip_heuristics \
      --task translate \
      $translation_arguments
    aws s3 sync ./ s3://geniehai/${owner}/workdir/${project}/ --exclude '*' --include '*unquoted-qpis-translated*'
    if ${process_translations} ; then
      make -B geniedir=/opt/genie-toolkit all_names=$f experiment_dialog=${experiment}/${dlg_side} skip_translation=true batch_translate_dlg_marian_${tgt_lang}
    fi
  done
}


run_translation





