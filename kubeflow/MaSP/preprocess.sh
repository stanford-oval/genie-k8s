#!/bin/bash

mkdir -p data
curl https://zenodo.org/record/3268649/files/CSQA_v9.zip > CSQA_v9.zip
unzip -q CSQA_v9.zip -d .
mv CSQA_v9 data/CSQA
rm -rf CSQA_v9.zip
curl https://zenodo.org/record/4052427/files/wikidata_proc_json.zip > wikidata_proc_json.zip
unzip -q wikidata_proc_json.zip -d .
mv "wikidata_proc_json 2" data/kb
rm -rf wikidata_proc_json.zip
curl https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip  > bert_base.zip
unzip -q bert_base.zip -d .
mv "uncased_L-12_H-768_A-12" bert_base
rm -rf bert_base.zip
python3 ./main_bfs_preprocess.py