# Upload all pipelines to kubeflow

# The following cell will upload all pipelines to kubeflow.
# You should see your newly uploaded pipelines or their updated versions at https://kubeflow.research.almond.stanford.edu/_/pipeline.
# To run the pipeline, make sure you switch the namespace to `research-kf`.


export THINGPEDIA_DEVELOPER_KEY= # paste your developer key here

python3 upload_pipeline.py generate_train_eval_pipeline generate-train-eval
python3 upload_pipeline.py train_eval_only_pipeline train-eval
python3 upload_pipeline.py generate_paraphrase_train_eval_pipeline generate-paraphrase-train-eval
python3 upload_pipeline.py generate_train_fewshot_eval_pipeline generate-train-fewshot-eval
python3 upload_pipeline.py generate_paraphrase_train_fewshot_eval_pipeline generate-paraphrase-train-fewshot-eval
python3 upload_pipeline.py eval_only_pipeline eval-only
python3 upload_pipeline.py selftrain_pipeline selftrain
python3 upload_pipeline.py selftrain_nopara_pipeline selftrain-nopara
python3 upload_pipeline.py paraphrase_train_eval_pipeline paraphrase-train-eval


# export CONTAINER_IMAGE=932360549041.dkr.ecr.us-west-2.amazonaws.com/masp:0.1
# python3 upload_pipeline.py masp_train_pipeline masp-train


export THINGPEDIA_DEVELOPER_KEY= # leave this empty
# python3 upload_pipeline.py train_eval_trade train-trade
# python3 upload_pipeline.py train_eval_sumbt train-sumbt
# python3 upload_pipeline.py eval_sumbt eval-sumbt
# python3 upload_pipeline.py train_eval_simpletod train-simpletod