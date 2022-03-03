# Upload all pipelines to kubeflow

# The following cell will upload all pipelines to kubeflow.
# You should see your newly uploaded pipelines or their updated versions at https://kubeflow.research.almond.stanford.edu/_/pipeline.
# To run the pipeline, make sure you switch the namespace to `research-kf`.


export THINGPEDIA_DEVELOPER_KEY= # paste your developer key here

python3 upload_pipeline.py generate_train_eval_pipeline generate-train-eval
python3 upload_pipeline.py train_eval_pipeline train-eval
python3 upload_pipeline.py bootleg_only_pipeline bootleg-only
python3 upload_pipeline.py generate_bootleg_train_eval_pipeline generate-bootleg-train-eval
python3 upload_pipeline.py bootleg_train_eval_pipeline bootleg-train-eval
python3 upload_pipeline.py generate_paraphrase_train_eval_pipeline generate-paraphrase-train-eval
python3 upload_pipeline.py gpu_generate_paraphrase_train_eval_pipeline gpu-generate-paraphrase-train-eval
python3 upload_pipeline.py gpu_generate_bootleg_paraphrase_train_eval_pipeline gpu-generate-bootleg-paraphrase-train-eval
python3 upload_pipeline.py generate_train_fewshot_eval_pipeline generate-train-fewshot-eval
python3 upload_pipeline.py generate_paraphrase_train_fewshot_eval_pipeline generate-paraphrase-train-fewshot-eval
python3 upload_pipeline.py eval_only_pipeline eval-only
python3 upload_pipeline.py selftrain_pipeline selftrain
python3 upload_pipeline.py selftrain_nopara_pipeline selftrain-nopara
python3 upload_pipeline.py paraphrase_train_eval_pipeline paraphrase-train-eval
python3 upload_pipeline.py paraphrase_only_pipeline paraphrase-only
python3 upload_pipeline.py paraphrase_filtering_pipeline paraphrase-filtering-only
python3 upload_pipeline.py bootleg_paraphrase_train_eval_pipeline bootleg-paraphrase-train-eval
python3 upload_pipeline.py generate_bootleg_train_fewshot_calibrate_eval_pipeline generate-bootleg-train-fewshot-calibrate-eval
python3 upload_pipeline.py predict_pipeline predict
python3 upload_pipeline.py predict_small_pipeline predict-small
python3 upload_pipeline.py train_predict_small_pipeline train-predict-small
python3 upload_pipeline.py train_predict_e2e_dialogue_pipeline train-predict-e2e-dialogue
python3 upload_pipeline.py predict_e2e_dialogue_pipeline predict-e2e-dialogue
python3 upload_pipeline.py translate_dialogue_pipeline translate-dialogue
python3 upload_pipeline.py translate_train_eval_pipeline translate-train-eval
python3 upload_pipeline.py generate_translate_train_eval_pipeline generate-translate-train-eval

# export CONTAINER_IMAGE=932360549041.dkr.ecr.us-west-2.amazonaws.com/masp:0.1
# python3 upload_pipeline.py masp_train_pipeline masp-train


export THINGPEDIA_DEVELOPER_KEY= # leave this empty
# python3 upload_pipeline.py train_eval_trade train-trade
# python3 upload_pipeline.py train_eval_sumbt train-sumbt
# python3 upload_pipeline.py eval_sumbt eval-sumbt
# python3 upload_pipeline.py train_eval_simpletod train-simpletod
