# Upload all pipelines to kubeflow

# The following cell will upload all pipelines to kubeflow.
# You should see your newly uploaded pipelines or their updated versions at https://kubeflow.research.almond.stanford.edu/_/pipeline.
# To run the pipeline, make sure you switch the namespace to `research-kf`.


export THINGPEDIA_DEVELOPER_KEY= # paste your developer key here
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=

python3 upload_pipeline.py generate_train_eval_pipeline generate-train-eval
python3 upload_pipeline.py train_eval_pipeline train-eval
python3 upload_pipeline.py bootleg_pipeline bootleg
python3 upload_pipeline.py generate_bootleg_train_eval_pipeline generate-bootleg-train-eval
python3 upload_pipeline.py bootleg_train_eval_pipeline bootleg-train-eval
python3 upload_pipeline.py generate_paraphrase_train_eval_pipeline generate-paraphrase-train-eval
python3 upload_pipeline.py gpu_generate_paraphrase_train_eval_pipeline gpu-generate-paraphrase-train-eval
python3 upload_pipeline.py gpu_generate_bootleg_paraphrase_train_eval_pipeline gpu-generate-bootleg-paraphrase-train-eval
python3 upload_pipeline.py generate_train_fewshot_eval_pipeline generate-train-fewshot-eval
python3 upload_pipeline.py generate_paraphrase_train_fewshot_eval_pipeline generate-paraphrase-train-fewshot-eval
python3 upload_pipeline.py eval_pipeline eval
python3 upload_pipeline.py selftrain_full_pipeline selftrain-full
python3 upload_pipeline.py selftrain_pipeline selftrain
python3 upload_pipeline.py selftrain_nopara_pipeline selftrain-nopara
python3 upload_pipeline.py paraphrase_train_eval_pipeline paraphrase-train-eval
python3 upload_pipeline.py paraphrase_pipeline paraphrase
python3 upload_pipeline.py paraphrase_filtering_pipeline paraphrase-filtering
python3 upload_pipeline.py bootleg_paraphrase_train_eval_pipeline bootleg-paraphrase-train-eval
python3 upload_pipeline.py generate_bootleg_train_fewshot_calibrate_eval_pipeline generate-bootleg-train-fewshot-calibrate-eval
python3 upload_pipeline.py predict_4gpu_pipeline predict-4gpu
python3 upload_pipeline.py predict_pipeline predict
python3 upload_pipeline.py train_predict_pipeline train-predict
python3 upload_pipeline.py train_predict_e2e_dialogue_pipeline train-predict-e2e-dialogue
python3 upload_pipeline.py predict_e2e_dialogue_pipeline predict-e2e-dialogue
python3 upload_pipeline.py translate_dialogue_pipeline translate-dialogue
python3 upload_pipeline.py translate_train_eval_pipeline translate-train-eval
python3 upload_pipeline.py generate_translate_train_eval_pipeline generate-translate-train-eval
# Add new pipelines here
