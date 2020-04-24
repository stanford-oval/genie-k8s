# Kubernetes scripts for Genie

These scripts are useful to generate datasets and train models with Genie
on a Kubernetes cluster. They are taylored towards our internal use, but
maybe you'll find them useful too!

This is not officially released software. Please **do not file** issues, sorry.

## Setup

Copy `config.sample` to `config`, and edit the data in as described in the wiki.

## Spawning a job

Use:
```
./generate-dataset.sh --experiment <experiment> --dataset <dataset-name>
./train.sh --experiment <experiment> --dataset <dataset-name> --model <model-name> --load_from <[load_path|None]>
./evaluate.sh --experiment <experiment> --dataset <dataset-name> --model <model-name>
./paraphrase.sh --experiment <experiment> --input_dataset <input-dataset-name> --output_dataset <output-dataset-name> --filtering_model <parser-name> --paraphrasing_model_full_path <path-to-paraphrasing-model>
```
(use `--help` to learn all commandline options; all options are required)

Experiment should be one of `thingtalk`, `spotify`, `multiwoz`.
Dataset and model name should be anything, and will be mapped to subdirectories
of your S3 research directory.
