# Kubeflow pipelines for Genie

These scripts are useful to generate datasets and train models with Genie
on a [Kubeflow](https://kubeflow.org) cluster. They are tailored towards our
internal use, but maybe you'll find them useful too!

This is not officially released software. Please **do not file** issues, sorry.

## Setup

Copy `config.sample` to `config`, and edit the data in as described inside.

## Building the docker image

The docker images are split into a base image and a regular Kubeflow image,
which is used by each step. There is an additional image that includes the
Jupyter notebook code.

To build the base image, set the `COMMON_IMAGE` field in the config file, then
run `./rebuild-common-image.sh`.

To build the regular image, use `./rebuild-image.sh`. See `./rebuild-image.sh --help`
for options.

The scripts will build the images and upload to ACR (Azure Container Registry).

## Uploading pipelines

Use:

```bash
python3 upload_pipeline.py ${pipeline_function} ${pipeline_name}
```

to upload a pipeline to your Kubeflow cluster. `${pipeline_function}` should be
the name of a Python function in `pipelines.py`.

You can also upload all pipelines at the same time via `upload_all_pipelines.sh`.
You need to set the following keys before running the script: `THINGPEDIA_DEVELOPER_KEY`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.

## Running jobs

Refer to the Kubeflow documentation to learn how to run jobs after uploading the pipeline.
