# Getting started with kubeflow pipeline

In this tutorial, we will go through a run of a typical machine learning pipeline in OVAL projects: generate dataset, train, and evaluate.
Then we will show how to add a new new pipeline to kubeflow and required changes to enable a project to run in kubeflow.

## Prerequisites:

You will need to request a first time login credentials from kubeflow admins and setup your password at

  https://kubeflow.research.almond.stanford.edu

Once your account is created, you will have :
* an isolated namespace to run Jupyter notebook and access to `s3://geniehai/`.
* permissions to create and run pipelines in `research-kf` pipeline.


## Run a pipeline
In this example, we will use run the existing pipeline `generate-train-evaluate` on the project `thingpedia-common-devices`.
Pipeline must run in t`research-kf` namespace, which has proper resources and permissions configured.  

1. Go to kubeflow dashboard and select `research-kf` on the drop down menu at upper left corner.
2. Click on `Pipelines` on the left side bar .
3. Select `generate-train-eval` and click on `Create run`
4. Choose the pipeline version. By default, the latest version is used.
5. Enter the run name. The names are in the format `${owner}/${model_name}`, eg. `gcampax/55`.
6. Choose an experiment. The names are in the format `${project}-${experiment}`, eg. `mario-main`, `schemaorg-restaurants`.
7. Review the default params. `image` is the executing docker container for the pipeline.  `genienlp_version`, `genine_version`, and `thingtalk_version` will be updated to the specified versions during runtime.
8. Set `workdir_repo` to your project git repo. In this example we will use, `git@github.com:stanford-oval/thingpedia-common-devices.git`
9. `workdir_version` should be the commit hash in your wip branch. For example, `0db4d113bd2436e85f7dfa7542f800106485f7a8`.
10. Additional generate dataset args can be defined in `generate_dataset_addtional_args`.  In this example, we want to generate a small test dataset. So we set the args to `subdatasets=1 target_pruning_size=25 max_turns=2 debug_level=2`.
11. Additional train args can be defined in `train_addtional_args`.  In this example, we want to do a quick training with 3 iterations: `--train_iterations 3 --save_every 1 --log_every 1 --val_every 1`
12. Additional eval args can be defined in `eval_additional_args`. We don't have any additional args in this example. So we leave it blank.
13. Click start to run your pipeline.
14. Click on `Experiments` on the side bar. Go to your experiment. You should see your run. This example run usually finishes in 1 hour.


## Modify or create a new pipeline

The source of the kubeflow pipeline are on `genie-k8s` repo. After modifying or creating a pipeline, you can submit it in a kubeflow Jupyter session, which has access to the kubeflow cluster.

### Prerequisites: 

#### Setup a Jupyter notebook
1. Go to the sidebar and click on `Notebook Servers`
2. Make sure the namespace is in your own namespace.
3. Click on `New server`, enter the name of your server, and use the default configurations. Note, do not use GPU for notebook servers.
4. Click on 'Launch' to start the server.

#### Setup `genie-k8s` directory 
Open a new terminal in Jupyter notebook and clone the `genie-k8s` repository

```
git config --global credentials.helper store
git clone https://<username>@github.com/stanford-oval/genie-k8s.git
```

### Modify or create a pipeline
 
Kubeflow pipelines are under `genie-k8s/kubeflow`. You can modify an existing pipeline such as `generate_train_eeval.py` or create a new one.

### To submit the pipeline

Run the following python code in Jupyter notebook:

```
import generate_train_eval 
from utils import upload_pipeline

resp = upload_pipeline('train', generate_train_eval.train_pipeline)
print(resp)
```

Go to https://kubeflow.research.almond.stanford.edu/_/pipeline/?ns=research-kf to see your newly submitted pipeline.


### Enable your project for kubeflow:

The Makefiles in current projects should work with kubeflow as is. We recommend to output eval results to kubeflow UI. To do so, create an addtional make target that writes results to the following local files:

   * '/tmp/mlpipeline-ui-metadata.json' to define the output viewers for eval results. In this example, we created csv table viewer for dialogue and nlu results. More output viewers are available at: https://www.kubeflow.org/docs/pipelines/sdk/output-viewer .
   * `/tmp/mlpipeline-metrics.json` defines the metrics to describe the performance of the model. In this example, we export only the most important metrics such as `first turn exact match` and `turn by turn exact match` accuracies. More info about metrics is availabel here: https://www.kubeflow.org/docs/pipelines/sdk/pipelines-metrics/. 
   * Export eval results to `s3_model_dir`. The eval results will be uploaded to the s3 dir passed in from the commandline.
   * Export eval results as an artifact. The same results will be uploaded as kubeflow artifact. We simply copy the results to the local artifacts directory from commandline.
   * All the changes in the example are in in the following PRs: https://github.com/stanford-oval/thingpedia-common-devices/pull/193 and https://github.com/stanford-oval/thingpedia-common-devices/pull/194.
  

# References
* An overview of Kubeflow pipeline https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/
* `generate_train_eval.py` defines the DAG workflow of the training pipeline along with input parameters and resource requirements. It is defined in python based domain specific language (DSL). More details are available here: https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.dsl.html
*  Details of the components are defined in yaml files in `components` directory.  The component specification are defined here: https://www.kubeflow.org/docs/pipelines/reference/component-spec

