# Getting started on kubeflow pipeline

The easiest way is to submit the pipeline inside the kubeflow cluster throguh a Jupyter notebook. Here are the steps.




## 1. Create a kubeflow account

Go to https://kubeflow.research.almond.stanford.edu to create a user profile with the given credentials from admin.



## 2. Setup Jupyter notebook server
* Each user will have an isolated namespace to run Jupyter notebook.
* From the sidebar bar on the left, click on `Notebook Servers`
* Click on `NEW SERVER`. Then fill out the name and select any one of the prefilled images to start the server. A customized image will be provided in the futre. 


# 3. Git clone `genie-k8s`

Open a new terminal in Juypter notebook and clone the `genie-k8s` repository

```
git clone https://<username>@github.com/stanford-oval/genie-k8s.git
```

# 4. Submit a pipeline

Navigate to `/genie-k8s/kubeflow` and open `upload_pipeline.ipynb`. Run the  cell to upload an example pipeline.

# 5. Run the example pipeline

Pipeline must run in t`research-kf` namespace, which has proper permissions configured.
* Go to kubeflow dashboard, on the drop down menu at upper left corner, 
select `research-kf`. 
* Go to pipelines, click on `train` and `Create run`
* Version should be the latest version you just uploaded
* Choose an experiment
* Enter `kf-user` for Service Account (this gives S3 and possibly other permissions to run your job)
* Review the default params and make changes 
* Click `Start` 

The default only trains for 3 iterations. You can modify the generate dataset and train flags in `e2e_example.py` for a longer training.

# References
* An overview of Kubeflow pipeline https://www.kubeflow.org/docs/pipelines/sdk/sdk-overview/
* `e2e_example.py` defines the DAG workflow of the training pipeline along with input parameters and resource requirements. It is defined in python based domain specific language (DSL). More details are available here: https://kubeflow-pipelines.readthedocs.io/en/latest/source/kfp.dsl.html
*  Details of the components are defined in yaml files in `components` directory.  The component specification are defined here: https://www.kubeflow.org/docs/pipelines/reference/component-spec

