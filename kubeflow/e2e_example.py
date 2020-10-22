import os
from datetime import datetime

import kfp
from kfp import dsl
from kfp import components

from kubernetes.client import V1Toleration, V1Affinity
from kubernetes.client.models import V1VolumeMount, V1Volume, V1PersistentVolumeClaimVolumeSource
from kubernetes import client as k8s_client

from utils import upload_pipeline
from utils import add_env

# Get the default container image from environment variable
default_image = os.environ.get('CONTAINER_IMAGE', '')

@dsl.pipeline(
    name='E2E Training pipeline',
    description='Runs the whole training pipeline'
)  
def train_pipeline(
    s3_bucket='geniehai',
    image=default_image,
    owner='jgd5',
    dataset_owner='jgd5',
    project='almond',
    genienlp_version='0d291829b94bef6287c30d592b83412aaf0b0d86',
    genie_version='cf078a09ca5e891562f22fc6e12eca111c5d103e',
    thingtalk_version='0a8688a20ccc292f26e49247c0dad810103e6c78',
    workdir_repo='git@github.com:stanford-oval/thingpedia-common-devices.git',
    workdir_version='07e690fade3576b17d721b54fe4df8720e358903',
    workdir_s3_config_dir='s3://geniehai/jgd5/config/almond',
    experiment='main',
    model='model',
    dataset='e2e_test_dataset',
    generate_dataset_parallel='6',
    generate_dataset_additional_args='subdatasets=1 target_pruning_size=25 max_turns=2 debug_level=2',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='--train_iterations 3 --save_every 1 --log_every 1 --val_every 1',
    eval_set='dev',
    eval_version='None',
    eval_additional_args=''
):

    repo_versions = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'WORKDIR_S3_CONFIG_DIR': workdir_s3_config_dir,
    }
    generate_dataset_op = components.load_component_from_file('components/generate-dataset.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            project=project,
            experiment=experiment,
            dataset=dataset,
            parallel=generate_dataset_parallel,
            additional_args=generate_dataset_additional_args)
    (generate_dataset_op.container
        .set_memory_limit('55Gi')
        .set_memory_request('55Gi')
        .set_cpu_limit('15.5')
        .set_cpu_request('15.5')
    )
    (add_env(generate_dataset_op, repo_versions)
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'm5.4xlarge')
    )
   
    train_repos = repo_versions.copy()
    train_repos.pop('WORKDIR_REPO')
    train_repos.pop('WORKDIR_VERSION')
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            dataset_owner=dataset_owner,
            task_name=train_task_name,
            project=project,
            experiment=experiment,
            dataset=dataset,
            model=model,
            load_from=train_load_from,
            s3_datadir=generate_dataset_op.outputs['s3_datadir'],
            additional_args=train_additional_args) 
    (train_op.container
        .set_memory_request('56Gi')
        .set_memory_limit('56Gi')
        .set_cpu_request('7.5')
        .set_cpu_limit('7.5')
        .set_gpu_limit(str(train_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (add_env(train_op, train_repos)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*train_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')))
        .after(generate_dataset_op)
    )
    
    eval_op = components.load_component_from_file('components/evaluate.yaml')(
            image=image,
            project=project,
            experiment=experiment,
            model=model,
            model_owner=owner,
            eval_set=eval_set,
            eval_version=eval_version,
            s3_model_dir=train_op.outputs['s3_model_dir'],
            additional_args=eval_additional_args)  
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))  
    (add_env(eval_op, repo_versions)
        .after(train_op)
    )

if __name__ == '__main__':
    resp = upload_pipeline('generate-train-eval', train_pipeline)
    print(resp)
