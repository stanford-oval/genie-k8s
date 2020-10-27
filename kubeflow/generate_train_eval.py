import os
from datetime import datetime

import kfp
from kfp import dsl
from kfp import components

from kubernetes.client import V1Toleration, V1Affinity
from kubernetes.client.models import (
    V1VolumeMount,
    V1Volume,
    V1PersistentVolumeClaimVolumeSource,
    V1SecretVolumeSource
)
from kubernetes import client as k8s_client

from utils import upload_pipeline
from utils import add_env

# Get the default container image from environment variable
default_image = os.environ['CONTAINER_IMAGE']
default_developer_key = os.environ['THINGPEDIA_DEVELOPER_KEY']

def add_ssh_volume(op):
    op.add_volume(V1Volume(name='ssh-v',
        secret=V1SecretVolumeSource(secret_name='ssh-secrets-k425k8d8h8', default_mode=0o600)))
    op.container.add_volume_mount(V1VolumeMount(name='ssh-v', mount_path='/root/.ssh'))
    return op

@dsl.pipeline(
    name='Generate, train and eval',
    description='Runs the whole training pipeline'
)
def generate_train_eval_pipeline(
    owner,
    dataset_owner,
    project,
    experiment,
    model,
    dataset,
    s3_bucket='geniehai',
    image=default_image,
    genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
    genie_version='84877f2488a0d0dea1e81f3e1f0b92dc6c05c568',
    thingtalk_version='755c3d23951abad9ea92cb6329ca5250489a9897',
    workdir_repo='git@github.com:stanford-oval/thingpedia-common-devices.git',
    workdir_version='0db4d113bd2436e85f7dfa7542f800106485f7a8',
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    eval_set='dev',
    eval_additional_args=''
):

    repo_versions = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
        'RUN_ID': kfp.dsl.RUN_ID_PLACEHOLDER,
        'EXECUTION_ID': kfp.dsl.EXECUTION_ID_PLACEHOLDER,
    }
    generate_dataset_op = components.load_component_from_file('components/generate-dataset.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=dataset_owner,
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
    (add_env(add_ssh_volume(generate_dataset_op), repo_versions)
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
    (add_env(add_ssh_volume(train_op), train_repos)
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
            s3_model_dir=train_op.outputs['s3_model_dir'],
            additional_args=eval_additional_args)
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))
    (add_env(add_ssh_volume(eval_op), repo_versions)
        .after(train_op)
    )

@dsl.pipeline(
    name='Train and eval',
    description='Trains and evaluate on an existing dataset'
)
def train_eval_only_pipeline(
    owner,
    dataset_owner,
    project,
    experiment,
    model,
    s3_datadir,
    s3_bucket='geniehai',
    image=default_image,
    genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
    genie_version='84877f2488a0d0dea1e81f3e1f0b92dc6c05c568',
    thingtalk_version='755c3d23951abad9ea92cb6329ca5250489a9897',
    workdir_repo='git@github.com:stanford-oval/thingpedia-common-devices.git',
    workdir_version='0db4d113bd2436e85f7dfa7542f800106485f7a8',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    eval_set='dev',
    eval_additional_args=''
):

    repo_versions = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'RUN_ID': kfp.dsl.RUN_ID_PLACEHOLDER,
        'EXECUTION_ID': kfp.dsl.EXECUTION_ID_PLACEHOLDER,
    }

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
            model=model,
            load_from=train_load_from,
            s3_datadir=s3_datadir,
            additional_args=train_additional_args)
    (train_op.container
        .set_memory_request('56Gi')
        .set_memory_limit('56Gi')
        .set_cpu_request('7.5')
        .set_cpu_limit('7.5')
        .set_gpu_limit(str(train_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (add_env(add_ssh_volume(train_op), train_repos)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*train_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')))
    )

    eval_op = components.load_component_from_file('components/evaluate.yaml')(
            image=image,
            project=project,
            experiment=experiment,
            model=model,
            model_owner=owner,
            eval_set=eval_set,
            s3_model_dir=train_op.outputs['s3_model_dir'],
            additional_args=eval_additional_args)
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))
    (add_env(add_ssh_volume(eval_op), repo_versions)
        .after(train_op)
    )
