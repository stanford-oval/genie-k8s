import os
from datetime import datetime

import kfp
from kfp import dsl
from kfp import components

from kubernetes.client import V1Toleration, V1Affinity
from kubernetes.client.models import V1VolumeMount, V1Volume, V1PersistentVolumeClaimVolumeSource
from kubernetes import client as k8s_client

from utils import disable_caching

@dsl.pipeline(
    name='E2E Training pipeline',
    description='Runs the whole training pipeline'
)  
def train_pipeline(
    s3_bucket='geniehai',
    image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-jgd5',
    owner='jgd5',
    dataset_owner='jgd5',
    project='almond',
    experiment='main',
    model='model',
    dataset='e2e_test_dataset',
):
    generate_dataset_args = 'subdatasets=1 target_pruning_size=25 max_turns=2 debug_level=2'
    # uncomment next line for longer training
    generate_dataset_args = '' 
    generate_dataset_op = components.load_component_from_file('components/generate-dataset.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            project=project,
            experiment=experiment,
            dataset=dataset,
            parallel='15',
            additional_args=generate_dataset_args)
    (generate_dataset_op.container
        .set_memory_limit('55Gi')
        .set_memory_request('55Gi')
        .set_cpu_limit('15.5')
        .set_cpu_request('15.5')
    )
    (disable_caching(generate_dataset_op)
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'm5.4xlarge'))
    
    # train_args = '--train_iterations 3'
    # uncomment next line for longer training:
    train_args = '--train_iterations 10000'
    num_gpus = 1
    train_op = components.load_component_from_file('components/train.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            dataset_owner=dataset_owner,
            task_name='almond_dialogue_nlu',
            project=project,
            experiment=experiment,
            dataset=dataset,
            model=model,
            load_from='None',
            additional_args=train_args) 
    (train_op.container
        .set_memory_request('56Gi')
        .set_memory_limit('56Gi')
        .set_cpu_request('7.5')
        .set_cpu_limit('7.5')
        .set_gpu_limit(str(num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (disable_caching(train_op)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')))
        .after(generate_dataset_op)
    )
    
    eval_op = components.load_component_from_file('components/evaluate.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            model_owner=owner,
            eval_set='dev',
            eval_version='eval_version',
            additional_args='')  
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))  
    (disable_caching(eval_op)
        .after(train_op)
    )
