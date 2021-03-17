#
# Copyright 2020 The Board of Trustees of the Leland Stanford Junior University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

from kfp import components
from kfp import dsl
from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1EmptyDirVolumeSource
)

from .common import *


@dsl.pipeline(
    name='Bootleg',
    description='Run a Bootleg model to extract and dump candidate features'
)
def bootleg_only_pipeline(
        owner,
        project,
        experiment,
        task_name,
        s3_datadir,
        s3_bucket='geniehai',
        s3_database_dir=S3_DATABASE_DIR,
        image='',
        genienlp_version='',
        bootleg_model='',
        train_languages='en',
        eval_languages='en',
        eval_set='',
        remove_original='false',
        bootleg_additional_args=''
):
    split_bootleg_merge_step(
        owner=owner,
        project=project,
        experiment=experiment,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        image=image,
        genienlp_version=genienlp_version,
        bootleg_model=bootleg_model,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        remove_original=remove_original,
        bootleg_additional_args=bootleg_additional_args
    )


def split_bootleg_merge_step(
        owner,
        project,
        experiment,
        task_name,
        s3_datadir,
        s3_bucket='geniehai',
        s3_database_dir=S3_DATABASE_DIR,
        image='',
        genienlp_version='',
        bootleg_model='',
        train_languages='en',
        eval_languages='en',
        eval_set='',
        remove_original='false',
        bootleg_additional_args=''
):
    num_chunks = 1
    split_op = split_step(
        image=image,
        task_name=task_name,
        s3_datadir=s3_datadir,
        num_chunks=num_chunks
    )
    
    bootleg_ops = []
    for i in range(num_chunks):
        bootleg_op = bootleg_step(
            owner=owner,
            project=project,
            experiment=experiment,
            task_name=task_name,
            s3_datadir=split_op.outputs['s3_output_datadir'],
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_model=bootleg_model,
            train_languages=train_languages,
            eval_languages=eval_languages,
            eval_set=eval_set,
            dataset_subfolder=str(i),
            bootleg_additional_args=bootleg_additional_args)
        bootleg_ops.append(bootleg_op)
    
    merge_op = merge_step(
        image=image,
        task_name=task_name,
        s3_datadir=split_op.outputs['s3_output_datadir'],
        bootleg_model=bootleg_model,
        num_chunks=num_chunks,
        remove_original=remove_original
    )
    
    merge_op.after(*bootleg_ops)
    
    s3_bootleg_prepped_data = merge_op.outputs['s3_output_datadir']
    
    return s3_bootleg_prepped_data


def split_step(
        image,
        task_name,
        s3_datadir,
        num_chunks
):
    split_env = {}
    
    split_op = components.load_component_from_file('components/split_file.yaml')(
        image=image,
        task_name=task_name,
        s3_datadir=s3_datadir,
        num_chunks=num_chunks)
    (split_op.container
     .set_memory_limit('12Gi')
     .set_memory_request('12Gi')
     .set_cpu_limit('7.5')
     .set_cpu_request('7.5'))
    (add_env(add_ssh_volume(split_op), split_env))
    
    split_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    return split_op


def merge_step(
        image,
        task_name,
        s3_datadir,
        bootleg_model,
        num_chunks,
        remove_original='false'
):
    merge_env = {}
    
    merge_op = components.load_component_from_file('components/merge_files.yaml')(
        image=image,
        task_name=task_name,
        s3_datadir=s3_datadir,
        bootleg_model=bootleg_model,
        num_chunks=num_chunks,
        remove_original=remove_original
    )
    (merge_op.container
     .set_memory_limit('12Gi')
     .set_memory_request('12Gi')
     .set_cpu_limit('7.5')
     .set_cpu_request('7.5'))
    (add_env(add_ssh_volume(merge_op), merge_env))
    
    merge_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    return merge_op


def bootleg_step(
        owner,
        project,
        experiment,
        task_name,
        s3_datadir,
        s3_bucket='geniehai',
        s3_database_dir=S3_DATABASE_DIR,
        image='',
        genienlp_version='',
        bootleg_model='',
        train_languages='en',
        eval_languages='en',
        eval_set='',
        dataset_subfolder='None',
        bootleg_additional_args=''
):
    bootleg_env = {
        'GENIENLP_VERSION': genienlp_version
    }
    
    bootleg_op = components.load_component_from_file('components/bootleg.yaml')(
        image=image,
        s3_bucket=s3_bucket,
        owner=owner,
        task_name=task_name,
        project=project,
        experiment=experiment,
        eval_set=eval_set,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        dataset_subfolder=dataset_subfolder,
        train_languages=train_languages,
        eval_languages=eval_languages,
        bootleg_model=bootleg_model,
        additional_args=bootleg_additional_args
    )
    (bootleg_op.container
     .set_memory_limit('61G')
     .set_memory_request('61G')
     .set_cpu_limit('15')
     .set_cpu_request('15')
     .add_volume_mount(V1VolumeMount(name='shm', mount_path='/dev/shm'))
     )
    (add_env(add_ssh_volume(bootleg_op), bootleg_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.4xlarge')
     .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
     )
    
    return bootleg_op
