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

from kfp import components, dsl
from kubernetes.client import V1Toleration
from kubernetes.client.models import V1EmptyDirVolumeSource

from .common import *


@dsl.pipeline(name='Bootleg', description='Run a Bootleg model to extract and dump candidate features')
def bootleg_only_pipeline(
    owner,
    project,
    experiment,
    task_name,
    s3_datadir,
    s3_bucket='geniehai',
    s3_database_dir=S3_DATABASE_DIR,
    s3_bootleg_subfolder='None',
    image='',
    genienlp_version='',
    bootleg_model='',
    train_languages='en',
    eval_languages='en',
    data_splits='train eval',
    file_extension='tsv',
    bootleg_additional_args='',
):
    split_bootleg_merge_step(
        owner=owner,
        project=project,
        experiment=experiment,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        s3_bootleg_subfolder=s3_bootleg_subfolder,
        image=image,
        genienlp_version=genienlp_version,
        bootleg_model=bootleg_model,
        train_languages=train_languages,
        eval_languages=eval_languages,
        data_splits=data_splits,
        file_extension=file_extension,
        bootleg_additional_args=bootleg_additional_args,
    )


def split_bootleg_merge_step(
    image,
    owner,
    project,
    experiment,
    task_name,
    s3_datadir,
    s3_bucket='geniehai',
    s3_database_dir=S3_DATABASE_DIR,
    s3_bootleg_subfolder='None',
    genienlp_version='',
    bootleg_model='',
    train_languages='en',
    eval_languages='en',
    data_splits='train eval',
    bootleg_additional_args='',
    file_extension='tsv',
):
    num_chunks = 2
    split_op = split_step(
        image=image,
        task_name=task_name,
        s3_datadir=s3_datadir,
        num_chunks=num_chunks,
        data_splits=data_splits,
        file_extension=file_extension,
    )

    s3_bootleg_outputs = []
    for i in range(num_chunks):
        bootleg_op = bootleg_step(
            owner=owner,
            project=project,
            experiment=experiment,
            task_name=task_name,
            s3_datadir=split_op.outputs['s3_output_datadir'],
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            s3_bootleg_subfolder=s3_bootleg_subfolder,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_model=bootleg_model,
            train_languages=train_languages,
            eval_languages=eval_languages,
            data_splits=data_splits,
            dataset_subfolder=str(i),
            bootleg_additional_args=bootleg_additional_args,
        )
        s3_bootleg_outputs.append(str(bootleg_op.outputs['s3_bootleg_prepped_data']))

    merge_op = merge_step(
        image=image,
        s3_datadir=split_op.outputs['s3_output_datadir'],
        s3_bootleg_prepped_data=' '.join(s3_bootleg_outputs),
        data_splits=data_splits,
    )

    s3_bootleg_prepped_data = merge_op.outputs['s3_output_datadir']

    return s3_bootleg_prepped_data


def split_step(image, task_name, s3_datadir, num_chunks, data_splits, file_extension):
    split_env = {}

    split_op = components.load_component_from_file('components/split_file.yaml')(
        image=image,
        task_name=task_name,
        s3_datadir=s3_datadir,
        num_chunks=num_chunks,
        data_splits=data_splits,
        file_extension=file_extension,
    )
    (split_op.container.set_memory_limit('12Gi').set_memory_request('12Gi').set_cpu_limit('7.5').set_cpu_request('7.5'))
    (add_env(add_ssh_volume(split_op), split_env))

    return split_op


def merge_step(image, s3_datadir, s3_bootleg_prepped_data, data_splits):
    merge_env = {}

    merge_op = components.load_component_from_file('components/merge_files.yaml')(
        image=image,
        s3_datadir=s3_datadir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        data_splits=data_splits,
    )
    (merge_op.container.set_memory_limit('12Gi').set_memory_request('12Gi').set_cpu_limit('7.5').set_cpu_request('7.5'))
    (add_env(add_ssh_volume(merge_op), merge_env))

    return merge_op


def bootleg_step(
    owner,
    project,
    experiment,
    task_name,
    s3_datadir,
    s3_bucket='geniehai',
    s3_database_dir=S3_DATABASE_DIR,
    s3_bootleg_subfolder='None',
    image='',
    genienlp_version='',
    bootleg_model='',
    train_languages='en',
    eval_languages='en',
    data_splits='train eval',
    dataset_subfolder='None',
    bootleg_additional_args='',
):
    bootleg_env = {'GENIENLP_VERSION': genienlp_version}

    bootleg_op = components.load_component_from_file('components/bootleg.yaml')(
        image=image,
        s3_bucket=s3_bucket,
        owner=owner,
        task_name=task_name,
        project=project,
        experiment=experiment,
        data_splits=data_splits,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_subfolder=s3_bootleg_subfolder,
        dataset_subfolder=dataset_subfolder,
        train_languages=train_languages,
        eval_languages=eval_languages,
        bootleg_model=bootleg_model,
        additional_args=bootleg_additional_args,
    )
    (
        bootleg_op.container.set_memory_request('60G')
        .set_memory_limit('60G')
        .set_cpu_request('15')
        .set_cpu_limit('15')
        .add_volume_mount(V1VolumeMount(name='shm', mount_path='/dev/shm'))
    )
    (
        add_env(add_ssh_volume(bootleg_op), bootleg_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.4xlarge')
        .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
    )

    return bootleg_op


def bootleg_filtering_step(
    image,
    owner,
    project,
    experiment,
    task_name,
    workdir_repo,
    genie_version,
    workdir_version,
    thingpedia_developer_key,
    dataset,
    s3_datadir,
    s3_bootleg_prepped_data,
    s3_bucket='geniehai',
    s3_bootleg_subfolder='None',
    dataset_subfolder='',
    bootleg_model='',
    valid_set='eval',
    additional_args=''
):
    bootleg_filtering_env = {
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }

    bootleg_filtering_op = components.load_component_from_file('components/filter-bootleg.yaml')(
        image=image,
        s3_bucket=s3_bucket,
        owner=owner,
        project=project,
        experiment=experiment,
        task_name=task_name,
        dataset=dataset,
        s3_datadir=s3_datadir,
        s3_bootleg_subfolder=s3_bootleg_subfolder,
        dataset_subfolder=dataset_subfolder,
        bootleg_model=bootleg_model,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        valid_set=valid_set,
        additional_args=additional_args
    )
    (bootleg_filtering_op.container.set_memory_limit('12Gi').set_memory_request('12Gi').set_cpu_limit('7.5').set_cpu_request('7.5'))
    (add_env(add_ssh_volume(bootleg_filtering_op), bootleg_filtering_env))

    return bootleg_filtering_op;
