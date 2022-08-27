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
from kubernetes.client.models import V1PersistentVolumeClaimVolumeSource, V1SecretVolumeSource, V1Volume, V1VolumeMount

from .common import *


def paraphrase_generation_step(
    image,
    owner,
    project,
    experiment,
    dataset,
    s3_input_datadir,
    train_task_name,
    paraphrasing_model,
    keep_original_duplicates,
    genienlp_version,
    paraphrase_subfolder,
    additional_args,
):
    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
    }

    paraphrase_num_gpus = 4
    paraphrase_op = components.load_component_from_file('components/generate-paraphrase.yaml')(
        image=image,
        s3_bucket=AZURE_BUCKET,
        owner=owner,
        task_name=train_task_name,
        project=project,
        experiment=experiment,
        dataset=dataset,
        s3_input_datadir=s3_input_datadir,
        paraphrasing_model=paraphrasing_model,
        keep_original_duplicates=keep_original_duplicates,
        paraphrase_subfolder=paraphrase_subfolder,
        additional_args=additional_args,
    )
    (
        paraphrase_op.container.set_memory_request('400G')
        .set_memory_limit('400G')
        .set_cpu_request('60')
        .set_cpu_limit('60')
        # not supported yet in the version of kfp we're using
        # .set_ephemeral_storage_request('75G')
        # .set_ephemeral_storage_limit('75G')
        .set_gpu_limit(str(paraphrase_num_gpus))
    )
    (
        add_env(add_ssh_volume(paraphrase_op), paraphrase_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'Standard_NC64as_T4_v3')
    )

    return paraphrase_op


def paraphrase_filtering_step(
    image,
    owner,
    project,
    experiment,
    dataset,
    s3_input_datadir,
    train_task_name,
    filtering_model,
    filtering_batch_size,
    genienlp_version,
    paraphrase_subfolder,
    s3_database_dir,
    s3_bootleg_prepped_data,
    s3_original_bootleg_prepped_data,
    additional_args,
):
    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
    }

    paraphrase_num_gpus = 4
    paraphrase_op = components.load_component_from_file('components/filter-paraphrase.yaml')(
        image=image,
        s3_bucket=AZURE_BUCKET,
        owner=owner,
        task_name=train_task_name,
        project=project,
        experiment=experiment,
        dataset=dataset,
        s3_input_datadir=s3_input_datadir,
        s3_database_dir=s3_database_dir,
        filtering_model=filtering_model,
        filtering_batch_size=filtering_batch_size,
        paraphrase_subfolder=paraphrase_subfolder,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        s3_original_bootleg_prepped_data=s3_original_bootleg_prepped_data,
        additional_args=additional_args,
    )
    (
        paraphrase_op.container.set_memory_request('400G')
        .set_memory_limit('400G')
        .set_cpu_request('60')
        .set_cpu_limit('60')
        # not supported yet in the version of kfp we're using
        # .set_ephemeral_storage_request('75G')
        # .set_ephemeral_storage_limit('75G')
        .set_gpu_limit(str(paraphrase_num_gpus))
    )
    (
        add_env(add_ssh_volume(paraphrase_op), paraphrase_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'Standard_NC64as_T4_v3')
    )

    return paraphrase_op
