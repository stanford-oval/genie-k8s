#
# Copyright 2020-2021 The Board of Trustees of the Leland Stanford Junior University
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

from kfp import dsl
from kfp import components
from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1PersistentVolumeClaimVolumeSource,
)

from .common import *


def ood_classification_step(
    image,
    owner,
    project,
    experiment,
    model,
    genienlp_version,
    skip_tensorboard,
    s3_datadir,
    s3_bucket='geniehai',
    additional_args=''
):
    ood_classification_env = {
        'GENIENLP_VERSION': genienlp_version,
    }
    ood_classification_num_gpus = 1
    ood_classification_op = components.load_component_from_file('components/ood_classification.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            skip_tensorboard=skip_tensorboard,
            s3_datadir=s3_datadir,
            additional_args=additional_args)
    (ood_classification_op.container
        .set_memory_request('56Gi')
        .set_memory_limit('56Gi')
        .set_cpu_request('7.5')
        .set_cpu_limit('7.5')
        .set_gpu_limit(str(ood_classification_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (add_env(add_ssh_volume(ood_classification_op), ood_classification_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*ood_classification_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))

    return ood_classification_op


@dsl.pipeline(
    name='Train an ood model',
    description='Runs the ood classification pipeline on an existing dataset folder'
)
def ood_classification_pipeline(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    eval_set,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    additional_args=''
):
    ood_classification_op = ood_classification_step(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            image=image,
            genienlp_version=genienlp_version,
            skip_tensorboard='false',
            s3_datadir=s3_datadir,
            additional_args=additional_args
    )

    eval_model = ood_classification_op.outputs['s3_model_dir']

    pred_op = prediction_step(
            image=image,
            owner=owner,
            genienlp_version=genienlp_version,
            task_name='ood_task',
            eval_sets=eval_sets,
            model_name_or_path=eval_model,
            s3_input_datadir=s3_datadir,
            s3_database_dir='None',
            s3_bootleg_prepped_data='None',
            model_type='',
            dataset_subfolder='None',
            val_batch_size=val_batch_size,
            additional_args='',
    )

