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

from .common import add_env, add_ssh_volume


@dsl.pipeline(name='Train and eval TRADE', description='Train and evaluate a TRADE model')
def train_eval_trade(owner, project, experiment, model, s3_datadir, train_additional_args='', eval_additional_args=''):
    train_env = {}

    train_num_gpus = 1
    train_op = components.load_component_from_file('components/train-trade.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        s3_datadir=s3_datadir,
        additional_args=train_additional_args,
    )
    (
        train_op.container.set_memory_request('100G')
        .set_memory_limit('100G')
        .set_cpu_request('5')
        .set_cpu_limit('5')
        .set_gpu_limit(str(train_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (
        add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'Standard_NC{6 * train_num_gpus}s_v3')
        .add_volume(
            V1Volume(
                name='tensorboard', persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')
            )
        )
    )

    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-trade.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_datadir=s3_datadir,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args,
    )
    (eval_op.container.set_memory_limit('15Gi').set_memory_request('15Gi').set_cpu_limit('4').set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)


@dsl.pipeline(name='Train and eval SimpleTOD', description='Train and evaluate a SimpleTOD model')
def train_eval_simpletod(owner, project, experiment, model, s3_datadir, train_additional_args='', eval_additional_args=''):
    train_env = {}

    train_num_gpus = 1
    train_op = components.load_component_from_file('components/train-simpletod.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        s3_datadir=s3_datadir,
        additional_args=train_additional_args,
    )
    (
        train_op.container.set_memory_request('100G')
        .set_memory_limit('100G')
        .set_cpu_request('5')
        .set_cpu_limit('5')
        .set_gpu_limit(str(train_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (
        add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'Standard_NC{6 * train_num_gpus}s_v3')
        .add_volume(
            V1Volume(
                name='tensorboard', persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')
            )
        )
    )

    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-trade.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_datadir=s3_datadir,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args,
    )
    (eval_op.container.set_memory_limit('15Gi').set_memory_request('15Gi').set_cpu_limit('4').set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)


@dsl.pipeline(name='Train and eval SUMBT', description='Train and evaluate a SUMBT model')
def train_eval_sumbt(owner, project, experiment, model, s3_datadir, train_additional_args='', eval_additional_args=''):
    train_env = {}

    train_num_gpus = 1
    train_op = components.load_component_from_file('components/train-sumbt.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        s3_datadir=s3_datadir,
        additional_args=train_additional_args,
    )
    (
        train_op.container.set_memory_request('100G')
        .set_memory_limit('100G')
        .set_cpu_request('5')
        .set_cpu_limit('5')
        .set_gpu_limit(str(train_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (
        add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'Standard_NC{6 * train_num_gpus}s_v3')
        .add_volume(
            V1Volume(
                name='tensorboard', persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')
            )
        )
    )

    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-sumbt.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_datadir=s3_datadir,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args,
    )
    (eval_op.container.set_memory_limit('15Gi').set_memory_request('15Gi').set_cpu_limit('4').set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)


@dsl.pipeline(name='Eval SUMBT', description='Train and evaluate a SUMBT model')
def eval_sumbt(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    s3_model_dir,
    image='932360549041.dkr.ecr.us-west-2.amazonaws.com/sumbt:20201122.2',
    eval_additional_args='',
):
    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-sumbt.yaml')(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_datadir=s3_datadir,
        s3_model_dir=s3_model_dir,
        image=image,
        additional_args=eval_additional_args,
    )
    (eval_op.container.set_memory_limit('12G').set_memory_request('12G').set_cpu_limit('3').set_cpu_request('3'))
    (
        add_env(add_ssh_volume(eval_op), eval_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'Standard_NC4as_T4_v3')
    )
