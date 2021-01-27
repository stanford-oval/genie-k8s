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

from kfp import dsl
from kfp import components
from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1PersistentVolumeClaimVolumeSource,
)

from .common import *

from .paraphrase import paraphrase_generation_step, paraphrase_filtering_step


def generate_dataset_step(
    image,
    owner,
    project,
    experiment,
    dataset,
    parallel,
    genie_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    gen_dataset_env = {
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }
    generate_dataset_op = components.load_component_from_file('components/generate-dataset.yaml')(
            image=image,
            s3_bucket='geniehai',
            owner=owner,
            project=project,
            experiment=experiment,
            dataset=dataset,
            parallel=parallel,
            additional_args=additional_args)
    (generate_dataset_op.container
        .set_memory_limit('55Gi')
        .set_memory_request('55Gi')
        .set_cpu_limit('15.5')
        .set_cpu_request('15.5')
    )
    (add_env(add_ssh_volume(generate_dataset_op), gen_dataset_env))

    return generate_dataset_op


def train_step(
    image,
    owner,
    project,
    experiment,
    model,
    task_name,
    load_from,
    s3_datadir,
    dataset_subfolder,
    genienlp_version,
    train_iterations,
    skip_tensorboard,
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    s3_database_dir='None',
    bootleg_version='',
    train_languages='en',
    eval_languages='en',
    s3_bucket='geniehai',
    use_bootleg='false',
    s3_bootleg_prepped_data='None',
    bootleg_model='None',
    additional_args=''
):
    train_env = {
        'GENIENLP_VERSION': genienlp_version,
        'BOOTLEG_VERSION': bootleg_version,
    }
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            task_name=task_name,
            project=project,
            experiment=experiment,
            model=model,
            load_from=load_from,
            s3_datadir=s3_datadir,
            s3_database_dir=s3_database_dir,
            dataset_subfolder=dataset_subfolder,
            train_iterations=train_iterations,
            skip_tensorboard=skip_tensorboard,
            calibrate=calibrate,
            calibration_ood_file=calibration_ood_file,
            calibration_additional_args=calibration_additional_args,
            train_languages=train_languages,
            eval_languages=eval_languages,
            use_bootleg=use_bootleg,
            s3_bootleg_prepped_data=s3_bootleg_prepped_data,
            bootleg_model=bootleg_model,
            additional_args=additional_args)
    (train_op.container
        .set_memory_request('56Gi')
        .set_memory_limit('56Gi')
        .set_cpu_request('7.5')
        .set_cpu_limit('7.5')
        .set_gpu_limit(str(train_num_gpus))
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*train_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))

    train_op.container.set_image_pull_policy('Always')
    
    return train_op


def train_step_4gpus(
        image,
        owner,
        project,
        experiment,
        model,
        task_name,
        load_from,
        s3_datadir,
        dataset_subfolder,
        genienlp_version,
        train_iterations,
        skip_tensorboard,
        calibrate='false',
        calibration_ood_file='None',
        calibration_additional_args='None',
        s3_database_dir='None',
        bootleg_version='',
        train_languages='en',
        eval_languages='en',
        s3_bucket='geniehai',
        use_bootleg='false',
        s3_bootleg_prepped_data='None',
        bootleg_model='None',
        additional_args=''
):
    train_env = {
        'GENIENLP_VERSION': genienlp_version,
        'BOOTLEG_VERSION': bootleg_version,
    }
    train_num_gpus = 4
    train_op = components.load_component_from_file('components/train.yaml')(
        image=image,
        s3_bucket=s3_bucket,
        owner=owner,
        task_name=task_name,
        project=project,
        experiment=experiment,
        model=model,
        load_from=load_from,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        dataset_subfolder=dataset_subfolder,
        train_iterations=train_iterations,
        skip_tensorboard=skip_tensorboard,
        calibrate=calibrate,
        calibration_ood_file=calibration_ood_file,
        calibration_additional_args=calibration_additional_args,
        train_languages=train_languages,
        eval_languages=eval_languages,
        use_bootleg=use_bootleg,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        bootleg_model=bootleg_model,
        additional_args=additional_args)
    (train_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(train_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     )
    (add_env(add_ssh_volume(train_op), train_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * train_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))
    
    train_op.container.set_image_pull_policy('Always')
    
    return train_op


def eval_step(
    image,
    owner,
    project,
    experiment,
    model,
    s3_model_dir,
    eval_set,
    genienlp_version,
    genie_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    eval_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }

    eval_op = components.load_component_from_file('components/evaluate.yaml')(
            image=image,
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            model_owner=owner,
            eval_set=eval_set,
            s3_model_dir=s3_model_dir,
            additional_args=additional_args)
    (eval_op.container
        .set_memory_limit('12Gi')
        .set_memory_request('12Gi')
        .set_cpu_limit('7.5')
        .set_cpu_request('7.5'))
    (add_env(add_ssh_volume(eval_op), eval_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.2xlarge'))

    return eval_op


def paraphrase_fewshot_step(
    do_paraphrase,
    do_fewshot,
    owner,
    project,
    experiment,
    model,
    dataset,
    image,
    genienlp_version,
    train_task_name,
    train_load_from,
    train_additional_args,
    train_iterations,
    train_s3_datadir,
    calibrate,
    calibration_ood_file,
    calibration_additional_args,
    s3_bucket,
    s3_database_dir,
    bootleg_model,
    bootleg_version,
    train_languages,
    eval_languages,
    eval_set,
    s3_bootleg_prepped_data,
    train_dataset_subfolder,
    filtering_train_iterations,
    filtering_batch_size,
    fewshot_train_iterations,
    keep_original_duplicates,
    paraphrasing_model,
    paraphrase_subfolder,
    paraphrase_additional_args,
    filtering_additional_args,
):
    if do_paraphrase:
        pretrain_op = train_step(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            task_name=train_task_name,
            s3_datadir=train_s3_datadir,
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            bootleg_model=bootleg_model,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_version=bootleg_version,
            load_from='None',
            train_languages=train_languages,
            eval_languages=eval_languages,
            dataset_subfolder='None',
            skip_tensorboard='true',
            train_iterations=filtering_train_iterations,
            s3_bootleg_prepped_data=s3_bootleg_prepped_data,
            additional_args=train_additional_args
        )

        paraphrase_generation_op = paraphrase_generation_step(image=image,
                                        owner=owner,
                                        project=project,
                                        experiment=experiment,
                                        dataset=dataset,
                                        s3_input_datadir=train_s3_datadir,
                                        train_task_name=train_task_name,
                                        paraphrasing_model=paraphrasing_model,
                                        keep_original_duplicates=keep_original_duplicates,
                                        genienlp_version=genienlp_version,
                                        paraphrase_subfolder=paraphrase_subfolder,
                                        additional_args=paraphrase_additional_args)

        paraphrase_filtering_op = paraphrase_filtering_step(image=image,
                                        owner=owner,
                                        project=project,
                                        experiment=experiment,
                                        dataset=dataset,
                                        s3_input_datadir=paraphrase_generation_op.outputs['s3_output_datadir'],
                                        train_task_name=train_task_name,
                                        filtering_model=pretrain_op.outputs['s3_model_dir'],
                                        filtering_batch_size=filtering_batch_size,
                                        genienlp_version=genienlp_version,
                                        paraphrase_subfolder=paraphrase_subfolder,
                                        additional_args=filtering_additional_args)

        train_s3_datadir = paraphrase_filtering_op.outputs['s3_output_datadir']

    train_op = train_step(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            task_name=train_task_name,
            s3_datadir=train_s3_datadir,
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            bootleg_model=bootleg_model,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_version=bootleg_version,
            load_from=train_load_from,
            train_languages=train_languages,
            eval_languages=eval_languages,
            dataset_subfolder=train_dataset_subfolder,
            skip_tensorboard='false',
            calibrate=calibrate,
            calibration_ood_file=calibration_ood_file,
            calibration_additional_args=calibration_additional_args,
            train_iterations=train_iterations,
            s3_bootleg_prepped_data=s3_bootleg_prepped_data,
            additional_args=train_additional_args,
            )
    eval_model = train_op.outputs['s3_model_dir']

    if do_fewshot:
        model = '%s-fs' % (model,)
        fewshot_op = train_step(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
            task_name=train_task_name,
            s3_datadir=train_s3_datadir,
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            bootleg_model=bootleg_model,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_version=bootleg_version,
            load_from=train_op.outputs['s3_model_dir'],
            train_languages=train_languages,
            eval_languages=eval_languages,
            dataset_subfolder='fewshot/',
            skip_tensorboard='false',
            calibrate=calibrate,
            calibration_ood_file=calibration_ood_file,
            calibration_additional_args=calibration_additional_args,
            train_iterations=fewshot_train_iterations,
            s3_bootleg_prepped_data=s3_bootleg_prepped_data,
            additional_args=train_additional_args,
        )
        eval_model = fewshot_op.outputs['s3_model_dir']

    return train_s3_datadir, eval_model

def paraphrase_only(
    owner,
    project,
    experiment,
    dataset,
    image,
    genienlp_version,
    s3_input_datadir,
    train_task_name,
    keep_original_duplicates,
    paraphrasing_model,
    filtering_model,
    filtering_batch_size,
    paraphrase_subfolder,
    paraphrase_additional_args,
    filtering_additional_args,
):
    paraphrase_generation_op = paraphrase_generation_step(image=image,
                                    owner=owner,
                                    project=project,
                                    experiment=experiment,
                                    dataset=dataset,
                                    s3_input_datadir=s3_input_datadir,
                                    train_task_name=train_task_name,
                                    paraphrasing_model=paraphrasing_model,
                                    keep_original_duplicates=keep_original_duplicates,
                                    genienlp_version=genienlp_version,
                                    paraphrase_subfolder=paraphrase_subfolder,
                                    additional_args=paraphrase_additional_args)

    paraphrase_filtering_op = paraphrase_filtering_step(image=image,
                                    owner=owner,
                                    project=project,
                                    experiment=experiment,
                                    dataset=dataset,
                                    s3_input_datadir=paraphrase_generation_op.outputs['s3_output_datadir'],
                                    train_task_name=train_task_name,
                                    filtering_model=filtering_model,
                                    filtering_batch_size=filtering_batch_size,
                                    genienlp_version=genienlp_version,
                                    paraphrase_subfolder=paraphrase_subfolder,
                                    additional_args=filtering_additional_args)

    output_s3_datadir = paraphrase_filtering_op.outputs['s3_output_datadir']

    return output_s3_datadir

def everything(
    do_generate,
    do_paraphrase,
    do_fewshot,
    owner,
    project,
    experiment,
    model,
    dataset='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    s3_bucket='geniehai',
    s3_database_dir='None',
    bootleg_model='None',
    bootleg_version='',
    train_languages='en',
    eval_languages='en',
    s3_bootleg_prepped_data='None',
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    train_s3_datadir='',
    train_dataset_subfolder='None',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    fewshot_train_iterations='20000',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='',
    eval_additional_args=''):

    if do_generate:
        generate_dataset_op = generate_dataset_step(image=image,
                                                    owner=owner,
                                                    project=project,
                                                    experiment=experiment,
                                                    dataset=dataset,
                                                    parallel=generate_dataset_parallel,
                                                    genie_version=genie_version,
                                                    workdir_repo=workdir_repo,
                                                    workdir_version=workdir_version,
                                                    thingpedia_developer_key=thingpedia_developer_key,
                                                    additional_args=generate_dataset_additional_args)
        train_s3_datadir = generate_dataset_op.outputs['s3_datadir']

    train_s3_datadir, eval_model = paraphrase_fewshot_step(
        do_paraphrase=do_paraphrase,
        do_fewshot=do_fewshot,
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        dataset=dataset,
        image=image,
        genienlp_version=genienlp_version,
        train_task_name=train_task_name,
        train_load_from=train_load_from,
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        train_s3_datadir=train_s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        bootleg_version=bootleg_version,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        train_dataset_subfolder=train_dataset_subfolder,
        calibrate=calibrate,
        calibration_ood_file=calibration_ood_file,
        calibration_additional_args=calibration_additional_args,
        filtering_train_iterations=filtering_train_iterations,
        filtering_batch_size=filtering_batch_size,
        fewshot_train_iterations=fewshot_train_iterations,
        keep_original_duplicates=keep_original_duplicates,
        paraphrasing_model=paraphrasing_model,
        paraphrase_subfolder=paraphrase_subfolder,
        paraphrase_additional_args=paraphrase_additional_args,
        filtering_additional_args=filtering_additional_args,
    )

    eval_op = eval_step(image=image,
                        owner=owner,
                        project=project,
                        experiment=experiment,
                        model=model,
                        s3_model_dir=eval_model,
                        eval_set=eval_set,
                        genienlp_version=genienlp_version,
                        genie_version=genie_version,
                        workdir_repo=workdir_repo,
                        workdir_version=workdir_version,
                        thingpedia_developer_key=thingpedia_developer_key,
                        additional_args=eval_additional_args)


@dsl.pipeline(
    name='Generate, train and eval',
    description='The minimal training pipeline'
)
def generate_train_eval_pipeline(
    owner,
    project,
    experiment,
    model,
    dataset,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=True,
               do_paraphrase=False,
               do_fewshot=False,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Train and eval',
    description='Trains and evaluate on an existing dataset'
)
def train_eval_only_pipeline(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    dataset_subfolder='None',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    train_task_name='',
    train_load_from='None',
    train_iterations='80000',
    train_additional_args='',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=False,
               do_paraphrase=False,
               do_fewshot=False,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               train_s3_datadir=s3_datadir,
               train_dataset_subfolder=dataset_subfolder,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_iterations=train_iterations,
               train_additional_args=train_additional_args,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Generate, paraphrase, train, and eval',
    description='Runs the whole training pipeline, including autoparaphrasing'
)
def generate_paraphrase_train_eval_pipeline(
    owner,
    project,
    experiment,
    model,
    dataset,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=True,
               do_paraphrase=True,
               do_fewshot=False,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
               filtering_additional_args=filtering_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Generate, train, fewshot, and eval',
    description='Runs the whole training pipeline, with fewshot finetuning'
)
def generate_train_fewshot_eval_pipeline(
    owner,
    project,
    experiment,
    model,
    dataset,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    fewshot_train_iterations='20000',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=True,
               do_paraphrase=False,
               do_fewshot=True,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               fewshot_train_iterations=fewshot_train_iterations,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Generate, paraphrase, train, fewshot, and eval',
    description='Runs the whole training pipeline, with autoparaphrasing and fewshot finetuning'
)
def generate_paraphrase_train_fewshot_eval_pipeline(
    owner,
    project,
    experiment,
    model,
    dataset,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    fewshot_train_iterations='20000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=True,
               do_paraphrase=True,
               do_fewshot=True,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               fewshot_train_iterations=fewshot_train_iterations,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
               filtering_additional_args=filtering_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)

@dsl.pipeline(
    name='Paraphrase, train, fewshot, and eval',
    description='Runs the whole training pipeline on an existing dataset folder, with autoparaphrasing and fewshot finetuning'
)
def paraphrase_train_fewshot_eval_pipeline(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    dataset_subfolder='None',
    dataset='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    fewshot_train_iterations='20000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=False,
               do_paraphrase=True,
               do_fewshot=True,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               train_s3_datadir=s3_datadir,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               fewshot_train_iterations=fewshot_train_iterations,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
               filtering_additional_args=filtering_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Paraphrase, train, and eval',
    description='Runs the whole auto-paraphrasing pipeline on an existing dataset folder, and trains a model'
)
def paraphrase_train_eval_pipeline(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    dataset_subfolder='None',
    dataset='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    train_task_name='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_ood_file='None',
    calibration_additional_args='None',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=False,
               do_paraphrase=True,
               do_fewshot=False,
               owner=owner,
               project=project,
               experiment=experiment,
               model=model,
               train_s3_datadir=s3_datadir,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               calibrate=calibrate,
               calibration_ood_file=calibration_ood_file,
               calibration_additional_args=calibration_additional_args,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
               filtering_additional_args=filtering_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Evaluate',
    description='Evaluate a previously trained model'
)
def eval_only_pipeline(
    owner,
    project,
    experiment,
    model,
    s3_model_dir,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    eval_set='',
    additional_args=''
):
    eval_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        s3_model_dir=s3_model_dir,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        thingpedia_developer_key=thingpedia_developer_key,
        eval_set=eval_set,
        additional_args=additional_args)

@dsl.pipeline(
    name='Paraphrase (and filter) a dataset',
    description='Runs auto-paraphrasing pipeline on an existing dataset folder'
)
def paraphrase_only_pipeline(
    owner,
    project,
    experiment,
    dataset='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    s3_input_datadir='',
    train_task_name='',
    filtering_batch_size='4000',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    filtering_model='',
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
):
    paraphrase_only(owner,
                    project,
                    experiment,
                    dataset,
                    image,
                    genienlp_version,
                    s3_input_datadir,
                    train_task_name,
                    keep_original_duplicates,
                    paraphrasing_model,
                    filtering_model,
                    filtering_batch_size,
                    paraphrase_subfolder,
                    paraphrase_additional_args,
                    filtering_additional_args,
                    )