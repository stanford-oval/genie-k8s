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

# Get the Thingpedia key from environment variable
default_developer_key = os.environ['THINGPEDIA_DEVELOPER_KEY']

default_image = '932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit-kf:20201113.1-next'
GENIENLP_VERSION = 'd04ed4a2c38788eab9a9f4694a20fddeba62ea7d'
GENIE_VERSION = '862f3444aaee0522c84aa7b24ad0a3f7203b9f48'
THINGTALK_VERSION = 'a3eb276cab0f554646ee6ef5620be12179f55ba7'
BOOTLEG_VERSION = 'f53e67397ddcd099f3a18a014c9ce82b02d2223c'
WORKDIR_REPO = 'git@github.com:stanford-oval/thingpedia-common-devices.git'
WORKDIR_VERSION = '0db4d113bd2436e85f7dfa7542f800106485f7a8'
PARAPHRASING_MODEL = 's3://geniehai/sinaj/models/schemaorg/paraphrase/bart-large-speedup-megabatch-5m/'


def add_ssh_volume(op):
    op.add_volume(V1Volume(name='ssh-v',
        secret=V1SecretVolumeSource(secret_name='ssh-secrets-k425k8d8h8', default_mode=0o600)))
    op.container.add_volume_mount(V1VolumeMount(name='ssh-v', mount_path='/root/.ssh'))
    return op


def generate_dataset_step(
    image,
    owner,
    project,
    experiment,
    dataset,
    parallel,
    genie_version,
    thingtalk_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    gen_dataset_env = {
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
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


def auto_annotate_step(
    image,
    owner,
    project,
    experiment,
    dataset,
    user_model,
    agent_model,
    genienlp_version,
    genie_version,
    thingtalk_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    auto_annotate_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }
    auto_annotate_op = components.load_component_from_file('components/auto-annotate-selftrain.yaml')(
            image=image,
            s3_bucket='geniehai',
            owner=owner,
            project=project,
            experiment=experiment,
            dataset=dataset,
            user_model=user_model,
            agent_model=agent_model,
            additional_args=additional_args)
    (auto_annotate_op.container
        .set_memory_limit('12Gi')
        .set_memory_request('12Gi')
        .set_cpu_limit('7.5')
        .set_cpu_request('7.5')
    )
    
    (add_env(add_ssh_volume(auto_annotate_op), auto_annotate_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.2xlarge'))
    
    return auto_annotate_op


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
    additional_args
):
    train_env = {
        'GENIENLP_VERSION': genienlp_version,
    }
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train.yaml')(
            image=image,
            s3_bucket='geniehai',
            owner=owner,
            task_name=task_name,
            project=project,
            experiment=experiment,
            model=model,
            load_from=load_from,
            s3_datadir=s3_datadir,
            dataset_subfolder=dataset_subfolder,
            train_iterations=train_iterations,
            skip_tensorboard=skip_tensorboard,
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
    thingtalk_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    eval_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
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
    ignore_context,
    genienlp_version,
    paraphrase_subfolder,
    additional_args
):
    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
    }
    
    paraphrase_num_gpus=4
    paraphrase_op = components.load_component_from_file('components/generate-paraphrase.yaml')(
        image=image,
        s3_bucket='geniehai',
        owner=owner,
        task_name=train_task_name,
        project=project,
        experiment=experiment,
        dataset=dataset,
        s3_input_datadir=s3_input_datadir,
        paraphrasing_model=paraphrasing_model,
        keep_original_duplicates=keep_original_duplicates,
        ignore_context=ignore_context,
        paraphrase_subfolder=paraphrase_subfolder,
        additional_args=additional_args)
    (paraphrase_op.container
        .set_memory_request('150G')
        .set_memory_limit('150G')
        .set_cpu_request('16')
        .set_cpu_limit('16')
        # not supported yet in the version of kfp we're using
        #.set_ephemeral_storage_request('75G')
        #.set_ephemeral_storage_limit('75G')
        .set_gpu_limit(str(paraphrase_num_gpus))
    )
    (add_env(add_ssh_volume(paraphrase_op), paraphrase_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.12xlarge'))
     
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
    additional_args
):
    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
    }
    
    paraphrase_num_gpus=4
    paraphrase_op = components.load_component_from_file('components/filter-paraphrase.yaml')(
        image=image,
        s3_bucket='geniehai',
        owner=owner,
        task_name=train_task_name,
        project=project,
        experiment=experiment,
        dataset=dataset,
        s3_input_datadir=s3_input_datadir,
        filtering_model=filtering_model,
        filtering_batch_size=filtering_batch_size,
        paraphrase_subfolder=paraphrase_subfolder,
        additional_args=additional_args)
    (paraphrase_op.container
        .set_memory_request('150G')
        .set_memory_limit('150G')
        .set_cpu_request('16')
        .set_cpu_limit('16')
        # not supported yet in the version of kfp we're using
        #.set_ephemeral_storage_request('75G')
        #.set_ephemeral_storage_limit('75G')
        .set_gpu_limit(str(paraphrase_num_gpus))
    )
    (add_env(add_ssh_volume(paraphrase_op), paraphrase_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.12xlarge'))
     
    return paraphrase_op


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
    train_dataset_subfolder,
    filtering_train_iterations,
    filtering_batch_size,
    fewshot_train_iterations,
    ignore_context,
    keep_original_duplicates,
    paraphrasing_model,
    paraphrase_subfolder,
    paraphrase_additional_args,
    filtering_additional_args,
):
    if do_paraphrase:
        pretrain_op = train_step(image=image,
                                 owner=owner,
                                 project=project,
                                 experiment=experiment,
                                 model=model,
                                 task_name=train_task_name,
                                 load_from='None',
                                 s3_datadir=train_s3_datadir,
                                 dataset_subfolder='None',
                                 genienlp_version=genienlp_version,
                                 train_iterations=filtering_train_iterations,
                                 skip_tensorboard='true',
                                 additional_args=train_additional_args)

        paraphrase_generation_op = paraphrase_generation_step(image=image,
                                        owner=owner,
                                        project=project,
                                        experiment=experiment,
                                        dataset=dataset,
                                        s3_input_datadir=train_s3_datadir,
                                        train_task_name=train_task_name,
                                        paraphrasing_model=paraphrasing_model,
                                        keep_original_duplicates=keep_original_duplicates,
                                        ignore_context=ignore_context,
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
    
    train_op = train_step(image=image,
                          owner=owner,
                          project=project,
                          experiment=experiment,
                          model=model,
                          task_name=train_task_name,
                          load_from=train_load_from,
                          s3_datadir=train_s3_datadir,
                          dataset_subfolder=train_dataset_subfolder,
                          genienlp_version=genienlp_version,
                          train_iterations=train_iterations,
                          skip_tensorboard='false',
                          additional_args=train_additional_args)
    eval_model = train_op.outputs['s3_model_dir']
    
    if do_fewshot:
        model = '%s-fs' % (model,)
        fewshot_op = train_step(image=image,
                                owner=owner,
                                project=project,
                                experiment=experiment,
                                model=model,
                                task_name=train_task_name,
                                load_from=train_op.outputs['s3_model_dir'],
                                s3_datadir=train_s3_datadir,
                                dataset_subfolder='fewshot/',
                                genienlp_version=genienlp_version,
                                train_iterations=fewshot_train_iterations,
                                skip_tensorboard='false',
                                additional_args=train_additional_args)
        eval_model = fewshot_op.outputs['s3_model_dir']
    
    return train_s3_datadir, eval_model


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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    train_s3_datadir='',
    train_dataset_subfolder='None',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    fewshot_train_iterations='20000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='dev',
    eval_additional_args=''):

    if do_generate:
        generate_dataset_op = generate_dataset_step(image=image,
                                                    owner=owner,
                                                    project=project,
                                                    experiment=experiment,
                                                    dataset=dataset,
                                                    parallel=generate_dataset_parallel,
                                                    genie_version=genie_version,
                                                    thingtalk_version=thingtalk_version,
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
        train_dataset_subfolder=train_dataset_subfolder,
        filtering_train_iterations=filtering_train_iterations,
        filtering_batch_size=filtering_batch_size,
        fewshot_train_iterations=fewshot_train_iterations,
        ignore_context=ignore_context,
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
                        thingtalk_version=thingtalk_version,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_iterations='80000',
    train_additional_args='',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_iterations=train_iterations,
               train_additional_args=train_additional_args,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               ignore_context=ignore_context,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    fewshot_train_iterations='20000',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    fewshot_train_iterations='20000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_parallel=generate_dataset_parallel,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               fewshot_train_iterations=fewshot_train_iterations,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               ignore_context=ignore_context,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    fewshot_train_iterations='20000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               fewshot_train_iterations=fewshot_train_iterations,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               ignore_context=ignore_context,
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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args='',
    eval_set='dev',
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
               thingtalk_version=thingtalk_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               train_task_name=train_task_name,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               filtering_train_iterations=filtering_train_iterations,
               filtering_batch_size=filtering_batch_size,
               ignore_context=ignore_context,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
               filtering_additional_args=filtering_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)


@dsl.pipeline(
    name='Selftrain',
    description='Runs the whole training pipeline, including two parallel autoparaphrasing and finetuning (for user and agent), auto annotation, and selftrain finetuning'
)
def selftrain_pipeline(
    owner,
    project,
    experiment,
    model,
    dataset,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_additional_args='',
    train_iterations='80000',
    fewshot_train_iterations='20000',
    selftrain_train_iterations='20000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_additional_args='',
    filtering_additional_args='',
    auto_annotate_additional_args='',
    eval_set='dev',
    eval_additional_args=''
):
    # first, generate the dataset
    generate_dataset_op = generate_dataset_step(image=image,
                                                    owner=owner,
                                                    project=project,
                                                    experiment=experiment,
                                                    dataset=dataset,
                                                    parallel=generate_dataset_parallel,
                                                    genie_version=genie_version,
                                                    thingtalk_version=thingtalk_version,
                                                    workdir_repo=workdir_repo,
                                                    workdir_version=workdir_version,
                                                    thingpedia_developer_key=thingpedia_developer_key,
                                                    additional_args=generate_dataset_additional_args)
    initial_datadir = generate_dataset_op.outputs['s3_datadir']
    
    # autoparaphrase and few-shot finetune the user model
    user_gen_datadir, user_model = paraphrase_fewshot_step(
        do_paraphrase=True,
        do_fewshot=True,
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        dataset=dataset,
        image=image,
        genienlp_version=genienlp_version,
        train_task_name='almond_dialogue_nlu',
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        train_s3_datadir=initial_datadir,
        train_load_from='None',
        train_dataset_subfolder='None',
        filtering_train_iterations=filtering_train_iterations,
        filtering_batch_size=filtering_batch_size,
        fewshot_train_iterations=fewshot_train_iterations,
        ignore_context=ignore_context,
        keep_original_duplicates=keep_original_duplicates,
        paraphrasing_model=paraphrasing_model,
        paraphrase_subfolder='user',
        paraphrase_additional_args=paraphrase_additional_args,
        filtering_additional_args=filtering_additional_args,
    )
    
    # autoparaphrase and few-shot finetune the agent model
    agent_gen_datadir, agent_model = paraphrase_fewshot_step(
        do_paraphrase=True,
        do_fewshot=True,
        owner=owner,
        project=project,
        experiment=experiment,
        model='%s-agent' % (model,),
        dataset='%s-agent' % (dataset,),
        image=image,
        genienlp_version=genienlp_version,
        train_task_name='almond_dialogue_nlu_agent',
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        train_s3_datadir=initial_datadir,
        train_load_from='None',
        train_dataset_subfolder='None',
        filtering_train_iterations=filtering_train_iterations,
        filtering_batch_size=filtering_batch_size,
        fewshot_train_iterations=fewshot_train_iterations,
        ignore_context=ignore_context,
        keep_original_duplicates=keep_original_duplicates,
        paraphrasing_model=paraphrasing_model,
        paraphrase_subfolder='agent',
        paraphrase_additional_args=paraphrase_additional_args,
        filtering_additional_args=filtering_additional_args,
    )
    
    auto_annotate_op = auto_annotate_step(image=image,
                                          owner=owner,
                                          project=project,
                                          experiment=experiment,
                                          dataset=dataset,
                                          user_model=user_model,
                                          agent_model=agent_model,
                                          genienlp_version=genienlp_version,       
                                          genie_version=genie_version,
                                          thingtalk_version=thingtalk_version,
                                          workdir_repo=workdir_repo,
                                          workdir_version=workdir_version,
                                          thingpedia_developer_key=thingpedia_developer_key,
                                          additional_args=auto_annotate_additional_args)
    selftrain_datadir = auto_annotate_op.outputs['s3_datadir']
    
    train_op = train_step(image=image,
                          owner=owner,
                          project=project,
                          experiment=experiment,
                          model='%s-selftrain' % (model,),
                          task_name='almond_dialogue_nlu',
                          load_from=user_model,
                          s3_datadir=selftrain_datadir,
                          dataset_subfolder='None',
                          genienlp_version=genienlp_version,
                          train_iterations=selftrain_train_iterations,
                          skip_tensorboard='false',
                          additional_args=train_additional_args)
    eval_model = train_op.outputs['s3_model_dir']
    
    eval_op = eval_step(image=image,
                        owner=owner,
                        project=project,
                        experiment=experiment,
                        model=model,
                        s3_model_dir=eval_model,
                        eval_set=eval_set,
                        genienlp_version=genienlp_version,
                        genie_version=genie_version,
                        thingtalk_version=thingtalk_version,
                        workdir_repo=workdir_repo,
                        workdir_version=workdir_version,
                        thingpedia_developer_key=thingpedia_developer_key,
                        additional_args=eval_additional_args)

    
@dsl.pipeline(
    name='Selftrain without paraphrase',
    description='Runs the whole training pipeline, including two parallel finetuning (for user and agent), auto annotation, and selftrain finetuning'
)
def selftrain_nopara_pipeline(
    owner,
    project,
    experiment,
    model,
    dataset,
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_additional_args='',
    train_iterations='80000',
    fewshot_train_iterations='20000',
    selftrain_train_iterations='20000',
    auto_annotate_additional_args='',
    eval_set='dev',
    eval_additional_args=''
):
    # first, generate the dataset
    generate_dataset_op = generate_dataset_step(image=image,
                                                    owner=owner,
                                                    project=project,
                                                    experiment=experiment,
                                                    dataset=dataset,
                                                    parallel=generate_dataset_parallel,
                                                    genie_version=genie_version,
                                                    thingtalk_version=thingtalk_version,
                                                    workdir_repo=workdir_repo,
                                                    workdir_version=workdir_version,
                                                    thingpedia_developer_key=thingpedia_developer_key,
                                                    additional_args=generate_dataset_additional_args)
    initial_datadir = generate_dataset_op.outputs['s3_datadir']
    
    # autoparaphrase and few-shot finetune the user model
    user_gen_datadir, user_model = paraphrase_fewshot_step(
        do_paraphrase=False,
        do_fewshot=True,
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        dataset=dataset,
        image=image,
        genienlp_version=genienlp_version,
        train_task_name='almond_dialogue_nlu',
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        train_s3_datadir=initial_datadir,
        train_load_from='None',
        train_dataset_subfolder='None',
        filtering_train_iterations='',
        filtering_batch_size='',
        fewshot_train_iterations=fewshot_train_iterations,
        ignore_context='',
        keep_original_duplicates='',
        paraphrasing_model='',
        paraphrase_subfolder='user',
        paraphrase_additional_args='',
        filtering_additional_args='',
    )
    
    # autoparaphrase and few-shot finetune the agent model
    agent_gen_datadir, agent_model = paraphrase_fewshot_step(
        do_paraphrase=False,
        do_fewshot=True,
        owner=owner,
        project=project,
        experiment=experiment,
        model='%s-agent' % (model,),
        dataset='%s-agent' % (dataset,),
        image=image,
        genienlp_version=genienlp_version,
        train_task_name='almond_dialogue_nlu_agent',
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        train_s3_datadir=initial_datadir,
        train_load_from='None',
        train_dataset_subfolder='None',
        filtering_train_iterations='',
        filtering_batch_size='',
        fewshot_train_iterations=fewshot_train_iterations,
        ignore_context='',
        keep_original_duplicates='',
        paraphrasing_model='',
        paraphrase_subfolder='agent',
        paraphrase_additional_args='',
        filtering_additional_args='',
    )
    
    auto_annotate_op = auto_annotate_step(image=image,
                                          owner=owner,
                                          project=project,
                                          experiment=experiment,
                                          dataset=dataset,
                                          user_model=user_model,
                                          agent_model=agent_model,
                                          genienlp_version=genienlp_version,       
                                          genie_version=genie_version,
                                          thingtalk_version=thingtalk_version,
                                          workdir_repo=workdir_repo,
                                          workdir_version=workdir_version,
                                          thingpedia_developer_key=thingpedia_developer_key,
                                          additional_args=auto_annotate_additional_args)
    selftrain_datadir = auto_annotate_op.outputs['s3_datadir']
    
    train_op = train_step(image=image,
                          owner=owner,
                          project=project,
                          experiment=experiment,
                          model='%s-selftrain' % (model,),
                          task_name='almond_dialogue_nlu',
                          load_from=user_model,
                          s3_datadir=selftrain_datadir,
                          dataset_subfolder='None',
                          genienlp_version=genienlp_version,
                          train_iterations=selftrain_train_iterations,
                          skip_tensorboard='false',
                          additional_args=train_additional_args)
    eval_model = train_op.outputs['s3_model_dir']
    
    eval_op = eval_step(image=image,
                        owner=owner,
                        project=project,
                        experiment=experiment,
                        model=model,
                        s3_model_dir=eval_model,
                        eval_set=eval_set,
                        genienlp_version=genienlp_version,
                        genie_version=genie_version,
                        thingtalk_version=thingtalk_version,
                        workdir_repo=workdir_repo,
                        workdir_version=workdir_version,
                        thingpedia_developer_key=thingpedia_developer_key,
                        additional_args=eval_additional_args)
    

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
    thingtalk_version=THINGTALK_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    eval_set='dev',
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
        thingtalk_version=thingtalk_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        thingpedia_developer_key=thingpedia_developer_key,
        eval_set=eval_set,
        additional_args=additional_args)
    
    
@dsl.pipeline(
    name='Train and eval TRADE',
    description='Train and evaluate a TRADE model'
)
def train_eval_trade(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    train_additional_args='',
    eval_additional_args=''
):
    train_env = {}
    
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train-trade.yaml')(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
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
    (add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*train_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))

    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-trade.yaml')(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_datadir=s3_datadir,
            s3_model_dir=train_op.outputs['s3_model_dir'],
            additional_args=eval_additional_args)
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)


@dsl.pipeline(
    name='Train and eval SimpleTOD',
    description='Train and evaluate a SimpleTOD model'
)
def train_eval_simpletod(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    train_additional_args='',
    eval_additional_args=''
):
    train_env = {}
    
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train-simpletod.yaml')(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
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
    (add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*train_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))

    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-trade.yaml')(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_datadir=s3_datadir,
            s3_model_dir=train_op.outputs['s3_model_dir'],
            additional_args=eval_additional_args)
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)


@dsl.pipeline(
    name='Train and eval SUMBT',
    description='Train and evaluate a SUMBT model'
)
def train_eval_sumbt(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    train_additional_args='',
    eval_additional_args=''
):
    train_env = {}
    
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train-sumbt.yaml')(
            owner=owner,
            project=project,
            experiment=experiment,
            model=model,
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
    (add_env(add_ssh_volume(train_op), train_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2*train_num_gpus}xlarge')
        .add_volume(V1Volume(name='tensorboard',
            persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))

    eval_env = {}

    eval_op = components.load_component_from_file('components/evaluate-sumbt.yaml')(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_datadir=s3_datadir,
            s3_model_dir=train_op.outputs['s3_model_dir'],
            additional_args=eval_additional_args)
    (eval_op.container
        .set_memory_limit('15Gi')
        .set_memory_request('15Gi')
        .set_cpu_limit('4')
        .set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)


def train_spl_step(
        owner='mehrad',
        project='multilanguage',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        dlg_side='user',
        s3_bootleg_prepped_data='None',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        bootleg_version='f53e67397ddcd099f3a18a014c9ce82b02d2223c',
        load_from='None',
        train_languages='es',
        eval_languages='es',
        eval_set='eval',
        dataset_subfolder='None',
        skip_tensorboard='false',
        train_iterations='',
        additional_args='--dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
):
    
    train_env = {
        'GENIENLP_VERSION': genienlp_version,
        'BOOTLEG_VERSION': bootleg_version
    }
    
    train_num_gpus = 1
    train_op = components.load_component_from_file('components/train-spl.yaml')(
        image=image,
        s3_bucket=s3_bucket,
        owner=owner,
        task_name=task_name,
        project=project,
        experiment=experiment,
        model=model,
        load_from=load_from,
        eval_set=eval_set,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        dataset_subfolder=dataset_subfolder,
        train_iterations=train_iterations,
        skip_tensorboard=skip_tensorboard,
        train_languages=train_languages,
        eval_languages=eval_languages,
        dlg_side=dlg_side,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=additional_args,)
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
    
    return train_op


def eval_spl_step(
        owner='mehrad',
        project='multilanguage',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
        workdir_version='master',
        pred_languages='es',
        eval_set='eval',
        annotated_set_name='annotated',
        is_oracle='false',
        additional_args='--evaluate valid --overwrite'
):
    eval_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    
    eval_op = components.load_component_from_file('components/evaluate-spl.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        eval_set=eval_set,
        annotated_set_name=annotated_set_name,
        is_oracle=is_oracle,
        pred_languages=pred_languages,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_model_dir=s3_model_dir,
        additional_args=additional_args)
    (eval_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(eval_op), eval_env)
    
    return eval_op


def bootleg_step(
        owner='mehrad',
        project='multilanguage',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        dlg_side='user',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        bootleg_version='f53e67397ddcd099f3a18a014c9ce82b02d2223c',
        load_from='None',
        train_languages='es',
        eval_languages='es',
        eval_set='eval',
        dataset_subfolder='None',
        skip_tensorboard='false',
        additional_args='',
):
    bootleg_env = {
        'GENIENLP_VERSION': genienlp_version,
        'BOOTLEG_VERSION': bootleg_version
    }
    
    bootleg_num_gpus = 1
    bootleg_op = components.load_component_from_file('components/bootleg.yaml')(
        image=image,
        s3_bucket=s3_bucket,
        owner=owner,
        task_name=task_name,
        project=project,
        experiment=experiment,
        model=model,
        load_from=load_from,
        eval_set=eval_set,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_languages=train_languages,
        eval_languages=eval_languages,
        dlg_side=dlg_side,
        additional_args=additional_args
    )
    (bootleg_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(bootleg_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     )
    (add_env(add_ssh_volume(bootleg_op), bootleg_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * bootleg_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))
    
    return bootleg_op


@dsl.pipeline(
    name='Eval SPL',
    description='Evaluate a model for SPL experiments'
)
def eval_spl(
        owner='mehrad',
        project='multilanguage',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/SPL.git',
        workdir_version='master',
        pred_languages='es',
        eval_set='eval',
        annotated_set_name='annotated',
        is_oracle='false',
        eval_additional_args='--evaluate valid --overwrite'
):
    eval_op = eval_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        thingtalk_version=thingtalk_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        pred_languages=pred_languages,
        eval_set=eval_set,
        annotated_set_name=annotated_set_name,
        is_oracle=is_oracle,
        s3_model_dir=s3_model_dir,
        additional_args=eval_additional_args
    )


@dsl.pipeline(
    name='Train and eval SPL',
    description='Train and evaluate pipeline for SPL experiments'
)
def train_eval_spl(
        owner='mehrad',
        project='multilanguage',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        dlg_side='user',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/SPL.git',
        workdir_version='master',
        load_from='None',
        train_languages='es',
        eval_languages='es',
        pred_languages='es',
        eval_set='eval',
        dataset_subfolder='None',
        annotated_set_name='annotated',
        is_oracle='false',
        skip_tensorboard='false',
        train_iterations='',
        train_additional_args='--dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
        eval_additional_args='--evaluate valid --overwrite'
):
    
    train_op = train_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        dlg_side=dlg_side,
        image=image,
        genienlp_version=genienlp_version,
        load_from=load_from,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        additional_args=train_additional_args
    )

    eval_op = eval_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        thingtalk_version=thingtalk_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        pred_languages=pred_languages,
        eval_set=eval_set,
        annotated_set_name=annotated_set_name,
        is_oracle=is_oracle,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args
    )

@dsl.pipeline(
    name='Train and eval Genie+Bootleg',
    description='Train and evaluate pipeline for Bootleg experiments'
)
def bootleg_train_eval(
        owner='mehrad',
        project='schemaorg',
        experiment='restaurants',
        model='',
        task_name='almond',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='s3://geniehai/mehrad/extras/bootleg_material/',
        dlg_side='user',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-bootleg',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
        workdir_version='master',
        bootleg_version='f53e67397ddcd099f3a18a014c9ce82b02d2223c',
        load_from='None',
        train_languages='en',
        eval_languages='en',
        pred_languages='en',
        eval_set='eval',
        dataset_subfolder='None',
        annotated_set_name='annotated',
        is_oracle='false',
        skip_tensorboard='false',
        train_iterations='',
        bootleg_additional_args='--do_ner --retrieve_method bootleg --lookup_method ngrams --features type freq --features_size 3 3 --features_default_val 0 1.0 --num_workers 0 --bootleg_input_dir bootleg_material --bootleg_model bootleg_wiki_types --entity_type_agg_method weighted',
        train_additional_args='--do_ner --retrieve_method bootleg --lookup_method ngrams --features type freq --features_size 3 3 --features_default_val 0 1.0 --num_workers 0 --bootleg_skip_feature_creation --bootleg_input_dir bootleg_material --bootleg_model bootleg_wiki_types --entity_type_agg_method weighted --dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
        eval_additional_args='--evaluate valid --overwrite'
):

    bootleg_op = bootleg_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        dlg_side=dlg_side,
        image=image,
        genienlp_version=genienlp_version,
        bootleg_version=bootleg_version,
        load_from=load_from,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        additional_args=bootleg_additional_args
    )

    train_op = train_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        dlg_side=dlg_side,
        image=image,
        genienlp_version=genienlp_version,
        bootleg_version=bootleg_version,
        load_from=load_from,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=bootleg_op.outputs['s3_bootleg_prepped_data'],
        additional_args=train_additional_args
    )


    eval_op = eval_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        thingtalk_version=thingtalk_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        pred_languages=pred_languages,
        eval_set=eval_set,
        annotated_set_name=annotated_set_name,
        is_oracle=is_oracle,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args
    )


def prepare_for_translation_step(
        owner='mehrad',
        project='schemaorg',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond_multilingual',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        dlg_side='user',
        prepare_for_translation='true',
        do_translation='false',
        post_process_translation='false',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
        workdir_version='master',
        additional_args=''

):
    prepare_for_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    
    prepare_for_translation_op = components.load_component_from_file('components/translate.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dlg_side=dlg_side,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        task_name=task_name,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (prepare_for_translation_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(prepare_for_translation_op), prepare_for_translation_env)
    
    return prepare_for_translation_op


def do_translation_step(
        owner='mehrad',
        project='schemaorg',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond_multilingual',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        dlg_side='user',
        prepare_for_translation='false',
        do_translation='true',
        post_process_translation='false',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
        workdir_version='master',
        additional_args=''

):
    do_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }

    do_translation_num_gpus=1
    do_translation_op = components.load_component_from_file('components/translate.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dlg_side=dlg_side,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        task_name=task_name,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (do_translation_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(do_translation_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     )
    (add_env(add_ssh_volume(do_translation_op), do_translation_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * do_translation_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))
    
    return do_translation_op


def post_process_translation_step(
        owner='mehrad',
        project='schemaorg',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond_multilingual',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        dlg_side='user',
        prepare_for_translation='false',
        do_translation='false',
        post_process_translation='true',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
        workdir_version='master',
        additional_args=''

):
    post_process_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    
    post_process_translation_op = components.load_component_from_file('components/translate.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        dlg_side=dlg_side,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        task_name=task_name,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (post_process_translation_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(post_process_translation_op), post_process_translation_env)
    
    return post_process_translation_op


@dsl.pipeline(
    name='Translation',
    description='Generate dataset using NMT'
)
def translate(
        owner='mehrad',
        project='schemaorg',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        dlg_side='user',
        prepare_for_translation='false',
        do_translation='false',
        post_process_translation='false',
        image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit:latest-mehrad-spl',
        genienlp_version='c6ffb08742fed0c414d6ffc5eeae679cabdb20ff',
        genie_version='5847c1941948fde5bb1ad3a5b2fefb0f841cd86c',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
        workdir_version='master',
        additional_args='--temperature 0.4 --repetition_penalty 1.0 --num_samples 1 --batch_size 512  --skip_heuristics --att_pooling mean --task translate'

):
    if prepare_for_translation:
        prepare_for_translation_op = prepare_for_translation_step(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_bucket=s3_bucket,
            task_name=task_name,
            s3_datadir=s3_datadir,
            model_name_or_path=model_name_or_path,
            input_splits=input_splits,
            train_output_per_example=train_output_per_example,
            nmt=nmt,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dlg_side=dlg_side,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
        )
    
    if do_translation:
        do_translation_op = do_translation_step(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_bucket=s3_bucket,
            task_name=task_name,
            s3_datadir=s3_datadir,
            model_name_or_path=model_name_or_path,
            input_splits=input_splits,
            train_output_per_example=train_output_per_example,
            nmt=nmt,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dlg_side=dlg_side,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
        )
    
        if prepare_for_translation:
            do_translation_op.after(prepare_for_translation_op)
    
    if post_process_translation:
        post_process_translation_op = post_process_translation_step(
                owner=owner,
                project=project,
                experiment=experiment,
                s3_bucket=s3_bucket,
                task_name=task_name,
                s3_datadir=s3_datadir,
                model_name_or_path=model_name_or_path,
                input_splits=input_splits,
                train_output_per_example=train_output_per_example,
                nmt=nmt,
                do_alignment=do_alignment,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dlg_side=dlg_side,
                image=image,
                genienlp_version=genienlp_version,
                genie_version=genie_version,
                thingtalk_version=thingtalk_version,
                workdir_repo=workdir_repo,
                workdir_version=workdir_version,
                additional_args=additional_args
            )
    
        if prepare_for_translation and do_translation:
            post_process_translation_op.after(do_translation_op, prepare_for_translation_op)
        elif do_translation:
            post_process_translation_op.after(do_translation_op)
