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


GENIENLP_VERSION = '00905bd4c1f67f8ba2577a7f8a09f093d024d061'
GENIE_VERSION = '84877f2488a0d0dea1e81f3e1f0b92dc6c05c568'
THINGTALK_VERSION = '2338ef36c76c90a84ef0942d021f909e47c6307f'
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
        .set_cpu_limit('8')
        .set_cpu_request('7'))
    add_env(add_ssh_volume(eval_op), eval_env)

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
    paraphrasing_model,
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
        paraphrasing_model=paraphrasing_model,
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
    fewshot_train_iterations='20000',
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
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
    
    if do_paraphrase:
        pretrain_op = train_step(image=image,
                                 owner=owner,
                                 project=project,
                                 experiment=experiment,
                                 model=model,
                                 task_name=train_task_name,
                                 load_from=train_load_from,
                                 s3_datadir=train_s3_datadir,
                                 dataset_subfolder='None',
                                 genienlp_version=genienlp_version,
                                 train_iterations=filtering_train_iterations,
                                 skip_tensorboard='true',
                                 additional_args=train_additional_args)
        if do_generate:
            pretrain_op.after(generate_dataset_op)

        paraphrase_generation_op = paraphrase_generation_step(image=image,
                                        owner=owner,
                                        project=project,
                                        experiment=experiment,
                                        dataset=dataset,
                                        s3_input_datadir=generate_dataset_op.outputs['s3_datadir'],
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
                                        paraphrasing_model=paraphrasing_model,
                                        genienlp_version=genienlp_version,
                                        paraphrase_subfolder=paraphrase_subfolder,
                                        additional_args=paraphrase_additional_args)
        if do_generate:
            paraphrase_generation_op.after(generate_dataset_op)
        paraphrase_filtering_op.after(paraphrase_generation_op, pretrain_op)
        
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
    if do_paraphrase:
        train_op.after(paraphrase_filtering_op)
    elif do_generate:
        train_op.after(generate_dataset_op)
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
        fewshot_op.after(train_op)
        eval_model = fewshot_op.outputs['s3_model_dir']
    
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
    if do_fewshot:
        eval_op.after(fewshot_op)
    else:
        eval_op.after(train_op)


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
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
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
               ignore_context=ignore_context,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
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
    ignore_context='true',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
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
               ignore_context=ignore_context,
               keep_original_duplicates=keep_original_duplicates,
               paraphrasing_model=paraphrasing_model,
               paraphrase_subfolder=paraphrase_subfolder,
               paraphrase_additional_args=paraphrase_additional_args,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)