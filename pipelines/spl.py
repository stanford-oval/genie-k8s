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
    V1PersistentVolumeClaimVolumeSource
)

from .common import *
from .training import train_step, train_step_4gpus


#############################
#####  Training & Evaluation
#############################

def eval_spl_step(
        owner='mehrad',
        project='spl',
        experiment='',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        s3_database_dir='None',
        bootleg_model='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        pred_languages='',
        eval_set='eval',
        annotated_set_name='annotated',
        is_oracle='false',
        additional_args='--evaluate valid --overwrite'
):
    eval_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
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
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        additional_args=additional_args)
    (eval_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit('1')
     )
    (add_env(add_ssh_volume(eval_op), eval_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.2xlarge'))
    
    eval_op.container.set_image_pull_policy('Always')
    
    return eval_op


def eval_spl_step_cpu(
        owner='mehrad',
        project='spl',
        experiment='',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        s3_database_dir='None',
        bootleg_model='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        pred_languages='',
        eval_set='eval',
        annotated_set_name='annotated',
        is_oracle='false',
        additional_args='--evaluate valid --overwrite'
):
    eval_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
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
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        additional_args=additional_args)
    (eval_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     )
    (add_env(add_ssh_volume(eval_op), eval_env))
    
    eval_op.container.set_image_pull_policy('Always')
    
    return eval_op



@dsl.pipeline(
    name='Eval SPL',
    description='Evaluate a model for SPL experiments'
)
def eval_spl(
        owner='mehrad',
        project='spl',
        experiment='',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        s3_database_dir='None',
        bootleg_model='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        pred_languages='',
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
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
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
    name='Eval SPL on CPU',
    description='Evaluate a model for SPL experiments'
)
def eval_spl_cpu(
        owner='mehrad',
        project='spl',
        experiment='',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        s3_database_dir='None',
        bootleg_model='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        pred_languages='',
        eval_set='eval',
        annotated_set_name='annotated',
        is_oracle='false',
        eval_additional_args='--evaluate valid --overwrite'
):
    eval_op = eval_spl_step_cpu(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
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
        project='spl',
        experiment='',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        load_from='None',
        train_languages='',
        eval_languages='',
        pred_languages='',
        eval_set='',
        dataset_subfolder='None',
        annotated_set_name='annotated',
        is_oracle='false',
        skip_tensorboard='false',
        train_iterations='',
        bootleg_model='None',
        s3_bootleg_prepped_data='None',
        train_additional_args='',
        eval_additional_args='--evaluate valid --overwrite'
):

    train_op = train_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        load_from=load_from,
        train_languages=train_languages,
        eval_languages=eval_languages,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args
    )
    
    eval_op = eval_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
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
    name='Train and eval SPL on 4 gpus',
    description='Train and evaluate pipeline for SPL experiments'
)
def train_eval_spl_4gpus(
        owner='mehrad',
        project='spl',
        experiment='',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        load_from='None',
        train_languages='',
        eval_languages='',
        pred_languages='',
        eval_set='',
        dataset_subfolder='None',
        annotated_set_name='annotated',
        is_oracle='false',
        skip_tensorboard='false',
        train_iterations='',
        bootleg_model='None',
        s3_bootleg_prepped_data='None',
        train_additional_args='',
        eval_additional_args='--evaluate valid --overwrite'
):
    train_op = train_step_4gpus(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        load_from=load_from,
        train_languages=train_languages,
        eval_languages=eval_languages,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args
    )
    
    eval_op = eval_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        pred_languages=pred_languages,
        eval_set=eval_set,
        annotated_set_name=annotated_set_name,
        is_oracle=is_oracle,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args
    )


#############################
#####  Translation
#############################

def prepare_for_translation_step(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''

):
    prepare_for_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
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
        prepare_for_translation='true',
        do_translation='false',
        post_process_translation='false',
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (prepare_for_translation_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(prepare_for_translation_op), prepare_for_translation_env)
    
    prepare_for_translation_op.name = 'prepare-for-translation'
    
    prepare_for_translation_op.container.set_image_pull_policy('Always')
    
    return prepare_for_translation_op


def do_translation_step(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    do_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    
    do_translation_num_gpus = 1
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
        prepare_for_translation='false',
        do_translation='true',
        post_process_translation='false',
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
    
    do_translation_op.name = 'translation'
    
    do_translation_op.container.set_image_pull_policy('Always')
    
    return do_translation_op


def post_process_translation_step(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''

):
    post_process_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
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
        prepare_for_translation='false',
        do_translation='false',
        post_process_translation='true',
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (post_process_translation_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(post_process_translation_op), post_process_translation_env)
    
    post_process_translation_op.name = 'post-process-translation'
    
    post_process_translation_op.container.set_image_pull_policy('Always')
    
    return post_process_translation_op


def all_translation_steps(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='Helsinki-NLP/opus-mt-en-{}',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='marian',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        prepare_for_translation=True,
        do_translation=True,
        post_process_translation=True,
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''

):
    if prepare_for_translation:
        prepare_for_translation_op = prepare_for_translation_step(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_bucket=s3_bucket,
            s3_datadir=s3_datadir,
            model_name_or_path=model_name_or_path,
            input_splits=input_splits,
            train_output_per_example=train_output_per_example,
            nmt=nmt,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=''
        )
    
    if do_translation:
        do_translation_op = do_translation_step(
            owner=owner,
            project=project,
            experiment=experiment,
            s3_bucket=s3_bucket,
            s3_datadir=s3_datadir,
            model_name_or_path=model_name_or_path,
            input_splits=input_splits,
            train_output_per_example=train_output_per_example,
            nmt=nmt,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
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
            s3_datadir=s3_datadir,
            model_name_or_path=model_name_or_path,
            input_splits=input_splits,
            train_output_per_example=train_output_per_example,
            nmt=nmt,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=''
        )
        
        if prepare_for_translation and do_translation:
            post_process_translation_op.after(do_translation_op, prepare_for_translation_op)
        elif do_translation:
            post_process_translation_op.after(do_translation_op)


@dsl.pipeline(
    name='Prepare and translate a dataset',
    description='Prepare, Translate, and Postprocess dataset'
)
def prepare_translate_process(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='Helsinki-NLP/opus-mt-en-{}',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='marian',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    all_translation_steps(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prepare_for_translation=True,
        do_translation=True,
        post_process_translation=True,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )




@dsl.pipeline(
    name='Translate a dataset',
    description='Translate, and Postprocess dataset'
)
def translate_process(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='Helsinki-NLP/opus-mt-en-{}',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='marian',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    all_translation_steps(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prepare_for_translation=False,
        do_translation=True,
        post_process_translation=True,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )


#############################
#####  Paraphrasing
#############################

def paraphrase_step(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='None',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        pivot_langs='None',
        marian_group_langs='None',
        do_alignment='true',
        tgt_lang='',
        sts_batch_size='',
        sts_model='',
        filtering_metric='',
        filtering_threshold='',
        train_fewshot='false',
        do_paraphrasing='true',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    
    paraphrase_num_gpus = 1
    paraphrase_op = components.load_component_from_file('components/sts-paraphrase.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs=pivot_langs,
        marian_group_langs=marian_group_langs,
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        train_fewshot=train_fewshot,
        do_paraphrasing=do_paraphrasing,
        prepare_paraphrases=prepare_paraphrases,
        filter_paraphrases=filter_paraphrases,
        paraphrasing_method=paraphrasing_method,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (paraphrase_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(paraphrase_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     )
    (add_env(add_ssh_volume(paraphrase_op), paraphrase_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * paraphrase_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))
    
    paraphrase_op.container.set_image_pull_policy('Always')
    
    return paraphrase_op


def paraphrase_step_4gpus(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='None',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        pivot_langs='None',
        marian_group_langs='None',
        do_alignment='true',
        tgt_lang='',
        sts_batch_size='',
        sts_model='',
        filtering_metric='',
        filtering_threshold='',
        train_fewshot='false',
        do_paraphrasing='true',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    
    paraphrase_num_gpus = 4
    paraphrase_op = components.load_component_from_file('components/sts-paraphrase.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs=pivot_langs,
        marian_group_langs=marian_group_langs,
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        train_fewshot=train_fewshot,
        do_paraphrasing=do_paraphrasing,
        prepare_paraphrases=prepare_paraphrases,
        filter_paraphrases=filter_paraphrases,
        paraphrasing_method=paraphrasing_method,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (paraphrase_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(paraphrase_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     )
    (add_env(add_ssh_volume(paraphrase_op), paraphrase_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * paraphrase_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))
    
    paraphrase_op.container.set_image_pull_policy('Always')
    
    return paraphrase_op


@dsl.pipeline(
    name='Paraphrase + STS filter + train + eval',
    description='Full multilingual paraphrase pipeline'
)
def multilingual_paraphrasing(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        task_name='almond',
        s3_datadir='',
        s3_database_dir='None',
        model_name_or_path='None',
        model='',
        bootleg_model='None',
        load_from='None',
        dataset_subfolder='None',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        pivot_langs='None',
        marian_group_langs='None',
        do_alignment='true',
        tgt_lang='',
        sts_batch_size='250',
        sts_model='xlm-r-distilroberta-base-paraphrase-v1',
        filtering_metric='',
        filtering_threshold='',
        train_fewshot='',
        do_paraphrasing='true',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        skip_tensorboard='false',
        train_iterations='',
        s3_bootleg_prepped_data='None',
        annotated_set_name='annotated',
        eval_set='',
        is_oracle='false',
        paraphrase_additional_args='',
        train_additional_args='',
        eval_additional_args=''
):
    paraphrase_op = paraphrase_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs=pivot_langs,
        marian_group_langs=marian_group_langs,
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        train_fewshot=train_fewshot,
        do_paraphrasing=do_paraphrasing,
        prepare_paraphrases=prepare_paraphrases,
        filter_paraphrases=filter_paraphrases,
        paraphrasing_method=paraphrasing_method,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=paraphrase_additional_args
    )

    train_op = train_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=paraphrase_op.outputs['s3_output_datadir'],
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        load_from=load_from,
        train_languages=tgt_lang,
        eval_languages=tgt_lang,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args
    )

    eval_op = eval_spl_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        pred_languages=tgt_lang,
        eval_set=eval_set,
        annotated_set_name=annotated_set_name,
        is_oracle=is_oracle,
        s3_model_dir=train_op.outputs['s3_model_dir'],
        additional_args=eval_additional_args
    )
    


@dsl.pipeline(
    name='Round-trip Paraphrasing + STS filtering',
    description='Use round-trip translation to generate paraphrases and use STS to filter them'
)
def round_trip_paraphrasing(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        pivot_langs='',
        marian_group_langs='',
        do_alignment='true',
        tgt_lang='',
        sts_batch_size='250',
        sts_model='xlm-r-distilroberta-base-paraphrase-v1',
        filtering_metric='',
        filtering_threshold='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs=pivot_langs,
        marian_group_langs=marian_group_langs,
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        do_paraphrasing='true',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method='round_trip',
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )


@dsl.pipeline(
    name='Masked Paraphrasing + STS filtering',
    description='Use denoisng models (e.g. BART family) to generate paraphrases and use STS to filter them'
)
def masked_paraphrasing(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='facebook/mbart-large-cc25',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        do_alignment='true',
        tgt_lang='',
        sts_batch_size='250',
        sts_model='xlm-r-distilroberta-base-paraphrase-v1',
        filtering_metric='',
        filtering_threshold='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs='masked',
        marian_group_langs='None',
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        do_paraphrasing='true',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method='masked',
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )



@dsl.pipeline(
    name='Masked Paraphrasing + STS filtering on 4 gpus',
    description='Use denoisng models (e.g. BART family) to generate paraphrases and use STS to filter them'
)
def masked_paraphrasing_4gpus(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='facebook/mbart-large-cc25',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        do_alignment='true',
        tgt_lang='',
        sts_batch_size='250',
        sts_model='xlm-r-distilroberta-base-paraphrase-v1',
        filtering_metric='',
        filtering_threshold='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_step_4gpus(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs='masked',
        marian_group_langs='None',
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        do_paraphrasing='true',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method='masked',
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )


@dsl.pipeline(
    name='STS Prepare',
    description='Prepare paraphrases for STS filtering and then filter them'
)
def sts_prepare_paraphrases(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        pivot_langs='',
        marian_group_langs='',
        do_alignment='true',
        paraphrasing_method='',
        tgt_lang='',
        sts_batch_size='250',
        sts_model='xlm-r-distilroberta-base-paraphrase-v1',
        filtering_metric='',
        filtering_threshold='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs=pivot_langs,
        marian_group_langs=marian_group_langs,
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        do_paraphrasing='false',
        prepare_paraphrases='true',
        filter_paraphrases='true',
        paraphrasing_method=paraphrasing_method,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )



@dsl.pipeline(
    name='STS Filtering',
    description='Use STS score to filter paraphrases'
)
def sts_filtering(
        owner='mehrad',
        project='spl',
        experiment='',
        s3_bucket='geniehai',
        s3_datadir='',
        model_name_or_path='',
        input_splits='train',
        train_output_per_example='1',
        nmt='marian',
        pivot_langs='',
        marian_group_langs='',
        do_alignment='true',
        paraphrasing_method='',
        tgt_lang='',
        sts_batch_size='250',
        sts_model='xlm-r-distilroberta-base-paraphrase-v1',
        filtering_metric='',
        filtering_threshold='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    paraphrase_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_langs=pivot_langs,
        marian_group_langs=marian_group_langs,
        do_alignment=do_alignment,
        tgt_lang=tgt_lang,
        sts_batch_size=sts_batch_size,
        sts_model=sts_model,
        filtering_metric=filtering_metric,
        filtering_threshold=filtering_threshold,
        do_paraphrasing='false',
        prepare_paraphrases='false',
        filter_paraphrases='true',
        paraphrasing_method=paraphrasing_method,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        additional_args=additional_args
    )
