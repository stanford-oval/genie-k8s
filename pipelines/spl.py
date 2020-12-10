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
    V1VolumeMount,
    V1Volume,
    V1PersistentVolumeClaimVolumeSource,
    V1SecretVolumeSource
)

from .common import *

from .training import train_step


def eval_spl_step(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
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


@dsl.pipeline(
    name='Eval SPL',
    description='Evaluate a model for SPL experiments'
)
def eval_spl(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_model_dir='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/SPL.git',
        workdir_version=GENIE_WORKDIR_VERSION,
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
        project='spl',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo='git@github.com:stanford-oval/SPL.git',
        workdir_version=GENIE_WORKDIR_VERSION,
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
        bootleg_model='',
        bootleg_version='',
        s3_bootleg_prepped_data='',
        train_additional_args='--dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
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
        bootleg_version=bootleg_version,
        load_from=load_from,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
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
        project='spl',
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
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
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
        prepare_for_translation='true',
        do_translation='false',
        post_process_translation='false',
        task_name=task_name,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (prepare_for_translation_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(prepare_for_translation_op), prepare_for_translation_env)

    prepare_for_translation_op.name = 'prepare-for-translation'

    return prepare_for_translation_op


def do_translation_step(
        owner='mehrad',
        project='spl',
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
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
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
        prepare_for_translation='false',
        do_translation='true',
        post_process_translation='false',
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

    do_translation_op.name = 'translation'

    return do_translation_op


def post_process_translation_step(
        owner='mehrad',
        project='spl',
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
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
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
        prepare_for_translation='false',
        do_translation='false',
        post_process_translation='true',
        task_name=task_name,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (post_process_translation_op.container
     .set_memory_limit('15Gi')
     .set_memory_request('15Gi')
     .set_cpu_limit('4')
     .set_cpu_request('4'))
    add_env(add_ssh_volume(post_process_translation_op), post_process_translation_env)

    post_process_translation_op.name = 'post-process-translation'

    return post_process_translation_op


def all_translation_steps(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond',
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
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
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
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=''
        )
        prepare_for_translation_op.container.set_image_pull_policy('Always')

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
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
        )
        do_translation_op.container.set_image_pull_policy('Always')

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
                image=image,
                genienlp_version=genienlp_version,
                genie_version=genie_version,
                thingtalk_version=thingtalk_version,
                workdir_repo=workdir_repo,
                workdir_version=workdir_version,
                additional_args=''
            )
        post_process_translation_op.container.set_image_pull_policy('Always')

        if prepare_for_translation and do_translation:
            post_process_translation_op.after(do_translation_op, prepare_for_translation_op)
        elif do_translation:
            post_process_translation_op.after(do_translation_op)

@dsl.pipeline(
    name='Translate a dataset',
    description='Prepare, Translate, and Postprocess dataset'
)
def translate(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond',
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
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args='--temperature 0.4 --repetition_penalty 1.0 --num_samples 1 --batch_size 512  --skip_heuristics --att_pooling mean --task translate'
):
    all_translation_steps(
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
            prepare_for_translation=True,
            do_translation=True,
            post_process_translation=True,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
    )


def paraphrase_step(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond_multilingual',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        pivot_lang='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        paraphrasing_method='false',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):

    paraphrase_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
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
        pivot_lang=pivot_lang,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        do_paraphrasing='true',
        paraphrasing_method=paraphrasing_method,
        filter_sts_paraphrase='false',
        task_name=task_name,
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

    paraphrase_op.name = 'sts-paraphrasing'

    return paraphrase_op


def sts_filtering_step(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond_multilingual',
        s3_datadir='',
        model_name_or_path='',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='',
        pivot_lang='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    sts_filtering_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'THINGTALK_VERSION': thingtalk_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }

    sts_filtering_num_gpus = 1
    sts_filtering_op = components.load_component_from_file('components/sts-paraphrase.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        model_name_or_path=model_name_or_path,
        input_splits=input_splits,
        train_output_per_example=train_output_per_example,
        nmt=nmt,
        pivot_lang=pivot_lang,
        do_alignment=do_alignment,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        do_paraphrasing='false',
        paraphrasing_method='',
        filter_sts_paraphrase='true',
        task_name=task_name,
        s3_datadir=s3_datadir,
        additional_args=additional_args)
    (sts_filtering_op.container
     .set_memory_request('56Gi')
     .set_memory_limit('56Gi')
     .set_cpu_request('7.5')
     .set_cpu_limit('7.5')
     .set_gpu_limit(str(sts_filtering_num_gpus))
     .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
     )
    (add_env(add_ssh_volume(sts_filtering_op), sts_filtering_env)
     .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
     .add_node_selector_constraint('beta.kubernetes.io/instance-type', f'p3.{2 * sts_filtering_num_gpus}xlarge')
     .add_volume(V1Volume(name='tensorboard',
                          persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf'))))

    sts_filtering_op.name = 'sts-filtering'

    return sts_filtering_op

def all_paraphrasing_steps(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond',
        s3_datadir='',
        model_name_or_path='facebook/mbart-large-cc25',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='marian',
        pivot_lang='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        do_paraphrasing=True,
        paraphrasing_method='',
        filter_sts_paraphrase=True,
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args=''
):
    if do_paraphrasing:
        paraphrase_op = paraphrase_step(
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
            pivot_lang=pivot_lang,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            paraphrasing_method=paraphrasing_method,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
        )
        paraphrase_op.container.set_image_pull_policy('Always')

    if filter_sts_paraphrase:
        sts_filtering_op = sts_filtering_step(
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
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
        )
        sts_filtering_op.container.set_image_pull_policy('Always')

        if do_paraphrasing:
            sts_filtering_op.after(paraphrase_op)


@dsl.pipeline(
    name='Round-trip Paraphrasing',
    description='Use round-trip translation to generate paraphrases and use STS to filter them'
)
def round_trip_paraphrasing(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond',
        s3_datadir='',
        model_name_or_path='facebook/mbart-large-cc25',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='marian',
        pivot_lang='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args='--temperature 0.4 --repetition_penalty 1.0 --num_samples 1 --batch_size 512  --skip_heuristics --att_pooling mean --id_column 0  --input_column 1 --gold_column 1 --return_attentions --output_example_ids_too --task translate --return_attentions'
):
    all_paraphrasing_steps(
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
            pivot_lang=pivot_lang,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            do_paraphrasing=True,
            paraphrasing_method='round_trip',
            filter_sts_paraphrase=True,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
    )

@dsl.pipeline(
    name='Masked Paraphrasing',
    description='Use denoisng models (e.g. BART family) to generate paraphrases and use STS to filter them'
)
def masked_paraphrasing(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        s3_bucket='geniehai',
        task_name='almond',
        s3_datadir='',
        model_name_or_path='facebook/mbart-large-cc25',
        input_splits='test+eval+train',
        train_output_per_example='1',
        nmt='marian',
        pivot_lang='',
        do_alignment='true',
        src_lang='en',
        tgt_lang='',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        additional_args='--infill_text --num_text_spans 1 --temperature 0.4 --repetition_penalty 1.0 --num_samples 1 --batch_size 512  --skip_heuristics --att_pooling mean --id_column 0  --input_column 1 --gold_column 1 --return_attentions --output_example_ids_too --task paraphrase --return_attentions'
):
    all_paraphrasing_steps(
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
            pivot_lang=pivot_lang,
            do_alignment=do_alignment,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            do_paraphrasing=True,
            paraphrasing_method='masked',
            filter_sts_paraphrase=True,
            image=image,
            genienlp_version=genienlp_version,
            genie_version=genie_version,
            thingtalk_version=thingtalk_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            additional_args=additional_args
    )