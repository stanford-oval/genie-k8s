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

from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1VolumeMount,
    V1Volume,
    V1PersistentVolumeClaimVolumeSource,
    V1SecretVolumeSource
)

from .common import *

# FIXME why is this using SPL at all?
from .spl import eval_spl_step


def bootleg_step(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        model='',
        task_name='almond_multilingual',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='None',
        dlg_side='user',
        bootleg_model='',
        image=default_image,
        genienlp_version='',
        bootleg_version='',
        train_languages='es',
        eval_languages='es',
        eval_set='eval',
        dataset_subfolder='None',
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
        eval_set=eval_set,
        s3_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        dataset_subfolder=dataset_subfolder,
        train_languages=train_languages,
        eval_languages=eval_languages,
        dlg_side=dlg_side,
        bootleg_model=bootleg_model,
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
    name='NED + Train + eval',
    description='Disambiguate, train, and evaluate for Bootleg experiments'
)
def ned_train_eval(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        model='',
        task_name='almond',
        s3_datadir='',
        s3_bucket='geniehai',
        s3_database_dir='s3://geniehai/mehrad/extras/bootleg_material/',
        dlg_side='user',
        do_ner='true',
        bootleg_model='bootleg_wiki_types',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        bootleg_version='',
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
        bootleg_additional_args='--almond_has_multiple_programs --do_ner --retrieve_method bootleg --lookup_method ngrams --features type freq --features_size 3 3 --features_default_val 0 1.0 --num_workers 0 --bootleg_integration 1 --entity_type_agg_method weighted --dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
        train_additional_args='--almond_has_multiple_programs --do_ner --retrieve_method bootleg --lookup_method ngrams --features type freq --features_size 3 3 --features_default_val 0 1.0 --num_workers 0 --bootleg_skip_feature_creation --bootleg_integration 1 --entity_type_agg_method weighted --dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
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
        bootleg_model=bootleg_model,
        image=image,
        genienlp_version=genienlp_version,
        bootleg_version=bootleg_version,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        dataset_subfolder=dataset_subfolder,
        additional_args=bootleg_additional_args
    )

    train_op = train_step(
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        dlg_side=dlg_side,
        do_ner=do_ner,
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



@dsl.pipeline(
    name='Train and eval for bootleg',
    description='Only Train and evaluate for Bootleg experiments'
)
def train_eval_bootleg(
        owner='mehrad',
        project='spl',
        experiment='restaurants',
        model='',
        task_name='almond',
        s3_datadir='',
        s3_bootleg_prepped_data='',
        s3_bucket='geniehai',
        s3_database_dir='s3://geniehai/mehrad/extras/bootleg_material/',
        dlg_side='user',
        do_ner='true',
        bootleg_model='bootleg_wiki_types',
        image=default_image,
        genienlp_version='',
        genie_version='',
        thingtalk_version=THINGTALK_VERSION,
        workdir_repo=GENIE_WORKDIR_REPO,
        workdir_version=GENIE_WORKDIR_VERSION,
        bootleg_version='',
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
        train_additional_args='--almond_has_multiple_programs --do_ner --retrieve_method bootleg --lookup_method ngrams --features type freq --features_size 3 3 --features_default_val 0 1.0 --num_workers 0 --bootleg_skip_feature_creation --bootleg_integration 1 --entity_type_agg_method weighted --dimension 768 --transformer_hidden 768 --trainable_decoder_embeddings 50 --encoder_embeddings=xlm-roberta-base --decoder_embeddings= --seq2seq_encoder=Identity --rnn_layers 1 --transformer_heads 12 --transformer_layers 0 --rnn_zero_state=average --train_encoder_embeddings --transformer_lr_multiply 0.08 --max_to_keep 1 --almond_has_multiple_programs --train_batch_tokens 5000',
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
        dlg_side=dlg_side,
        do_ner=do_ner,
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
