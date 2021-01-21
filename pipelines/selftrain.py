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

from .training import generate_dataset_step, paraphrase_fewshot_step, \
    train_step, eval_step


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
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    auto_annotate_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
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
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    s3_bucket='geniehai',
    s3_database_dir='',
    bootleg_model='',
    bootleg_version='',
    train_languages='',
    eval_languages='',
    s3_bootleg_prepped_data='',
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_additional_args='',
    train_iterations='80000',
    calibrate='false',
    calibration_additional_args='',
    fewshot_train_iterations='20000',
    selftrain_train_iterations='20000',
    filtering_train_iterations='10000',
    filtering_batch_size='4000',
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
        calibrate=calibrate,
        calibration_additional_args=calibration_additional_args,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        bootleg_version=bootleg_version,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        train_load_from='None',
        train_dataset_subfolder='None',
        filtering_train_iterations=filtering_train_iterations,
        filtering_batch_size=filtering_batch_size,
        fewshot_train_iterations=fewshot_train_iterations,
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
        calibrate=calibrate,
        calibration_additional_args=calibration_additional_args,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        bootleg_version=bootleg_version,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        train_load_from='None',
        train_dataset_subfolder='None',
        filtering_train_iterations=filtering_train_iterations,
        filtering_batch_size=filtering_batch_size,
        fewshot_train_iterations=fewshot_train_iterations,
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
                                          workdir_repo=workdir_repo,
                                          workdir_version=workdir_version,
                                          thingpedia_developer_key=thingpedia_developer_key,
                                          additional_args=auto_annotate_additional_args)
    selftrain_datadir = auto_annotate_op.outputs['s3_datadir']

    train_op = train_step(
            owner=owner,
            project=project,
            experiment=experiment,
            model='%s-selftrain' % (model,),
            task_name='almond_dialogue_nlu',
            s3_datadir=selftrain_datadir,
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            bootleg_model=bootleg_model,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_version=bootleg_version,
            load_from=user_model,
            train_languages=train_languages,
            eval_languages=eval_languages,
            dataset_subfolder='None',
            skip_tensorboard='false',
            calibrate=calibrate,
            calibration_additional_args=calibration_additional_args,
            train_iterations=selftrain_train_iterations,
            s3_bootleg_prepped_data=s3_bootleg_prepped_data,
            additional_args=train_additional_args
            )
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
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    s3_bucket='geniehai',
    s3_database_dir='',
    bootleg_model='',
    bootleg_version='',
    train_languages='',
    eval_languages='',
    s3_bootleg_prepped_data='',
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_additional_args='',
    train_iterations='80000',
    fewshot_train_iterations='20000',
    selftrain_train_iterations='20000',
    calibrate='false',
    calibration_additional_args='',
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
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        bootleg_version=bootleg_version,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        train_load_from='None',
        train_dataset_subfolder='None',
        calibrate=calibrate,
        calibration_additional_args=calibration_additional_args,
        filtering_train_iterations='',
        filtering_batch_size='',
        fewshot_train_iterations=fewshot_train_iterations,
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
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        bootleg_model=bootleg_model,
        bootleg_version=bootleg_version,
        train_languages=train_languages,
        eval_languages=eval_languages,
        eval_set=eval_set,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        train_load_from='None',
        train_dataset_subfolder='None',
        calibrate=calibrate,
        calibration_additional_args=calibration_additional_args,
        filtering_train_iterations='',
        filtering_batch_size='',
        fewshot_train_iterations=fewshot_train_iterations,
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
                                          workdir_repo=workdir_repo,
                                          workdir_version=workdir_version,
                                          thingpedia_developer_key=thingpedia_developer_key,
                                          additional_args=auto_annotate_additional_args)
    selftrain_datadir = auto_annotate_op.outputs['s3_datadir']

    train_op = train_step(
            owner=owner,
            project=project,
            experiment=experiment,
            model='%s-selftrain' % (model,),
            task_name='almond_dialogue_nlu',
            s3_datadir=selftrain_datadir,
            s3_bucket=s3_bucket,
            s3_database_dir=s3_database_dir,
            bootleg_model=bootleg_model,
            image=image,
            genienlp_version=genienlp_version,
            bootleg_version=bootleg_version,
            load_from=user_model,
            train_languages=train_languages,
            eval_languages=eval_languages,
            dataset_subfolder='None',
            skip_tensorboard='false',
            calibrate=calibrate,
            calibration_additional_args=calibration_additional_args,
            train_iterations=selftrain_train_iterations,
            s3_bootleg_prepped_data=s3_bootleg_prepped_data,
            additional_args=train_additional_args,
    )
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
                        workdir_repo=workdir_repo,
                        workdir_version=workdir_version,
                        thingpedia_developer_key=thingpedia_developer_key,
                        additional_args=eval_additional_args)
