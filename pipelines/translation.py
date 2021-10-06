from kfp import components, dsl
from kubernetes.client import V1Toleration
from kubernetes.client.models import V1PersistentVolumeClaimVolumeSource

from . import training
from .common import *


@dsl.pipeline(name='Full multilingual pipeline for dialogue datasets', description='Generate, Translate, Train, and Eval')
def generate_translate_train_eval_pipeline(
    owner='mehrad',
    project='mario',
    experiment='custom',
    s3_bucket='geniehai',
    source='user',
    input_splits='train eval',
    model_name_or_path='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt='marian',
    src_lang='en',
    tgt_lang='it',
    model='test-2',
    dataset='test-2',
    image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit-kf:20210923.1',
    genienlp_version='1a0243bc6464a6a4b7149eef66f68f025b7a6c46',
    genie_version='c3e124f7a38bf1419c2245cc0c5895a02288c787',
    workdir_repo=WORKDIR_REPO,
    workdir_version='6800b43ee15048f0f2eea9e82019662de03790a7',
    thingpedia_developer_key='88c03add145ad3a3aa4074ffa828be5a391625f9d4e1d0b034b445f18c595656',
    translation_additional_args='val_batch_size=3000 temperature=0.2 custom_devices=main/org.thingpedia.weather',
    generate_dataset_parallel='6',
    generate_dataset_additional_args='custom_devices=main/org.thingpedia.weather subdatasets=1 target_pruning_size=20 max_turns=3 max_depth=3 simple_subdatasets=1 simple_target_pruning_size=20 simple_max_turns=3 simple_max_depth=3',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='80000',
    valid_set='eval',
    eval_set='dev',
    eval_parallel_jobs='2',
    eval_additional_args='custom_devices=main/org.thingpedia.weather',
    fewshot_train_iterations='20000',
):
    training.everything(
        do_generate=True,
        do_translate=True,
        do_bootleg=False,
        do_paraphrase=False,
        do_fewshot=False,
        do_calibrate=False,
        do_ood=False,
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
        s3_bucket=s3_bucket,
        generate_dataset_parallel=generate_dataset_parallel,
        generate_dataset_additional_args=generate_dataset_additional_args,
        train_task_name=train_task_name,
        train_load_from=train_load_from,
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        valid_set=valid_set,
        eval_set=eval_set,
        eval_parallel_jobs=eval_parallel_jobs,
        eval_additional_args=eval_additional_args,
        source=source,
        translation_input_splits=input_splits,
        translation_model_name_or_path=model_name_or_path,
        nmt=nmt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        translation_additional_args=translation_additional_args,
        fewshot_train_iterations=fewshot_train_iterations,
    )


@dsl.pipeline(name='Translate, train, and eval pipeline for dialogue datasets', description='Translate, Train, and Eval')
def translate_train_eval_pipeline(
    owner='mehrad',
    project='mario',
    experiment='custom',
    s3_bucket='geniehai',
    source='user',
    input_splits='train eval',
    model_name_or_path='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt='marian',
    src_lang='en',
    tgt_lang='it',
    model='test-1',
    s3_datadir='s3://geniehai/mehrad/dataset/mario/custom/test-1/1633479535/',
    image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit-kf:20210923.1',
    genienlp_version='1a0243bc6464a6a4b7149eef66f68f025b7a6c46',
    genie_version='c3e124f7a38bf1419c2245cc0c5895a02288c787',
    workdir_repo=WORKDIR_REPO,
    workdir_version='6800b43ee15048f0f2eea9e82019662de03790a7',
    thingpedia_developer_key='88c03add145ad3a3aa4074ffa828be5a391625f9d4e1d0b034b445f18c595656',
    translation_additional_args='val_batch_size=3000 temperature=0.2 custom_devices=main/org.thingpedia.weather',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='--pretrained_model facebook/mbart-large-50 --train_batch_tokens 700 --val_batch_size 2000 --preprocess_special_tokens --train_languages it --train_tgt_languages it --eval_languages it --eval_tgt_languages it',
    train_iterations='1000',
    valid_set='eval',
    eval_set='dev',
    eval_parallel_jobs='2',
    eval_additional_args='custom_devices=main/org.thingpedia.weather',
    fewshot_train_iterations='20000',
):
    training.everything(
        do_generate=False,
        do_translate=True,
        do_bootleg=False,
        do_paraphrase=False,
        do_fewshot=False,
        do_calibrate=False,
        do_ood=False,
        owner=owner,
        project=project,
        experiment=experiment,
        model=model,
        image=image,
        train_s3_datadir=s3_datadir,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        thingpedia_developer_key=thingpedia_developer_key,
        s3_bucket=s3_bucket,
        train_task_name=train_task_name,
        train_load_from=train_load_from,
        train_additional_args=train_additional_args,
        train_iterations=train_iterations,
        valid_set=valid_set,
        eval_set=eval_set,
        eval_parallel_jobs=eval_parallel_jobs,
        eval_additional_args=eval_additional_args,
        source=source,
        translation_input_splits=input_splits,
        translation_model_name_or_path=model_name_or_path,
        nmt=nmt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        translation_additional_args=translation_additional_args,
        fewshot_train_iterations=fewshot_train_iterations,
    )


@dsl.pipeline(name='Translate a dialogue dataset', description='Prepare, Translate, and Postprocess a dialogue dataset')
def translate_dialogue_pipeline(
    owner='mehrad',
    project='wwvw',
    experiment='everything',
    s3_bucket='geniehai',
    s3_datadir='s3://geniehai/mehrad/dataset/wwvw/everything/1/',
    source='user',
    input_splits='train eval',
    model_name_or_path='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt='marian',
    src_lang='en',
    tgt_lang='it',
    image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit-kf:20210923.1',
    genienlp_version='1a0243bc6464a6a4b7149eef66f68f025b7a6c46',
    genie_version='c3e124f7a38bf1419c2245cc0c5895a02288c787',
    workdir_repo=GENIE_WORKDIR_REPO,
    workdir_version='b8e728e01a866b73c2c8d9bd809f78bb7af3c67b',
    thingpedia_developer_key=default_developer_key,
    additional_args='--val_batch_size 2000 --temperature 0.2 --translate_example_split',
):
    translation_op = dialogue_translation_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        source=source,
        input_splits=input_splits,
        model_name_or_path=model_name_or_path,
        nmt=nmt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        thingpedia_developer_key=thingpedia_developer_key,
        additional_args=additional_args,
    )


def dialogue_translation_step(
    owner='mehrad',
    project='wwvw',
    experiment='everything',
    s3_bucket='geniehai',
    s3_datadir='s3://geniehai/mehrad/dataset/wwvw/everything/1/',
    source='user',
    input_splits='train eval',
    model_name_or_path='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt='marian',
    src_lang='en',
    tgt_lang='it',
    image='932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit-kf:20210923.1',
    genienlp_version='fafb846161fa9d34ef92aa67e98b4e26a0c7118d',
    genie_version='c3e124f7a38bf1419c2245cc0c5895a02288c787',
    workdir_repo=GENIE_WORKDIR_REPO,
    workdir_version='b8e728e01a866b73c2c8d9bd809f78bb7af3c67b',
    thingpedia_developer_key=default_developer_key,
    additional_args='--val_batch_size 2000 --temperature 0.2 --translate_example_split',
):
    do_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }

    do_translation_op = components.load_component_from_file('components/translate-dialogues.yaml')(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        input_splits=input_splits,
        source=source,
        model_name_or_path=model_name_or_path,
        nmt=nmt,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prepare_for_translation='true',
        do_translation='true',
        post_process_translation='true',
        additional_args=additional_args,
    )
    (
        do_translation_op.container.set_memory_request('31G')
        .set_memory_limit('31G')
        .set_cpu_request('7.5')
        .set_cpu_limit('7.5')
        .add_volume_mount(V1VolumeMount(name='tensorboard', mount_path='/shared/tensorboard'))
    )
    (
        add_env(add_ssh_volume(do_translation_op), do_translation_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.2xlarge')
        .add_volume(
            V1Volume(
                name='tensorboard', persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('tensorboard-research-kf')
            )
        )
    )

    # do_translation_op.human_name = 'translation'

    return do_translation_op
