from kfp import components, dsl
from kubernetes.client import V1EmptyDirVolumeSource, V1Toleration

from . import training
from .common import *


@dsl.pipeline(name='Full multilingual pipeline for dialogue datasets', description='Generate, Translate, Train, and Eval')
def generate_translate_train_eval_pipeline(
    owner='',
    project='',
    experiment='',
    s3_bucket='geniehai',
    source='user',
    input_splits='train eval',
    translation_model='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt_id='nmt',
    src_lang='en',
    tgt_lang='',
    model='',
    dataset='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    translation_additional_args='',
    generate_dataset_parallel='6',
    generate_dataset_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='20000',
    valid_set='eval',
    eval_set='dev',
    eval_parallel_jobs='2',
    eval_additional_args='',
    fewshot_train_iterations='10000',
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
        translation_model=translation_model,
        nmt_id=nmt_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        translation_additional_args=translation_additional_args,
        fewshot_train_iterations=fewshot_train_iterations,
    )


@dsl.pipeline(name='Translate, train, and eval pipeline for dialogue datasets', description='Translate, Train, and Eval')
def translate_train_eval_pipeline(
    owner='',
    project='',
    experiment='',
    s3_bucket='geniehai',
    source='user',
    input_splits='train eval',
    translation_model='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt_id='nmt',
    src_lang='en',
    tgt_lang='',
    model='',
    s3_datadir='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    translation_additional_args='',
    train_task_name='almond_dialogue_nlu',
    train_load_from='None',
    train_additional_args='',
    train_iterations='20000',
    valid_set='eval',
    eval_set='dev',
    eval_parallel_jobs='2',
    eval_additional_args='',
    fewshot_train_iterations='10000',
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
        translation_model=translation_model,
        nmt_id=nmt_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        translation_additional_args=translation_additional_args,
        fewshot_train_iterations=fewshot_train_iterations,
    )


@dsl.pipeline(name='Translate a dialogue dataset', description='Prepare, Translate, and Postprocess a dialogue dataset')
def translate_dialogue_pipeline(
    owner='mehrad',
    project='zeroshot',
    experiment='bitod_en',
    s3_bucket='geniehai',
    s3_datadir='s3://geniehai/mehrad/dataset/zeroshot/bitod_e2e/en_v14/',
    source='en_v10',
    input_splits='valid',
    translation_model='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt_id='marian',
    src_lang='en',
    tgt_lang='zh',
    image=default_image,
    genienlp_version='a60e259c7efc0aa5cd615c314fdbb96cd5c105a7',
    genie_version=GENIE_VERSION,
    workdir_repo='git@github.com:stanford-oval/genie-workdirs.git',
    workdir_version='master',
    thingpedia_developer_key='ad4e938f1b1e0ee5bbb0615315f6222436071d976262c79491213ae544b415f6',
    prepare_for_translation='true',
    do_translation='true',
    post_process_translation='false',
    additional_args='no_random=true skip_ent_translation=true do_align_help=false translate_extra_args="--val_batch_size=2000 --temperature=0.3"',
):
    translation_op = dialogue_translation_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        source=source,
        input_splits=input_splits,
        translation_model=translation_model,
        nmt_id=nmt_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        thingpedia_developer_key=thingpedia_developer_key,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        additional_args=additional_args,
    )


def dialogue_translation_step(
    owner,
    project,
    experiment,
    s3_bucket='geniehai',
    s3_datadir='',
    source='user',
    input_splits='train eval',
    translation_model='',
    nmt_id='',
    src_lang='en',
    tgt_lang='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    thingpedia_developer_key=default_developer_key,
    workdir_repo=GENIE_WORKDIR_REPO,
    workdir_version=GENIE_WORKDIR_VERSION,
    prepare_for_translation='true',
    do_translation='true',
    post_process_translation='true',
    additional_args='',
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
        translation_model=translation_model,
        nmt_id=nmt_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        additional_args=additional_args,
    )
    (
        do_translation_op.container.set_memory_request('110G')
        .set_memory_limit('110G')
        .set_cpu_request('28')
        .set_cpu_limit('28')
        .add_volume_mount(V1VolumeMount(name='shm', mount_path='/dev/shm'))
    )
    (
        add_env(add_ssh_volume(do_translation_op), do_translation_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.8xlarge')
        .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
    )

    # do_translation_op.human_name = 'translation'

    return do_translation_op


def dialogue_translation_4gpu_step(
    owner,
    project,
    experiment,
    s3_bucket='geniehai',
    s3_datadir='',
    source='user',
    input_splits='train eval',
    translation_model='',
    nmt_id='',
    src_lang='en',
    tgt_lang='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    thingpedia_developer_key=default_developer_key,
    workdir_repo=GENIE_WORKDIR_REPO,
    workdir_version=GENIE_WORKDIR_VERSION,
    prepare_for_translation='true',
    do_translation='true',
    post_process_translation='true',
    additional_args='',
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
        translation_model=translation_model,
        nmt_id=nmt_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        additional_args=additional_args,
    )
    (
        do_translation_op.container.set_memory_request('190G')
        .set_memory_limit('190G')
        .set_cpu_request('40')
        .set_cpu_limit('40')
        # not supported yet in the version of kfp we're using
        # .set_ephemeral_storage_request('75G')
        # .set_ephemeral_storage_limit('75G')
        .set_gpu_limit('4')
    )
    (
        add_env(add_ssh_volume(do_translation_op), do_translation_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.12xlarge')
    )

    # do_translation_op.human_name = 'translation'

    return do_translation_op


@dsl.pipeline(
    name='Translate a dialogue dataset on 4 gpus', description='Prepare, Translate, and Postprocess a dialogue dataset'
)
def translate_dialogue_4gpu_pipeline(
    owner='',
    project='',
    experiment='',
    s3_bucket='geniehai',
    s3_datadir='',
    source='',
    input_splits='train eval',
    translation_model='Helsinki-NLP/opus-mt-$(src_lang)-$(tgt_lang)',
    nmt_id='marian',
    src_lang='en',
    tgt_lang='',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    prepare_for_translation='true',
    do_translation='true',
    post_process_translation='true',
    additional_args='',
):
    translation_op = dialogue_translation_4gpu_step(
        owner=owner,
        project=project,
        experiment=experiment,
        s3_bucket=s3_bucket,
        s3_datadir=s3_datadir,
        source=source,
        input_splits=input_splits,
        translation_model=translation_model,
        nmt_id=nmt_id,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        image=image,
        genienlp_version=genienlp_version,
        genie_version=genie_version,
        workdir_repo=workdir_repo,
        workdir_version=workdir_version,
        thingpedia_developer_key=thingpedia_developer_key,
        prepare_for_translation=prepare_for_translation,
        do_translation=do_translation,
        post_process_translation=post_process_translation,
        additional_args=additional_args,
    )
