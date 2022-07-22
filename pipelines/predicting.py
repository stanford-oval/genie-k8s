from kfp import components, dsl
from kubernetes.client import V1EmptyDirVolumeSource, V1Toleration

from . import split_bootleg_merge_step, training
from .common import *


def prediction_4gpu_step(
    image,
    owner,
    genienlp_version,
    task_name,
    eval_sets,
    model_name_or_path,
    s3_input_datadir,
    s3_database_dir,
    s3_bootleg_prepped_data,
    model_type,
    dataset_subfolder,
    val_batch_size,
    additional_args,
):

    predict_env = {
        'GENIENLP_VERSION': genienlp_version,
    }

    predict_num_gpus = 4
    predict_op = components.load_component_from_file('components/predict.yaml')(
        image=image,
        owner=owner,
        eval_sets=eval_sets,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_input_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=additional_args,
    )
    (
        predict_op.container.set_memory_request('400G')
        .set_memory_limit('400G')
        .set_cpu_request('60')
        .set_cpu_limit('60')
        # not supported yet in the version of kfp we're using
        # .set_ephemeral_storage_request('75G')
        # .set_ephemeral_storage_limit('75G')
        .set_gpu_limit(str(predict_num_gpus))
        .add_volume_mount(V1VolumeMount(name='shm', mount_path='/dev/shm'))
    )
    (
        add_env(add_ssh_volume(predict_op), predict_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'Standard_NC64as_T4_v3')
        .add_volume(V1Volume(name='shm', empty_dir=V1EmptyDirVolumeSource(medium='Memory')))
    )

    return predict_op


def prediction_step(
    image,
    owner,
    genienlp_version,
    task_name,
    eval_sets,
    model_name_or_path,
    s3_input_datadir,
    s3_database_dir,
    s3_bootleg_prepped_data,
    model_type,
    dataset_subfolder,
    val_batch_size,
    additional_args,
):

    predict_env = {'GENIENLP_VERSION': genienlp_version}

    predict_op = components.load_component_from_file('components/predict.yaml')(
        image=image,
        owner=owner,
        eval_sets=eval_sets,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_input_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=additional_args,
    )
    (predict_op.container.set_memory_limit('100G').set_memory_request('100G').set_cpu_limit('5').set_cpu_request('5'))
    (
        add_env(add_ssh_volume(predict_op), predict_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'Standard_NC6s_v3')
    )

    return predict_op


def prediction_step_e2e(
    image,
    owner,
    genienlp_version,
    task_name,
    eval_sets,
    eval_lang,
    model_name_or_path,
    s3_input_datadir,
    model_type,
    dataset_subfolder,
    additional_args,
):

    predict_env = {'GENIENLP_VERSION': genienlp_version}

    predict_op = components.load_component_from_file('components/predict-e2e.yaml')(
        image=image,
        owner=owner,
        eval_sets=eval_sets,
        eval_lang=eval_lang,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_input_datadir,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        additional_args=additional_args,
    )
    (predict_op.container.set_memory_limit('31G').set_memory_request('31G').set_cpu_limit('7.5').set_cpu_request('7.5'))
    (
        add_env(add_ssh_volume(predict_op), predict_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'Standard_NC8as_T4_v3')
    )

    return predict_op


@dsl.pipeline(
    name='Bootleg, train, and prediction pipeline', description='Bootleg the dataset, train a model and do prediction'
)
def bootleg_train_predict_pipeline(
    owner,
    project,
    experiment,
    model,
    task_name,
    s3_datadir,
    s3_bucket='geniehai',
    s3_database_dir=S3_DATABASE_DIR,
    s3_bootleg_subfolder='None',
    model_type='None',
    image=default_image,
    genienlp_version='',
    load_from='None',
    valid_set='eval',
    eval_sets='eval',
    dataset_subfolder='None',
    train_languages='en',
    eval_languages='en',
    file_extension='tsv',
    skip_tensorboard='false',
    train_iterations='',
    train_additional_args='',
    bootleg_model='',
    bootleg_data_splits='train eval',
    bootleg_additional_args='',
    val_batch_size='4000',
    pred_additional_args='',
):

    s3_bootleg_prepped_data = split_bootleg_merge_step(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        s3_bootleg_subfolder=s3_bootleg_subfolder,
        genienlp_version=genienlp_version,
        bootleg_model=bootleg_model,
        train_languages=train_languages,
        eval_languages=eval_languages,
        data_splits=bootleg_data_splits,
        file_extension=file_extension,
        bootleg_additional_args=bootleg_additional_args,
    )

    train_op = training.train_step(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        genienlp_version=genienlp_version,
        model=model,
        task_name=task_name,
        valid_set=valid_set,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        load_from=load_from,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args,
    )

    pred_op = prediction_step(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_sets,
        model_name_or_path=train_op.outputs['s3_model_dir'],
        s3_input_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=pred_additional_args,
    )


@dsl.pipeline(name='Train and prediction pipeline', description='Train a model and do prediction')
def train_predict_pipeline(
    owner,
    project,
    experiment,
    model,
    task_name,
    s3_datadir,
    s3_bucket='geniehai',
    s3_database_dir='None',
    model_type='',
    image=default_image,
    genienlp_version='',
    load_from='None',
    valid_set='eval',
    eval_sets='eval',
    dataset_subfolder='None',
    skip_tensorboard='false',
    train_iterations='',
    s3_bootleg_prepped_data='None',
    train_additional_args='',
    val_batch_size='4000',
    pred_additional_args='',
):
    train_op = training.train_step(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        genienlp_version=genienlp_version,
        model=model,
        task_name=task_name,
        valid_set=valid_set,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        load_from=load_from,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args,
    )

    pred_op = prediction_step(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_sets,
        model_name_or_path=train_op.outputs['s3_model_dir'],
        s3_input_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=pred_additional_args,
    )


@dsl.pipeline(
    name='Train and prediction pipeline for e2e dialogue', description='Train a model and do prediction on e2e dialogue'
)
def train_predict_e2e_dialogue_pipeline(
    owner,
    project,
    experiment,
    model,
    s3_datadir,
    task_name='',
    s3_bucket='geniehai',
    model_type='None',
    image=default_image,
    genienlp_version='',
    load_from='None',
    valid_set='valid',
    eval_sets='valid test',
    eval_e2e_sets='test',
    eval_lang='',
    dataset_subfolder='None',
    skip_tensorboard='false',
    train_iterations='',
    train_additional_args='',
    val_batch_size='4000',
    pred_additional_args='',
):
    train_op = training.train_step(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        genienlp_version=genienlp_version,
        model=model,
        task_name=task_name,
        valid_set=valid_set,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        load_from=load_from,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        additional_args=train_additional_args,
    )

    pred_op = prediction_step(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_sets,
        model_name_or_path=train_op.outputs['s3_model_dir'],
        s3_input_datadir=s3_datadir,
        s3_database_dir='None',
        s3_bootleg_prepped_data='None',
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=pred_additional_args,
    )

    pred_e2e_op = prediction_step_e2e(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_e2e_sets,
        eval_lang=eval_lang,
        model_name_or_path=train_op.outputs['s3_model_dir'],
        s3_input_datadir=s3_datadir,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        additional_args=pred_additional_args,
    )


@dsl.pipeline(name='prediction pipeline for e2e dialogue', description='do prediction on e2e dialogue')
def predict_e2e_dialogue_pipeline(
    owner='',
    model_name_or_path='',
    s3_datadir='',
    task_name='',
    model_type='None',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    eval_lang='en',
    eval_sets='',
    eval_e2e_sets='',
    dataset_subfolder='None',
    val_batch_size='4000',
    pred_additional_args='--extra_metrics e2e_dialogue_score',
):

    pred_op = prediction_step(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_sets,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_datadir,
        s3_database_dir='None',
        s3_bootleg_prepped_data='None',
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=pred_additional_args,
    )

    pred_e2e_op = prediction_step_e2e(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_e2e_sets,
        eval_lang=eval_lang,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_datadir,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        additional_args=pred_additional_args,
    )


@dsl.pipeline(name='Train and prediction pipeline', description='Train a model on 4 gpus and do prediction')
def train_4gpu_predict_pipeline(
    owner,
    project,
    experiment,
    model,
    task_name,
    s3_datadir,
    s3_bucket='geniehai',
    s3_database_dir='None',
    model_type='',
    image=default_image,
    genienlp_version='',
    load_from='None',
    valid_set='eval',
    eval_sets='eval',
    dataset_subfolder='None',
    skip_tensorboard='false',
    train_iterations='',
    s3_bootleg_prepped_data='None',
    train_additional_args='',
    val_batch_size='4000',
    pred_additional_args='',
):
    train_op = training.train_step(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        genienlp_version=genienlp_version,
        model=model,
        task_name=task_name,
        valid_set=valid_set,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        load_from=load_from,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        num_gpus='4',
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args,
    )

    pred_op = prediction_step(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_sets,
        model_name_or_path=train_op.outputs['s3_model_dir'],
        s3_input_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=pred_additional_args,
    )


@dsl.pipeline(name='Predict', description='Run genienlp predict on a previously trained model')
def predict_4gpu_pipeline(
    image=default_image,
    owner='',
    eval_sets='',
    task_name='',
    genienlp_version=GENIENLP_VERSION,
    model_name_or_path='',
    s3_input_datadir='',
    s3_database_dir='None',
    s3_bootleg_prepped_data='None',
    model_type='',
    dataset_subfolder='None',
    val_batch_size='4000',
    additional_args='',
):
    prediction_4gpu_step(
        image=image,
        owner=owner,
        eval_sets=eval_sets,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_input_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=additional_args,
        genienlp_version=genienlp_version,
    )


@dsl.pipeline(name='Predict using Standard_NC16as_T4_v3', description='Run genienlp predict on a previously trained model')
def predict_pipeline(
    image=default_image,
    owner='',
    task_name='',
    genienlp_version=GENIENLP_VERSION,
    model_name_or_path='',
    s3_input_datadir='',
    s3_database_dir='None',
    s3_bootleg_prepped_data='None',
    model_type='None',
    dataset_subfolder='None',
    eval_sets='eval test',
    val_batch_size='4000',
    additional_args='',
):
    prediction_step(
        image=image,
        owner=owner,
        eval_sets=eval_sets,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_input_datadir,
        s3_database_dir=s3_database_dir,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=additional_args,
        genienlp_version=genienlp_version,
    )
