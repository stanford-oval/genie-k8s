@dsl.pipeline(
    name='Train and prediction pipeline',
    description='Train a model and do prediction with it'
)
def train_predict_small_pipeline(
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
        eval_sets='eval',
        dataset_subfolder='None',
        skip_tensorboard='false',
        train_iterations='',
        s3_bootleg_prepped_data='None',
        train_additional_args='',
        val_batch_size='1000',
        pred_additional_args='--evaluate valid --overwrite'
):
    train_op = train_step(
        image=image,
        owner=owner,
        project=project,
        experiment=experiment,
        genienlp_version=genienlp_version,
        model=model,
        task_name=task_name,
        s3_datadir=s3_datadir,
        s3_bucket=s3_bucket,
        s3_database_dir=s3_database_dir,
        load_from=load_from,
        dataset_subfolder=dataset_subfolder,
        skip_tensorboard=skip_tensorboard,
        train_iterations=train_iterations,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=train_additional_args
    )
    
    pred_op = prediction_step_small(
        image=image,
        owner=owner,
        genienlp_version=genienlp_version,
        task_name=task_name,
        eval_sets=eval_sets,
        model_name_or_path=train_op.outputs['s3_model_dir'],
        s3_input_datadir=s3_datadir,
        s3_database_dir=s3_database_dir,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        s3_bootleg_prepped_data=s3_bootleg_prepped_data,
        additional_args=pred_additional_args,
    )


@dsl.pipeline(
    name='Predict',
    description='Run genienlp predict on a previously trained model'
)
def predict_pipeline(
        image=default_image,
        genienlp_version=GENIENLP_VERSION,
        owner='',
        eval_sets='',
        task_name='',
        model_name_or_path='',
        s3_input_datadir='',
        model_type='None',
        dataset_subfolder='None',
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
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=additional_args,
        genienlp_version=genienlp_version
    )


@dsl.pipeline(
    name='Predict using g4dn.4xlarge',
    description='Run genienlp predict on a previously trained model'
)
def predict_small_pipeline(
        image=default_image,
        genienlp_version=GENIENLP_VERSION,
        owner='',
        task_name='',
        model_name_or_path='',
        s3_input_datadir='',
        model_type='None',
        dataset_subfolder='None',
        eval_sets='eval test',
        val_batch_size='4000',
        additional_args='',
):
    prediction_step_small(
        image=image,
        owner=owner,
        eval_sets=eval_sets,
        task_name=task_name,
        model_name_or_path=model_name_or_path,
        s3_input_datadir=s3_input_datadir,
        model_type=model_type,
        dataset_subfolder=dataset_subfolder,
        val_batch_size=val_batch_size,
        additional_args=additional_args,
        genienlp_version=genienlp_version,
    )


