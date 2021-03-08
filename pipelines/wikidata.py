from kfp import dsl
from kfp import components
from kubernetes.client import V1Toleration
from kubernetes.client.models import (
    V1PersistentVolumeClaimVolumeSource,
)

from .common import *

from .paraphrase import paraphrase_generation_step, paraphrase_filtering_step

def generate_dataset_step(
    image,
    owner,
    project,
    experiment,
    canonical,
    dataset,
    genie_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    gen_dataset_env = {
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }
    generate_dataset_op = components.load_component_from_file('components/generate-dataset-wikidata.yaml')(
            image=image,
            s3_bucket='geniehai',
            owner=owner,
            project=project,
            experiment=experiment,
            canonical=canonical,
            dataset=dataset,
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
    canonical,
    model,
    load_from,
    s3_datadir,
    s3_bucket,
    genienlp_version,
    train_iterations,
    skip_tensorboard,
    genie_version,
    workdir_repo,
    workdir_version,
    additional_args
):
    train_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
    }
    train_num_gpus=1
    train_op = components.load_component_from_file('components/train-wikidata.yaml')(
            image=image,
            s3_bucket=s3_bucket,
            owner=owner,
            project=project,
            experiment=experiment,
            canonical=canonical,
            model=model,
            load_from=load_from,
            s3_datadir=s3_datadir,
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

    train_op.container.set_image_pull_policy('Always')
    
    return train_op

def eval_step(
    image,
    owner,
    project,
    experiment,
    canonical,
    model,
    s3_model_dir,
    #eval_set,
    genienlp_version,
    genie_version,
    workdir_repo,
    workdir_version,
    thingpedia_developer_key,
    additional_args
):
    eval_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
        'THINGPEDIA_DEVELOPER_KEY': thingpedia_developer_key,
    }

    eval_op = components.load_component_from_file('components/evaluate-wikidata.yaml')(
            image=image,
            owner=owner,
            project=project,
            experiment=experiment,
            canonical=canonical,
            model=model,
            model_owner=owner,
            #eval_set=eval_set,
            s3_model_dir=s3_model_dir,
            additional_args=additional_args)
    (eval_op.container
        .set_memory_limit('55Gi')
        .set_memory_request('55Gi')
        .set_cpu_limit('15.5')
        .set_cpu_request('15.5')
    )
    (add_env(add_ssh_volume(eval_op), eval_env)
        .add_toleration(V1Toleration(key='nvidia.com/gpu', operator='Exists', effect='NoSchedule'))
        .add_node_selector_constraint('beta.kubernetes.io/instance-type', 'g4dn.2xlarge'))

    return eval_op

def everything(
    do_generate,
    owner,
    project,
    experiment,
    canonical,
    model,
    dataset,
    train_iterations,
    train_additional_args,
    eval_set,
    eval_additional_args,
    genienlp_version,
    genie_version,
    workdir_repo,
    workdir_version,
    image,
    thingpedia_developer_key,
    s3_bucket='geniehai',
    generate_dataset_additional_args='',
    train_load_from='None',
    train_s3_datadir='',
    keep_original_duplicates='false',
    paraphrasing_model=PARAPHRASING_MODEL,
    paraphrase_subfolder='user',
    paraphrase_additional_args='',
    filtering_additional_args=''):

    if do_generate:
        generate_dataset_op = generate_dataset_step(image=image,
                                                    owner=owner,
                                                    project=project,
                                                    experiment=experiment,
                                                    canonical=canonical,
                                                    dataset=dataset,
                                                    genie_version=genie_version,
                                                    workdir_repo=workdir_repo,
                                                    workdir_version=workdir_version,
                                                    thingpedia_developer_key=thingpedia_developer_key,
                                                    additional_args=generate_dataset_additional_args)
        train_s3_datadir = generate_dataset_op.outputs['s3_datadir']
    #train_s3_datadir = 's3://geniehai/yamamura/dataset/wikidata294/1/auto/country/1613544170'

    train_op = train_step(
            image=image,
            owner=owner,
            project=project,
            experiment=experiment,
            canonical=canonical,
            model=model,
            s3_datadir=train_s3_datadir,
            s3_bucket=s3_bucket,
            genienlp_version=genienlp_version,
            load_from=train_load_from,
            skip_tensorboard='false',
            genie_version=genie_version,
            workdir_repo=workdir_repo,
            workdir_version=workdir_version,
            train_iterations=train_iterations,
            additional_args=train_additional_args,
            )
 
    s3_model_dir = train_op.outputs['s3_model_dir']

    eval_op = eval_step(image=image,
                        owner=owner,
                        project=project,
                        experiment=experiment,
                        canonical=canonical,
                        model=model,
                        s3_model_dir=s3_model_dir,
                        #eval_set=eval_set,
                        genienlp_version=genienlp_version,
                        genie_version=genie_version,
                        workdir_repo=workdir_repo,
                        workdir_version=workdir_version,
                        thingpedia_developer_key=thingpedia_developer_key,
                        additional_args=eval_additional_args)

@dsl.pipeline(
    name='Generate, train and eval',
    description='The Wikidata training pipeline'
)
def wikidata_pipeline(
    owner='yamamura',
    project='wikidata294',
    experiment='0',
    canonical='country', # added
    model='0',
    dataset='auto',
    image=default_image,
    genienlp_version=GENIENLP_VERSION,
    genie_version=GENIE_VERSION,
    workdir_repo=WORKDIR_REPO,
    workdir_version=WORKDIR_VERSION,
    thingpedia_developer_key=default_developer_key,
    generate_dataset_additional_args='',
    train_load_from='None',
    train_additional_args='',
    train_iterations='20000',
    eval_set='',
    eval_additional_args=''
):
    everything(do_generate=True,
               owner=owner,
               project=project,
               experiment=experiment,
               canonical=canonical,
               model=model,
               dataset=dataset,
               image=image,
               genienlp_version=genienlp_version,
               genie_version=genie_version,
               workdir_repo=workdir_repo,
               workdir_version=workdir_version,
               thingpedia_developer_key=thingpedia_developer_key,
               generate_dataset_additional_args=generate_dataset_additional_args,
               train_load_from=train_load_from,
               train_additional_args=train_additional_args,
               train_iterations=train_iterations,
               eval_set=eval_set,
               eval_additional_args=eval_additional_args)