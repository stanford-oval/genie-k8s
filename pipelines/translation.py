from kfp import components, dsl
from kubernetes.client import V1Toleration
from kubernetes.client.models import V1PersistentVolumeClaimVolumeSource

from .common import *


@dsl.pipeline(name='Translate a dialogue dataset', description='Prepare, Translate, and Postprocess a dialogue dataset')
def translate_dialogue_pipeline(
    owner='',
    project='',
    experiment='',
    s3_bucket='geniehai',
    s3_datadir='',
    source='',
    input_splits='train eval',
    model_name_or_path='',
    nmt='',
    src_lang='en',
    tgt_lang='',
    image=default_image,
    genienlp_version='',
    genie_version='',
    workdir_repo=GENIE_WORKDIR_REPO,
    workdir_version=GENIE_WORKDIR_VERSION,
    additional_args='',
):
    do_translation_env = {
        'GENIENLP_VERSION': genienlp_version,
        'GENIE_VERSION': genie_version,
        'WORKDIR_REPO': workdir_repo,
        'WORKDIR_VERSION': workdir_version,
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
