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

import os

from kubernetes.client import V1PersistentVolumeClaimVolumeSource
from kubernetes.client.models import V1EnvVar, V1SecretVolumeSource, V1Volume, V1VolumeMount

# Get the Thingpedia key from environment variable
default_developer_key = os.getenv('THINGPEDIA_DEVELOPER_KEY')

# For AWS access on Azure machines
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

AZURE_SP_APP_ID = os.getenv('AZURE_SP_APP_ID')
AZURE_SP_TENANT_ID = os.getenv('AZURE_SP_TENANT_ID')
AZURE_SP_PASSWORD = os.getenv('AZURE_SP_PASSWORD')

default_image = 'stanfordoval.azurecr.io/genie/kubeflow:20220725'
GENIENLP_VERSION = '554ec076107774febcc40c110e176734c311a160'
GENIE_VERSION = '681b21d161e35a65f2dec29fde6967c012fa024b'
WORKDIR_REPO = 'git@github.com:stanford-oval/thingpedia-common-devices.git'
WORKDIR_VERSION = '35592e1f22f9318d1f26ca79f0ab86e50e55ae87'
GENIE_WORKDIR_REPO = 'git@github.com:stanford-oval/genie-workdirs.git'
GENIE_WORKDIR_VERSION = 'master'
PARAPHRASING_MODEL = '/shared/data/sinaj/models/schemaorg/paraphrase/bart-large-speedup-megabatch-5m/'
S3_DATABASE_DIR = 'https://nfs009a5d03c43b4e7e8ec2.blob.core.windows.net/pvc-a8853620-9ac7-4885-a30e-0ec357f17bb6/mehrad/extras/bootleg_files_v1.0.0/'

# name of a secret in Kubernetes containing the SSH credentials (GitHub deploy key for genie-workdirs)
SSH_VOLUME = 'ssh-secrets-7fdcbg96c4'
DATA_VOLUME = 'shared-data'


def add_ssh_volume(op):
    op.add_volume(V1Volume(name='ssh-v', secret=V1SecretVolumeSource(secret_name=SSH_VOLUME, default_mode=0o600)))
    op.container.add_volume_mount(V1VolumeMount(name='ssh-v', mount_path='/root/.ssh'))

    op.add_volume(V1Volume(name='shared-data', persistent_volume_claim=V1PersistentVolumeClaimVolumeSource('shared-data')))
    op.container.add_volume_mount(V1VolumeMount(name='shared-data', mount_path='/shared/data'))

    return op


def disable_caching(op):
    """Disable caching by setting the staleness to 0.
    By default kubeflow will cache operation if the inputs are the same.
    even if the underlying datafiles have changed.
    """
    op.execution_options.caching_strategy.max_cache_staleness = 'P0D'
    return op


def add_env(op, envs):
    """Add a dict of environments to container"""
    for k, v in envs.items():
        op.container.add_env_variable(V1EnvVar(name=k, value=v))

    op.container.add_env_variable(V1EnvVar(name='AWS_ACCESS_KEY_ID', value=AWS_ACCESS_KEY_ID))
    op.container.add_env_variable(V1EnvVar(name='AWS_SECRET_ACCESS_KEY', value=AWS_SECRET_ACCESS_KEY))

    op.container.add_env_variable(V1EnvVar(name='AZURE_SP_APP_ID', value=AZURE_SP_APP_ID))
    op.container.add_env_variable(V1EnvVar(name='AZURE_SP_TENANT_ID', value=AZURE_SP_TENANT_ID))
    op.container.add_env_variable(V1EnvVar(name='AZURE_SP_PASSWORD', value=AZURE_SP_PASSWORD))

    return op
