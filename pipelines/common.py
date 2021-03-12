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

from kubernetes.client.models import (
    V1VolumeMount,
    V1Volume,
    V1EnvVar,
    V1SecretVolumeSource
)

# Get the Thingpedia key from environment variable
default_developer_key = os.getenv('THINGPEDIA_DEVELOPER_KEY')

default_image = '932360549041.dkr.ecr.us-west-2.amazonaws.com/genie-toolkit-kf:20210212.2'
GENIENLP_VERSION = 'a24f5106bb63642d3dc1fbea039ffe49c829e7be'
GENIE_VERSION = 'c96402da098696d71bc04a9c36a5753d1beb8982'
WORKDIR_REPO = 'git@github.com:stanford-oval/thingpedia-common-devices.git'
WORKDIR_VERSION = '1efc491d27c756967890261c6d045fff0755450b'
GENIE_WORKDIR_REPO = 'git@github.com:stanford-oval/genie-workdirs.git'
GENIE_WORKDIR_VERSION = 'master'
PARAPHRASING_MODEL = 's3://geniehai/sinaj/models/schemaorg/paraphrase/bart-large-speedup-megabatch-5m/'
S3_DATABASE_DIR = 's3://geniehai/mehrad/extras/bootleg_files'

# name of a secret in Kubernetes containing the SSH credentials (GitHub deploy key)
SSH_VOLUME = 'ssh-secrets-k425k8d8h8'


def add_ssh_volume(op):
    op.add_volume(V1Volume(name='ssh-v',
                           secret=V1SecretVolumeSource(secret_name=SSH_VOLUME, default_mode=0o600)))
    op.container.add_volume_mount(V1VolumeMount(name='ssh-v', mount_path='/root/.ssh'))
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
    return op
