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

import kfp
import os
import argparse

from utils import list_pipelines, list_pipeline_versions, list_experiments, prepare_unknown_args
from pipelines.common import default_image


parser = argparse.ArgumentParser()

parser.add_argument('--owner', type=str)
parser.add_argument('--project', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--image', type=str, default=default_image)
parser.add_argument('--genienlp_version', type=str)
parser.add_argument('--bootleg_version', type=str)
parser.add_argument('--genie_version', type=str)
parser.add_argument('--thingtalk_version', type=str)
parser.add_argument('--workdir_repo', type=str)
parser.add_argument('--workdir_version', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--s3_datadir', type=str)

parser.add_argument('--kf_pipeline_name', type=str)
parser.add_argument('--kf_pipeline_version', type=str)
parser.add_argument('--kf_experiment_name', type=str)
parser.add_argument('--kf_job_name', type=str)

args, unknown_args = parser.parse_known_args()

extra_param_args = {}
if len(unknown_args):
    extra_param_args = prepare_unknown_args(unknown_args)

kfp_dir = os.path.expanduser('~/.config/kfp/')
if not os.path.isdir(kfp_dir):
    os.makedirs(kfp_dir)
client = kfp.Client()
client.set_user_namespace('research-kf')

pipelines = list_pipelines(client)
our_pipeline = None

for p in pipelines:
    if p.name == args.kf_pipeline_name:
        our_pipeline = p

if our_pipeline is None:
    raise ValueError('No pipelines with this name were found')

pipeline_versions = list_pipeline_versions(client, our_pipeline.id)

our_pipeline_version = None

if not args.kf_pipeline_version:
    # choose latest
    our_pipeline_version = pipeline_versions[-1]

for v in pipeline_versions:
    if v.name == args.kf_pipeline_version:
        our_pipeline_version = v

if our_pipeline_version is None:
    raise ValueError('No pipelines with this version were found')


experiments = list_experiments(client)

our_experiment = None
for e in experiments:
    if e.name == args.kf_experiment_name:
        our_experiment = e

if our_experiment is None:
    raise ValueError('No experiments with this name were found')


params = {}
args_vars = vars(args)
for k, v in args_vars.items():
    # drop kf specific args, drop empty args
    if k.startswith('kf_') or v is None:
        continue
    params[k] = args_vars[k]


params.update(extra_param_args)


client.run_pipeline(experiment_id=our_experiment.id,
                    job_name=args.kf_job_name,
                    params=params,
                    pipeline_id=our_pipeline.id,
                    version_id=our_pipeline_version.id)
