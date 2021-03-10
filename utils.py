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

from datetime import datetime
import os

import kfp


def list_experiments(client):
    resp = client.list_experiments(page_size=100)
    experiments = resp.experiments
    while resp.next_page_token:
        resp = client.list_experiments(page_token=resp.next_page_token, page_size=100)
        experiments.extend(resp.pipelines)
    return experiments


def list_pipelines(client):
    resp = client.list_pipelines(page_size=100)
    pipelines = resp.pipelines
    while resp.next_page_token:
        resp = client.list_pipelines(page_token=resp.next_page_token, page_size=100)
        pipelines.extend(resp.pipelines)
    return pipelines


def list_pipeline_versions(client, pipeline_id):
    resp = client.pipelines.list_pipeline_versions(resource_key_type="PIPELINE", resource_key_id=pipeline_id, page_size=100)
    pipeline_versions = resp.versions
    while resp.next_page_token:
        resp = client.pipelines.list_pipeline_versions(resource_key_type="PIPELINE", resource_key_id=pipeline_id, page_size=100, page_token=resp.next_page_token)
        pipeline_versions.extend(resp.versions)
    return pipeline_versions


def upload_pipeline(name, pipeline):
    """Upload pipeline to kubeflow"""
    compiled_pipeline_path = f'{name}.tar.gz'
    kfp.compiler.Compiler().compile(pipeline, compiled_pipeline_path)

    client = kfp.Client()
    pipelines = list_pipelines(client)
    pipelines = [] if pipelines is None else pipelines
    version = datetime.now().strftime("%m%d%Y-%H:%M:%S")

    # use upload_pipeline_version if an existing pipeline is found
    pid = None
    for p in pipelines:
        if p.name == name:
            pid = p.id
            break

    if pid:
        resp = client.pipeline_uploads.upload_pipeline_version(compiled_pipeline_path, name=version, pipelineid=pid)
    else:
        resp = client.pipeline_uploads.upload_pipeline(compiled_pipeline_path, name=name, description=pipeline._component_description)

    os.remove(compiled_pipeline_path)
    return resp


def prepare_unknown_args(args_list):

    assert len(args_list) % 2 == 0
    assert all(args_list[i].startswith(("-", "--")) for i in range(0, len(args_list), 2))
    
    cleaned_args = {}
    for i in range(0, len(args_list), 2):
        arg, value = args_list[i], args_list[i+1]
        cleaned_args[arg.strip('-')] = value

    return cleaned_args
