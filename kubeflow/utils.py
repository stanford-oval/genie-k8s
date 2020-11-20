from datetime import datetime
import os

from kubernetes.client.models import V1EnvVar
import kfp

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


def list_pipelines(client):
    resp = client.list_pipelines(page_size=100)
    pipelines = resp.pipelines
    while resp.next_page_token:
        resp = client.list_pipelines(page_token=resp.next_page_token, page_size=100)
        pipelines.extend(resp.pipelines)
    return pipelines


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


