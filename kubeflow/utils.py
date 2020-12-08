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
    
    # relax following assertion since we may pass several arguments as value (e.g. xxx_additional_args)
    # assert all(not args_list[i].startswith(("-", "--")) for i in range(1, len(args_list), 2))
    
    cleaned_args = {}
    for i in range(0, len(args_list), 2):
        arg, value = args_list[i], args_list[i+1]
        cleaned_args[arg.strip('-')] = value
        
    return cleaned_args
