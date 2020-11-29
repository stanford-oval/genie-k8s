import kfp
import os
import argparse
from utils import list_pipelines, list_experiments

parser = argparse.ArgumentParser()

parser.add_argument('--owner', type=str)
parser.add_argument('--project', type=str)
parser.add_argument('--experiment', type=str)
parser.add_argument('--image', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--train_additional_args', type=str)
parser.add_argument('--eval_additional_args', type=str)
parser.add_argument('--pipeline_name', type=str)
parser.add_argument('--pipeline_version', type=str)
parser.add_argument('--kf_experiment_name', type=str)
parser.add_argument('--kf_job_name', type=str)

args = parser.parse_args()

kfp_dir = os.path.expanduser('~/.config/kfp/')
if not os.path.isdir(kfp_dir):
    os.makedirs(kfp_dir)
client = kfp.Client()
client.set_user_namespace('research-kf')

pipelines = list_pipelines(client)
our_pipeline = None

for p in pipelines:
    if p.name == args.pipeline_name:
        our_pipeline = p

if our_pipeline is None:
    raise ValueError('No pipelines with this name were found')

pipeline_versions = client.pipelines.list_pipeline_versions(
    resource_key_type="PIPELINE",
    resource_key_id=our_pipeline.id,
    page_token='',
).versions

our_pipeline_version = None
for v in pipeline_versions:
    if v.name == args.pipeline_version:
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


params = {
    'owner': args.owner,
    'project': args.project,
    'experiment': args.experiment,
    'image': args.image,
    'model': args.model,
    'dataset': args.dataset,
    'train_additional_args': args.train_additional_args,
    'eval_additional_args': args.eval_additional_args,

}


client.run_pipeline(experiment_id=our_experiment.id,
                    job_name=args.kf_job_name,
                    params=params,
                    pipeline_id=our_pipeline.id,
                    version_id=our_pipeline_version.id)