#!/bin/bash

set -e
set -o pipefail

pwd=$(pwd)
model=$1
dataset=$2
shift
shift

dataset_basename=$(basename $dataset)

thingpedia=/srv/data/home/gcampagn/pldi19-artifact/thingpedia-8strict.json

echo -n "$model," | sed -E 's/-([0-9]+),$/,\1,/'
case $dataset_basename in
*dialog*.txt)
	node ~/mobisocial/genie-toolkit/tool/genie.js evaluate-dialog --url "file://$pwd/$model" --thingpedia $thingpedia $dataset --no-debug --csv "$@"
	;;
*)
	node ~/mobisocial/genie-toolkit/tool/genie.js evaluate-server --contextual --url "file://$pwd/$model" --thingpedia $thingpedia $dataset --no-debug --csv "$@"
	;;
esac
