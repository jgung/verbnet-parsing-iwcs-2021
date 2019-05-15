#!/bin/bash

job="$1"
jobpath="gs://gunglab/experiments/${job}"
out="$2"
outpath=${out}/${job}

mkdir -p ${outpath}

gsutil cp ${jobpath}/config.json ${outpath}
gsutil cp ${jobpath}/predictions.* ${outpath}
gsutil cp -R ${jobpath}/vocab ${outpath}

mkdir -p ${outpath}/model
gsutil -m cp -R ${jobpath}/model/export ${outpath}/model/
