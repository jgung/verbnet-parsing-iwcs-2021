#!/bin/bash

job="$1"
jobpath="gs://gunglab/experiments/${job}"
out="$2"
outpath=${out}

mkdir -p ${outpath}

gsutil cp -n ${jobpath}/config.json ${outpath}
gsutil cp -n ${jobpath}/predictions.* ${outpath}
gsutil cp -n ${jobpath}/eval-summary.*.tsv ${outpath}
gsutil cp -n ${jobpath}/train.log ${outpath}
gsutil cp -n ${jobpath}/eval.log ${outpath}
gsutil cp -n ${jobpath}/*.eval*.txt ${outpath}
gsutil cp -R -n ${jobpath}/vocab ${outpath}

mkdir -p ${outpath}/model
gsutil -m cp -R -n ${jobpath}/model/export ${outpath}/model/
