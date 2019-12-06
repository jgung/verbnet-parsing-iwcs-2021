#!/bin/bash

job="$1"
jobpath="gs://gunglab/experiments/${job}"
out="$2"
outpath=${out}

mkdir -p ${outpath}

gsutil cp ${jobpath}/config.json ${outpath}
gsutil cp ${jobpath}/predictions.* ${outpath}
gsutil cp ${jobpath}/eval-summary.tsv ${outpath}
gsutil cp ${jobpath}/train.log ${outpath}
gsutil cp ${jobpath}/eval.log ${outpath}
gsutil cp ${jobpath}/*.eval*.txt ${outpath}
gsutil cp -R ${jobpath}/vocab ${outpath}

mkdir -p ${outpath}/model
gsutil -m cp -R ${jobpath}/model/export ${outpath}/model/
