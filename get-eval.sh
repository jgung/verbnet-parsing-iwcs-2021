#!/bin/bash

job="$1"
jobpath="gs://gunglab/experiments/${job}"
out="$2"
outpath=${out}

mkdir -p ${outpath}

gsutil cp -n ${jobpath}/config.json ${outpath}
gsutil cp -n ${jobpath}/eval-summary.*.tsv ${outpath}
gsutil cp -n ${jobpath}/train.log ${outpath}
gsutil cp -n ${jobpath}/eval.log ${outpath}
gsutil cp -n ${jobpath}/*.eval*.txt ${outpath}
