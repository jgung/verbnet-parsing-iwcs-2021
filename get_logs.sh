#!/bin/bash

outputdir=$2
if [[ -z ${outputdir} ]]; then
    outputdir=`pwd`
fi

filelines=`cat $1`
mkdir -p ${outputdir}

for line in ${filelines} ; do
    gcloud ml-engine jobs stream-logs ${line} --allow-multiline-logs > ${outputdir}/${line}.log &
done