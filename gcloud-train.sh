#!/bin/bash

program_name=$0

function usage()
{
    echo "Train and test model specified by a given configuration file in Google Cloud ML Engine."
    echo ""
    echo "$program_name --config path/to/config.json --train path/to/train.txt --valid path/to/valid.txt --test path/to/test.txt --bucket bucket_name"
    echo -e "\t-h --help"
    echo -e "\t--config\tPath to .json file used to configure features and model hyper-parameters"
    echo -e "\t--train\t\tPath to training corpus file"
    echo -e "\t--valid\t\tPath to validation corpus file"
    echo -e "\t--test\t\tComma-separated list of paths to test files"
    echo -e "\t--bucket\tGoogle Cloud Storage bucket name"
    echo -e "\t--job-name\tJob name (optional)"
}

while [[ $# -gt 0 ]]
do
key="$1"

case ${key} in
    -h|--help)
    usage
    exit
    ;;
    -c|--config)
    config=$2
    shift
    shift
    ;;
    -t|--train)
    train_file=$2
    shift
    shift
    ;;
    -d|-v|--valid|--dev)
    valid_file=$2
    shift
    shift
    ;;
    --test)
    comma_separated_test_files=$2
    shift
    shift
    ;;
    --job-name|--name|--tag)
    base_job_name=$2
    shift
    shift
    ;;
    --bucket)
    bucket_name=$2
    shift
    shift
    ;;
    *)
    echo "Unknown option: $1"
    usage
    exit 1
    ;;
esac
done

if [[ -z "CONFIG" ]] || [[ -z "$train_file" ]] || [[ -z "$valid_file" ]] || [[ -z "$comma_separated_test_files" ]]; then
    usage
    exit
fi

now=$(date +"%Y%m%d_%H%M%S")

if [[ -z "$base_job_name" ]]; then
    base_job_name=$(basename ${config} .json)
    echo "Using default job name (${base_job_name}) since none was provided (use --job-name to specify one)"
fi

job_name="${base_job_name}_${now}"
job_dir=gs://${bucket_name}/experiments/${job_name}

echo "Setting output directory to $job_dir"

python setup.py sdist
gsutil cp dist/tfnlp-1.0.tar.gz ${job_dir}/app.tar.gz
gsutil cp ${config} ${job_dir}/config.json
gsutil cp ${train_file} ${job_dir}/train.txt
gsutil cp ${valid_file} ${job_dir}/valid.txt

cloud_test_files=""
for local_test_file in ${comma_separated_test_files//,/ }
do
    cloud_test_file="${job_dir}/${local_test_file##*/}"
    gsutil cp ${local_test_file} ${cloud_test_file}
    cloud_test_files="${cloud_test_files},${cloud_test_file}"
done


gcloud ml-engine jobs submit training ${job_name} \
--packages ${job_dir}/app.tar.gz \
--config config.yaml \
--runtime-version 1.10 \
--module-name tfnlp.trainer \
--region us-east1 \
--stream-logs \
-- \
--job-dir ${job_dir} \
--train ${job_dir}/train.txt \
--valid ${job_dir}/valid.txt  \
--test ${cloud_test_files} \
--output predictions.txt \
--mode train \
--config ${job_dir}/config.json \
--resources gs://${bucket_name}/resources/
