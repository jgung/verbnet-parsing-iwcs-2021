#!/bin/bash

PROGRAM_NAME=$0

function usage()
{
    echo "Train and test model specified by a given configuration file in Google Cloud ML Engine."
    echo ""
    echo "$PROGRAM_NAME --config path/to/config.json --train path/to/train.txt --valid path/to/valid.txt --test path/to/test.txt --bucket bucket_name"
    echo -e "\t-h --help"
    echo -e "\t--config\tPath to .json file used to configure features and model hyper-parameters"
    echo -e "\t--train\t\tPath to training corpus file"
    echo -e "\t--valid\t\tPath to validation corpus file"
    echo -e "\t--test\t\tPath to test corpus file path"
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
    CONFIG=$2
    shift
    shift
    ;;
    -t|--train)
    TRAIN_FILE=$2
    shift
    shift
    ;;
    -d|-v|--valid|--dev)
    VALID_FILE=$2
    shift
    shift
    ;;
    --test)
    TEST_FILE=$2
    shift
    shift
    ;;
    --job-name|--name|--tag)
    NAME=$2
    shift
    shift
    ;;
    --bucket)
    BUCKET_NAME=$2
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

if [[ -z "CONFIG" ]] || [[ -z "$TRAIN_FILE" ]] || [[ -z "$VALID_FILE" ]] || [[ -z "$TEST_FILE" ]]; then
    usage
    exit
fi

now=$(date +"%Y%m%d_%H%M%S")

echo $NAME

if [[ -z "$NAME" ]]; then
    NAME=$(basename $CONFIG .json)
    echo "Using default job name ($NAME) since none was provided (use --job-name to specify one)"
fi

echo $NAME

JOB_NAME="${NAME}_${now}"
echo $JOB_NAME
JOB_DIR=gs://$BUCKET_NAME/experiments/$JOB_NAME

echo "Setting output directory to $JOB_DIR"

python setup.py sdist
gsutil cp dist/tfnlp-1.0.tar.gz $JOB_DIR/app.tar.gz
gsutil cp $CONFIG $JOB_DIR/config.json
gsutil cp $TRAIN_FILE $JOB_DIR/train.txt
gsutil cp $VALID_FILE $JOB_DIR/valid.txt
gsutil cp $TEST_FILE $JOB_DIR/test.txt

gcloud ml-engine jobs submit training $JOB_NAME \
--packages $JOB_DIR/app.tar.gz \
--config config.yaml \
--runtime-version 1.10 \
--module-name tfnlp.trainer \
--region us-east1 \
--stream-logs \
-- \
--job-dir $JOB_DIR \
--train $JOB_DIR/train.txt \
--valid $JOB_DIR/valid.txt  \
--test $JOB_DIR/test.txt \
--mode train \
--config $JOB_DIR/config.json \
--resources gs://$BUCKET_NAME/resources/
