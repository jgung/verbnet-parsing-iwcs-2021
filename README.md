# Tensorflow NLP

This project aims to provide modular boilerplate code for natural language processing applications 
(sequence tagging, semantic role labeling, and parsing) implemented in Tensorflow. 
The project is compatible with [Google Cloud Platform](https://cloud.google.com/) 
and [ML Engine](https://cloud.google.com/ml-engine/) for easy cloud-based training.

## Getting Started
### Local Prerequisites
* Python 3
* virtualenv
* (optional) TF-compatible GPU (CUDA 9 / CuDNN 7)

You can use virtualenv to create a self-contained installation with the necessary dependencies. 
For GPU training, you will need to follow the [TF GPU installation instructions](
https://www.tensorflow.org/install/install_linux#NVIDIARequirements).

```bash
virtualenv -p python3 ~/.venvs/tfnlp
source ~/.venvs/tfnlp/bin/activate
cd tfnlp
# if no compatible GPU available, use requirements-cpu.txt
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:`pwd`
python tfnlp/trainer.py
```

We use [GloVe word vectors](https://nlp.stanford.edu/projects/glove/) in the below examples.
You can download them as follows:
```
wget -O data/vectors/glove.6B.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip data/vectors/glove.6B.zip -d data/vectors && rm data/vectors/glove.6B.zip
```


## Example Applications
We provide several examples of NLP applications–named entity recognition, semantic role labeling, 
and dependency parsing.

### Named Entity Recognition
We can closely replicate the 
[LSTM-CNN-CRF architecture from Xuezhe Ma and Eduard Hovy (2016)](
http://www.aclweb.org/anthology/P16-1101) 
using the provided configuration in `data/config/ner/ner-glove-config.json`.

Follow the [instructions](https://www.clips.uantwerpen.be/conll2003/ner/) 
for downloading and preparing the CoNLL 2003 English NER dataset.
You should end up with 3 files,
`eng.train`, `eng.testa`, and `eng.testb`, which will comprise our training, development, 
and test data. Note that the official CoNLL 2003 evaluation script requires Perl to run.

You can then begin training with the following command:
```bash
python tfnlp.trainer.py --job-dir data/experiments/conll-03 \
--train path/to/conll03/eng.train \
--valid path/to/conll03/eng.testa \
--test path/to/conll03/eng.testb \
--config data/config/ner/ner-glove-config.json \
--resources data/
```
where `path/to/conll03` gives the location of the CoNLL 2003 corpus.
You can run the same command to resume training from a saved checkpoint,


`--job-dir` provides the output directory for a particular run.
Feature vocabularies, extracted features, checkpoints and final models are saved here.

`--resources data/` indicates the base path to any resources, such as word embeddings, 
specified in the configuration file given by `--config data/config/ner/ner-glove-config.json`.

Typically, the development F1 score reaches ~94, with a test F1 score of ~90.5.

To train using ELMo–described in the paper 
[Deep contextualized word representations](https://arxiv.org/abs/1802.05365), 
use `ner-elmo-config.json` instead of `ner-glove-config.json`.

To train using a model with BERT, described in
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
use `ner-bert-config.json`:
```
INFO:tensorflow:Evaluating on data/datasets/conll03-ner/eng.testb
INFO:tensorflow:processed 46666 tokens with 5648 phrases; found: 5731 phrases; correct: 5208.
accuracy:  98.29%; precision:  90.87%; recall:  92.21%; FB1:  91.54
              LOC: precision:  93.71%; recall:  92.87%; FB1:  93.29  1653
             MISC: precision:  79.14%; recall:  84.33%; FB1:  81.66  748
              ORG: precision:  88.24%; recall:  91.27%; FB1:  89.73  1718
              PER: precision:  96.22%; recall:  95.92%; FB1:  96.07  1612
```

### Semantic Role Labeling
To train a model based on
[Deep Semantic Role Labeling: What works and what's next](
https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf), 
you will need to download and prepare PropBank data from either the 
[CoNLL-2005 shared task](http://www.lsi.upc.edu/~srlconll/soft.html) 
or the the train-dev-test split from [CoNLL-2012](http://cemantix.org/data/ontonotes.html).
Instructions for downloading and preparing the data can be found at 
https://github.com/jgung/semantic-role-labeling.

For CoNLL-2005, you should end up with 4 files: `train-set.conll` and `dev-set.conll`, 
and optional test files `test-wsj.conll` and `test-brown.conll`.

You can then begin training using the following command:
```bash
python tfnlp.trainer.py --job-dir data/experiments/conll-05 \
--train path/to/conll05/train-set.conll \
--valid path/to/conll05/dev-set.conll \
--test path/to/conll05/test-wsj.conll \
--config data/config/srl/srl-glove-config.json \
--resources data/
```
You can run the same command to resume training from a saved checkpoint. 
The development F1 score typically reaches between 80 and 81.

For training on CoNLL-2012, you will only need to change the `"reader": "conll_2005"` field in 
`srl-glove-config.json` to `"reader": "conll_2012"`, or override it using `--param_overrides reader=conll_2012`.

To train with BERT, use `data/config/srl/srl-bert-config.json` instead of `srl-glove-config.json`.

### Dependency Parsing
We also provide an implementation of the graph-based dependency parser described in the paper 
[Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734).

To train on CoNLL-2009 data, you will need to follow the 
[instructions](http://ufal.mff.cuni.cz/conll2009-st/train-dev-data.html) 
available for the official shared task. For replicating the PTB-SD experiments, 
https://github.com/hankcs/TreebankPreprocessing provides a script to preprocess WSJ data 
into the Stanford Dependency format.

To train on the English CoNLL-2009 dependency data (using provided predicted POS tags), 
you can use the following command:
```bash
python tfnlp.trainer.py --job-dir data/experiments/conll-09-en \
--train path/to/CoNLL2009-ST-English/CoNLL2009-ST-English-train.txt \
--valid path/to/CoNLL2009-ST-English-development.txt \
--config data/config/parsing/parser-config.json \
--script data/scripts/eval09.pl \
--resources data/
```
For Stanford Dependency (CoNLL-X formatted) corpora, you should instead use `parser-sd-config.json`.

### Google Cloud Platform
Alternatively, you can train models via [Google Cloud ML Engine](https://cloud.google.com/ml-engine/).

Just [install the SDK](https://cloud.google.com/sdk/install).
Then [create a storage bucket](https://cloud.google.com/storage/docs/creating-buckets)
to persist your training results.

A script, `./gcloud-train.sh` has been provided to simplify the training process:
```text
./gcloud-train.sh --config path/to/config.json --train path/to/train.txt --valid path/to/valid.txt --test path/to/test.txt --bucket bucket_name
        -h --help
        --config        Path to .json file used to configure features and model hyper-parameters
        --train         Path to training corpus file
        --valid         Path to validation corpus file
        --test          Comma-separated list of paths to test files (optional)
        --bucket        Google Cloud Storage bucket name
        --job-name      Job name (optional)
        --runtime       Tensorflow runtime version (optional, 1.13 by default)
```

Alternatively, follow the [latest instructions](
https://cloud.google.com/ml-engine/docs/tensorflow/training-jobs) 
for running a training job in ML Engine. You can use the same training arguments you would locally after the empty `--` flag, 
other than specifying `--job-dir` before this flag.