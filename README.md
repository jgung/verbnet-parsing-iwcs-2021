# Semantic Role Labeling with VerbNet Classes

Code used for experiments for the [IWCS 2021](https://iwcs2021.github.io/index.html) paper *Predicate Representations and Polysemy in VerbNet Semantic Parsing*. This repository is cloned from [https://github.com/jgung/tf-nlp](https://github.com/jgung/tf-nlp).

## Setup
### Data
We use a subset of [SemLink 1.1](https://verbs.colorado.edu/semlink/) with VerbNet roles fully mapped to PropBank for our experiments.

You'll need at a minimum:
* A copy of the Penn Treebank [Treebank-3](https://catalog.ldc.upenn.edu/LDC99T42) release through LDC
* Perl, with Perl modules from [srlconll-1.1.tgz](http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz) installed
  * e.g. `export PERL5LIB=$HOME/soft/srlconll-1.1/lib:$PERL5LIB`

Run the provided script at `verbnet-parsing-iwcs-2021/scripts/semlink/semlink2conll.sh` from the root of this repo:
```bash
scripts/semlink/semlink2conll.sh --ptb path/to/ptb/parsed/mrg/
--roleset both \
--senses vn \
--brown data/datasets/conll-brown-release/prop.txt
```

This should result in the following files:
* `semlink1.1/vn.both-train.txt`
* `semlink1.1/vn.both-valid.txt`
* `semlink1.1/vn.both-test-wsj.txt`
* `semlink1.1/vn.both-test-brown.txt`

Each file should have PropBank and VerbNet thematic roles with VerbNet classes:
```text
 0 0 Ford NNP - - (ARG0$Agent* * 
 0 1 Motor NNP - - * * 
 0 2 Co. NNP - - *) * 
 0 3 said VBD 37.7 say (V*) * 
 0 5 it PRP - - (ARG1$Topic* (ARG0$Agent*) 
 0 6 acquired VBD 13.5.2-1 acquire * (V*) 
 0 7 5 CD - - * (ARG1$Theme* 
 0 8 % NN - - * * 
 0 9 of IN - - * * 
 0 10 the DT - - * * 
 0 11 shares NNS - - * * 
 0 12 in IN - - * * 
 0 13 Jaguar NNP - - * * 
 0 14 PLC NNP - - *) *) 
 0 15 . . - - * * 
```


### GPU
Given that this project uses an earlier version of Tensorflow which is incompatible with the latest version of Cuda, it's easier to use Docker to run:

1. [Install Docker](https://docs.docker.com/engine/install/).
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit).
3. Download the Tensorflow 1.13.2 docker image:
```bash
docker pull tensorflow/tensorflow:1.13.2-gpu-py3
```
4. Clone this repo if you haven't already, then run the TF image in a container in interactive mode:
```bash
git clone https://github.com/jgung/verbnet-parsing-iwcs-2021
cd verbnet-parsing-iwcs-2021
docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.13.2-gpu-py3
```
5. You should now be inside the container. Install dependencies and call the trainer CLI:
```bash
pip install -r requirements.txt && export PYTHONPATH=/tmp/tfnlp: && python tfnlp/trainer.py
```
6. If all succeeded, you should see a help message.

### CPU
Running on CPU is not recommended for training, but can be done for inference. You'll need the following:

* Python 3.7
* virtualenv

Create a virtual environment, install dependencies for the repository and try to run the CLI:
```bash
virtualenv -p python3.7 ~/.venvs/tfnlp
source ~/.venvs/tfnlp/bin/activate
cd verbnet-parsing-iwcs-2021
pip install -r requirements-cpu.txt && export PYTHONPATH=$PYTHONPATH:`pwd` && python3.7 tfnlp/trainer.py
```
You should see a help message if all was successful.

## Training
You can train and evaluate a model using the trainer CLI.

To train and evaluate the baseline model on PropBank roles, use:
```bash
python tfnlp/trainer.py \
--config iwcs-2021-config/baseline-config-pb.json \
--job-dir iwcs-2021/baseline-pb \
--resources data/ \
--train semlink1.1/vn.both-train.txt \
--valid semlink1.1/vn.both-valid.txt \
--test semlink1.1/vn.both-test-wsj.txt,semlink1.1/vn.both-test-brown.txt
```

Training typically completes in less than 2 hours on a GTX 1080TI.
Output will be saved to the directory given by `--job-dir`, in this case `iwcs-2021/baseline-pb`.

### Configs
A complete table of possible configs in `iwcs-2021-config/` is given below:

| Config | Paper Name | Description |
| --- | --- | --- |
| baseline-config-pb.json | Baseline | BERT baseline PropBank SRL model |
| baseline-config-vn.json | Baseline | BERT baseline VerbNet SRL model |
| joint-srl-vsd-pb.json | SRL + VSD | Multitask PropBank SRL/VerbNet classification model |
| joint-srl-vsd-vn.json | SRL + VSD | Multitask VerbNet SRL/VerbNet classification model |
| joint-srl-vsd-cond-pb.json | SRL &#124; VSD | Multitask PropBank SRL/VerbNet classification model, SRL conditioned on VerbNet class predictions |
| joint-srl-vsd-cond-vn.json | SRL &#124; VSD | Multitask VerbNet SRL/VerbNet classification model, SRL conditioned on VerbNet class predictions |
| predicted-class-pb.json | Predicted Class | PropBank SRL model with predicted VerbNet classes as features |
| predicted-class-vn.json | Predicted Class | VerbNet SRL model with predicted VerbNet classes as features |
| all-classes-pb.json | All Classes | PropBank SRL model with all possible VerbNet classes for each predicate as features |
| all-classes-vn.json | All Classes | VerbNet SRL model with all possible VerbNet classes for each predicate as features |
| gold-class-pb.json | Gold Class | PropBank SRL model with gold VerbNet classes as features |
| gold-class-vn.json | Gold Class | VerbNet SRL model with gold VerbNet classes as features |
