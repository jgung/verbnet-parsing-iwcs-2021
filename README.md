# Semantic Role Labeling with VerbNet Classes

Code used for experiments for the IWCS 2021 paper *Predicate Representations and Polysemy in VerbNet Semantic Parsing*. This repository is cloned from [https://github.com/jgung/tf-nlp](https://github.com/jgung/tf-nlp).

## Setup
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
