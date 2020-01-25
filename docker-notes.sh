docker run --gpus all -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow:1.13.1-gpu-py3
pip install -r requirements.txt && export PYTHONPATH=/tmp/tfnlp: && python tfnlp/trainer.py


