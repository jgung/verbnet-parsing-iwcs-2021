from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "tensorflow-gpu==1.15.0",
    "tensorflow-hub==0.7.0",
    "tensor2tensor==1.15.5",
    "bert-tensorflow==1.0.1",
    "tensorflow-probability==0.8.0",
    "numpy>=1.14.2",
    "nltk>=3.2.5",
]

setup(
    name="tfnlp",
    version="1.0",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description="Common NLP applications implemented in Tensorflow.",
    author="James Gung",
    author_email="gungjm@gmail.com",
    url="https://github.com/jgung/tf-nlp"
)
