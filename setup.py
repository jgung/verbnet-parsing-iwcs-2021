from setuptools import find_packages, setup

REQUIRED_PACKAGES = ["tensorflow-gpu==1.12.0", "numpy>=1.14.2", "nltk>=3.2.5", "tensorflow-hub>=0.1.1", "tensor2tensor>=1.9.0"]

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
