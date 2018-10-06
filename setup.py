from setuptools import setup, find_packages

REQUIRED_PACKAGES = ["tensorflow-gpu>=1.10.0", "numpy>=1.14.2", "nltk>=3.2.5", "tensorflow-hub>=0.1.1", "tensor2tensor>=1.9.0"]

setup(
    name="tfnlp",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    install_package_data=True,
    description="Common NLP applications implemented in Tensorflow."
)
