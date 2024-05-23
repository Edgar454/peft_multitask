# setup.py
from setuptools import setup, find_packages

setup(
    name="peft_multitask",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        "torch",
        "transformers",

    ],
    authors="Edgar Meva'a , Medtech",
    author_email="mevaed4@gmail.com",
    description="A multitask learning library with parameter-efficient fine-tuning for Donut.",
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Edgar404/peft_multitask",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
