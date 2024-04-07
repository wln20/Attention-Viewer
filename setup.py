#!/usr/bin/env python

from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="attn_viewer",
    version="0.1.0",
    description="Visualization tool for activation weights",
    author="Luning Wang",
    author_email="wangluning2@gmail.com",
    url="https://github.com/wln20/Attention-Viewer.git",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    exclude = ["main"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
