#!/usr/bin/env python
# setup.py - Para instalar Cerebro Zero como paquete

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cerebro-zero",
    version="1.0.0",
    author="Manuel (lecodev-26)",
    author_email="axiomsystemsechepares@gmail.com",
    description="🧠 Inteligencia Artificial desde cero en Termux",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lecodev-26/cerebro-zero",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.14",
    install_requires=[
        "numpy>=2.4.0",
        "flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "cerebro-zero=todo_en_uno:main",
        ],
    },
)
