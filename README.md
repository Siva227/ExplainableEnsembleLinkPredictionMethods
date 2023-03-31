[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Explainable Ensemble Link Prediction
====================================

This repository contains experiments and code written for the purpose of explainable ensemble link prediction.

Project Organization
--------------------
    ├── README.md               <- The top-level README for developers using this project.
    ├── data                    
    │   |── external            <- Data from third party sources.
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data sets for modeling.
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
    │  
    ├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                              the creator's initials, and a short `-` delimited description, e.g.
    │                              `01.0-ss-initial-data-exploration`.
    │   
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures             <- Generated graphics and figures to be used in reporting
    │
    ├── requirements            <- The requirements file for reproducing the analysis environment
    │
    ├── eelp  
    │   ├── models              <- Modules containing graph models used in the project
    │   ├── scripts             <- Click scripts to download or generate data
    │   └── utils               <- Modules containing utility functions
    │
    └── tox.ini                 <- tox file with settings for running tox; see tox.readthedocs.io
    

Setting up a clean environment
------------------------------
- Create a virtual environment with Python version 3.10
- Install developer requirements using command `pip install -r requirements/dev_requirements.txt`
- Install the editable version of this repository by running the command `pip install -e .` in the project root directory
