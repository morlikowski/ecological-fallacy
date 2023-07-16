# The Ecological Fallacy in Annotation
This repository contains experimentation code for Orlikowski et al. (2023): "The Ecological Fallacy in Annotation: Modelling Human Label Variation goes beyond Sociodemographics" ([Link to paper](https://aclanthology.org/2023.acl-short.88/)).

The work was presented at ACL 2023. Please cite the paper if any of the code is useful to you:

> Matthias Orlikowski, Paul Röttger, Philipp Cimiano, and Dirk Hovy. 2023. The Ecological Fallacy in Annotation: Modeling Human Label Variation goes beyond Sociodemographics. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1017–1029, Toronto, Canada. Association for Computational Linguistics.


## Setup

The project uses [poetry](https://python-poetry.org/docs/) to manage dependencies. See their documentation for a description of how to install it.

You should be able to install the package and its dependecies for development using `poetry install` in the project directory. The repository is already initalized, that is, it contains a `pyproject.toml` and `poetry.lock` file.

## Usage

The main script is `run.py` in the `ecological_fallacy` top-level module. 

Example: `poetry run python -m ecological_fallacy.run experiments/0.2-baseline/`

The script expects a path to an experiment directory as the only argument. The experiment directory is expected to contain a `config.json` and will be used to write results to a CSV file. Example files for configs and results can be found in the subdirectories of `experiments/`.

## Tests

There are tests in `tests/` for some important parts of the custom code. Note that currently a few of the tests expect a data sample in a specific location which can be created with `notebooks/02_create_sample_kumar.ipynb`.
