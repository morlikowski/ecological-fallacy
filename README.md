# The Ecological Fallacy in Annotation
This repository contains experiment code for Orlikowski et al. (2023): "The Ecological Fallacy in Annotation: Modelling Human Label Variation goes beyond Sociodemographics" ([Link to Preprint](https://arxiv.org/abs/2306.11559)).

The paper will be presented at ACL 2023. For now, please cite the preprint version if any of the code is useful to you.


## Setup

The project uses [poetry](https://python-poetry.org/docs/) to manage dependencies.

You should be able to install the package and its dependecies for development using `poetry install` in the project directory. The repository is already initalized, that is, it contains a `pyproject.toml` and `poetry.lock` file.

## Usage

The main script is `run.py` in the `ecological_fallacy` top-level module. 

Example: `poetry run python -m ecological_fallacy.run experiments/0.2-baseline/`

The script expects a path to an experiment directory as the only argument. The experiment directory is expected to contain a `config.json` and will be used to write results to a CSV file. Example files for configs and results can be found in the subdirectories of `experiments/`.

## Tests

There a tests in `tests/` for some important parts of the custom code. Note that currently a few of the tests expect a data sample in a specific location which can be created with `notebooks/02_create_sample_kumar.ipynb`.
