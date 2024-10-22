# Epita-S9-DNN

## Description

This repository presents the project I carried out at **Epita** in the **SCIA 2023** major for the DNN (Deep Neural Network) subject.

**The aim of this project is to re-implement the main innovative points presented by a research paper.**

## Work done

**My work is on the paper: [Axiomatic Attribution for Deep Neural Networks](https://arxiv.org/abs/1703.01365)**.

It is divided into 3 parts:
- *Reading & Summary of the paper as well as additional papers,*
    - 👉 [**/slides**](slides/README.md)
- *Naive exploration of the method presented in the paper, using an excellent online tutorial, using the Tensorflow librairy*
    - 👉 **/notebooks/tensorflow.ipynb**
- *A more optimised implementation of the method presented in the article using the PyTorch library*
    - 👉 **/notebooks/pytorch.ipynb**



## Installation
Just create a venv, install requirements and run the notebooks from the root directory.

### We recommend using pyenv to manage your python versions and poetry to manage your virtual environments
```
> pyenv install 3.9.7
> pyenv shell 3.9.7
> poetry install
> poetry shell
```

#### In case of problem

Try the following :

```
> pyenv install 3.9.7
> pyenv shell 3.9.7
> rm -rf .venv poetry.lock
> cp .poetry.lock poetry.lock
> poetry install
> poetry shell
```

### Alternatively you can use python's venv module

> **We are assuming you have python 3.9.7 installed !**

```
> python -m venv env
> source env/bin/activate
> python -m pip install -r requirements.txt
> python src/main.py
```

## Repository Convention & Architecture

### Architecture

* The notebooks folder contains the notebooks that we will return with our results
* All algorithms performing complex or persistent operations must be implemented in the src folder, one file per feature. Give preference to object-oriented programming. These objects will then be called in the notebooks.
* We use Poetry for managing virtual environments & installing dependencies. See
    * https://python-poetry.org/
    * https://github.com/pyenv/pyenv

### Convention

* **[DOCSTRING]** : We use typed google docstrings for all functions and methods. See https://www.python.org/dev/peps/pep-0484/ and https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html . Docstring and notebooks should be written in English.
* **[TEST]** : All files in src should be summarily tested. Ideally, leave a simplistic example of use commented out at the bottom of each file to demonstrate its use. No need to use PyTest absolutely, we're not monsters either :D
* **[GIT]** : We use the the commit convention described here: https://www.conventionalcommits.org/en/v1.0.0/ . You should never work on master, but on a branch named after the feature you are working after opening an issue to let other members know what you are working on so that you can discuss it. When you are done, you can open a pull request to merge your branch into master. We will then review your code and merge it if everything is ok. Issues and pull requests can be written in French.

> Of course, anyone who doesn't follow these rules, arbitrarily written by a tyrannical mind, is subject to judgmental looks, cookie embargoes and denunciatory messages with angry animal emojis.
