# Epita-S9-DNN

## Description

This repository presents the project carried out at **Epita** in the **SCIA 2023** major for the DNN (Deep Neural Network) subject by the students:

* adrien.merat@epita.fr
* corentin.duchene@epita.fr
* hao.ye@epita.fr
* henri.jamet@epita.fr
* theo.perinet@epita.fr

The aim of this project is to re-implement the main innovative points presented by a research paper.

### Project Description

We choose to re-implement the paper [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365). This paper presents a new method to explain the decisions of a neural network.

In this project, we will do the following:
1. Follow a [simpler tutorial](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
2. Follow a more detailled and [complete tutorial](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/integrated_gradients/integrated_gradients.ipynb)
3. Create and train our own network for image classification [MNIST](http://yann.lecun.com/exdb/mnist/) to apply our own implementation of Integrated Gradients.

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