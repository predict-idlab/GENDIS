# GENDIS: GENetic DIscovery of Shapelets

In the time series classification domain, shapelets are small subseries that are discriminative for a certain class. It has been shown that by projecting the original dataset to a distance space, where each axis corresponds to the distance to a certain shapelet, classifiers are able to achieve state-of-the-art results on a plethora of datasets.

This repository contains an implementation of `GENDIS`, an algorithm that searched for a set of shapelets in a genetic fashion. The algorithm is insensitive to its parameters (such as population size, crossover and mutation probability, ...) and can quickly extract a small set of shapelets that is able to achieve predictive performances similar (or better) to that of other shapelet techniques.

## Installation

## Example

A simple example is provided in [this notebook](gendis/example.ipynb)

## Paper experiments

In order to reproduce the results from the corresponding paper...

### Dependent vs independent discovery

### Simple artificial two-class problem that cannot be solved by brute-force approach

### [LTS](https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf) vs GENDIS

## Tests

We provide a few doctest. Deploy them by running `python3 -m doctest -v <FILE>`, where `<FILE>` is the Python file you want to run the doctests from.

## Cite

For now, please refer to this repository. A paper, to which you can then refer, will be published in the nearby future.
