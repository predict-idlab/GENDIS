.. GENDIS documentation master file, created by
   sphinx-quickstart on Thu Aug 23 11:07:52 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GENDIS's documentation!
==================================

In the time series classification domain, shapelets are small subseries
that are discriminative for a certain class. It has been shown that by
projecting the original dataset to a distance space, where each axis
corresponds to the distance to a certain shapelet, classifiers are able
to achieve state-of-the-art results on a plethora of datasets.

This repository contains an implementation of ``GENDIS``, an algorithm
that searches for a set of shapelets in a genetic fashion. The algorithm
is insensitive to its parameters (such as population size, crossover and
mutation probability, …) and can quickly extract a small set of
shapelets that is able to achieve predictive performances similar (or
better) to that of other shapelet techniques.

.. toctree::
   :maxdepth: 2

   Installation <install.rst>
   Getting Started <start.rst>
   GENDIS <gendis.rst>
   Contributing, Citing and Contact <ccc.rst>
