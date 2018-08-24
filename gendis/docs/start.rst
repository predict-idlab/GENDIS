Getting Started
===============

1. Loading & preprocessing the datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a first step, we need to construct at least a matrix with timeseries
(``X_train``) and a vector with labels (``y_train``). Additionally, test
data can be loaded as well in order to evaluate the pipeline in the end.

.. code:: python

   import pandas as pd
   # Read in the datafiles
   train_df = pd.read_csv(<DATA_FILE>)
   test_df = pd.read_csv(<DATA_FILE>)
   # Split into feature matrices and label vectors
   X_train = train_df.drop('target', axis=1)
   y_train = train_df['target']
   X_test = test_df.drop('target', axis=1)
   y_test = test_df['target']

2. Creating a ``GeneticExtractor`` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Construct the object. For a list of all possible parameters, and a
description, please refer to the documentation in the `code`_

.. code:: python

   from gendis.genetic import GeneticExtractor
   genetic_extractor = GeneticExtractor(population_size=50, iterations=25, verbose=False, 
                                        normed=False, add_noise_prob=0.3, add_shapelet_prob=0.3, 
                                        wait=10, plot='notebook', remove_shapelet_prob=0.3, 
                                        crossover_prob=0.66, n_jobs=4)

3. Fit the ``GeneticExtractor`` and construct distance matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   shapelets = genetic_extractor.fit(X_train, y_train)
   distances_train = genetic_extractor.transform(X_train)
   distances_test = genetic_extractor.transform(X_test)

4. Fit ML classifier on constructed distance matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   lr = LogisticRegression()
   lr.fit(distances_train, y_train)

   print('Accuracy = {}'.format(accuracy_score(y_test, lr.predict(distances_test))))


.. _code: gendis/genetic.py