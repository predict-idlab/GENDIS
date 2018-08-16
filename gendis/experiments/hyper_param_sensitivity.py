import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import random
import pandas as pd

import sys
sys.path.append('..')
from genetic import GeneticExtractor

from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from tslearn.shapelets import ShapeletModel


def grabocka_params_to_shapelet_size_dict(n_ts, ts_sz, n_shapelets, l, r):
    base_size = int(l * ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        d[shp_sz] = n_shapelets
    return d

# Load in our data
TRAIN_PATH = '../data/partitioned/SonyAIBORobotSurface2/SonyAIBORobotSurface2_train.csv'
TEST_PATH = '../data/partitioned/SonyAIBORobotSurface2/SonyAIBORobotSurface2_test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target']
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target']

map_dict = {}
for j, c in enumerate(np.unique(y_train)):
    map_dict[c] = j
y_train = y_train.map(map_dict) 
y_test = y_test.map(map_dict)

gendis_results = []
lts_results = []

i = 0
while i < 15:
	try:
	    # Sample random hyper-parameters for LTS
	    K = np.random.choice([0.05, 0.15, 0.3])
	    L = np.random.choice([0.025, 0.075, 0.125, 0.175, 0.2])
	    R = np.random.choice([1, 2, 3])
	    _lambda = np.random.choice([0.01, 0.1, 1])
	    n_iterations = np.random.choice([2000, 5000, 10000])

	    shapelet_dict = grabocka_params_to_shapelet_size_dict(
	            X_train.shape[0], X_train.shape[1], int(K*X_train.shape[1]), L, R
	    )
	    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
	                        max_iter=n_iterations, verbose_level=0, batch_size=1,
	                        optimizer='sgd', weight_regularizer=_lambda)

	    clf.fit(
	        np.reshape(
	            X_train, 
	            (X_train.shape[0], X_train.shape[1], 1)
	        ), 
	        y_train
	    )

	    X_distances_train = clf.transform(X_train)
	    X_distances_test = clf.transform(X_test)

	    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
	    lr.fit(X_distances_train, y_train)

	    acc = accuracy_score(y_test, lr.predict(X_distances_test))

	    print([K, L, R, _lambda, n_iterations], acc)

	    lts_results.append([K, L, R, _lambda, n_iterations, acc])

	    # Sample random hyper-parameters for GENDIS
	    wait = np.random.choice([5, 10, 25, 50])
	    cx_prob = np.random.choice([0.1, 0.25, 0.5, 0.9])
	    mut_prob = np.random.choice([0.1, 0.25, 0.5, 0.9])
	    pop_size = np.random.choice([10, 50, 100, 250])
	    genetic_extractor = GeneticExtractor(verbose=True, population_size=pop_size, iterations=100, wait=wait,
	                                         add_noise_prob=mut_prob, add_shapelet_prob=mut_prob, 
	                                         remove_shapelet_prob=mut_prob, crossover_prob=cx_prob)
	    shapelets = genetic_extractor.fit(X_train, y_train)

	    X_distances_train = genetic_extractor.transform(X_train)
	    X_distances_test = genetic_extractor.transform(X_test)

	    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
	    lr.fit(X_distances_train, y_train)

	    acc = accuracy_score(y_test, lr.predict(X_distances_test))

	    print([wait, cx_prob, mut_prob, pop_size], acc)

	    gendis_results.append([wait, cx_prob, mut_prob, pop_size, acc])
	    i += 1
	except:
		pass



lts_df = pd.DataFrame(lts_results)
lts_df.columns = ['K', 'L', 'R', '_lambda', 'n_iterations', 'acc']
gendis_df = pd.DataFrame(gendis_results)
gendis_df.columns = ['wait', 'cx_prob', 'mut_prob', 'pop_size', 'acc']

<<<<<<< HEAD
lts_df.to_csv('results/lts_hyperparams_SonyAIBORobotSurface2.csv')
gendis_df.to_csv('results/gendis_hyperparams_SonyAIBORobotSurface2.csv')
=======
lts_df.to_csv('results/lts_hyperparams_SonyAIBORobotSurface1.csv')
gendis_df.to_csv('results/gendis_hyperparams_SonyAIBORobotSurface1.csv')
>>>>>>> e6e69ca34a4b4c1b4d84f9e6d2b316887a29cc34
