import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
sys.path.append('..')

from genetic import GeneticExtractor
from data.load_all_datasets import load_data_train_test

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

def fit_lr(X_distances_train, y_train, X_distances_test, y_test, out_path):
    lr = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
    lr.fit(X_distances_train, y_train)
    
    hard_preds = lr.predict(X_distances_test)
    proba_preds = lr.predict_proba(X_distances_test)

    print("[LR] Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[LR] Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_lr_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_lr_proba.csv')

def fit_lts(X_train, y_train, X_test, y_test,  nr_shap, l, r, reg, max_it, shap_out_path, pred_out_path, timing_out_path):
    # Fit LTS model, print metrics on test-set, write away predictions and shapelets
    shapelet_dict = grabocka_params_to_shapelet_size_dict(
            X_train.shape[0], X_train.shape[1], int(nr_shap*X_train.shape[1]), l, r
    )
    
    clf = ShapeletModel(n_shapelets_per_size=shapelet_dict, 
                        max_iter=max_it, verbose_level=0, batch_size=1,
                        optimizer='sgd', weight_regularizer=reg)

    start = time.time()
    clf.fit(
        np.reshape(
            X_train, 
            (X_train.shape[0], X_train.shape[1], 1)
        ), 
        y_train
    )
    learning_time = time.time() - start

    print('Learning shapelets took {}s'.format(learning_time))

    with open(shap_out_path, 'w+') as ofp:
        for shap in clf.shapelets_:
            ofp.write(str(np.reshape(shap, (-1))) + '\n')

    with open(timing_out_path, 'w+') as ofp:
        ofp.write(str(learning_time))

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    fit_lr(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)

def fit_genetic(X_train, y_train, X_test, y_test, shap_out_path, pred_out_path, timing_out_path):
    genetic_extractor = GeneticExtractor(verbose=True, population_size=50, iterations=50, wait=25)
    start = time.time()
    shapelets = genetic_extractor.fit(X_train, y_train)
    genetic_time = time.time() - start

    print('Genetic shapelet discovery took {}s'.format(genetic_time))

    with open(shap_out_path, 'w+') as ofp:
        for shap in shap_transformer.shapelets:
            ofp.write(str(np.reshape(shap, (-1))) + '\n')

    with open(timing_out_path, 'w+') as ofp:
        ofp.write(str(genetic_time))

    X_distances_train = genetic_extractor.transform(X_train)
    X_distances_test = genetic_extractor.transform(X_test)

    fit_lr(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)

# For each dataset we specify the:
#    * Number of shapelets to extract of each length (specified as a fraction of TS length)
#    * Minimal shapelet length (specified as a fraction of TS length)
#    * Different scales of shapelet lengths
#    * Weight regularizer
#    * Number of iterations
hyper_parameters_lts = {
	'Adiac': 					[0.3,  0.2,   3, 0.01, 10000],
	'Beef': 					[0.15, 0.125, 3, 0.01, 10000],
	'BeetleFly': 				[0.15, 0.125, 1, 0.01, 5000],
	'BirdChicken': 				[0.3,  0.075, 1, 0.1,  10000],
	'ChlorineConcentration':    [0.3,  0.2,   3, 0.01, 10000],
	'Coffee': 					[0.05, 0.075, 2, 0.01, 5000],
	'DiatomSizeReduction': 		[0.3,  0.175, 2, 0.01, 10000],
	'ECGFiveDays': 				[0.05, 0.125, 2, 0.01, 10000],
	'FaceFour': 				[0.3,  0.175, 3, 1.0,  5000],
	'GunPoint': 				[0.15, 0.2,   3, 0.1,  10000],
	'ItalyPowerDemand':			[0.3,  0.2,   3, 0.01, 5000],
	'Lightning7': 				[0.05, 0.075, 3, 1,    5000],
	'MedicalImages': 			[0.3,  0.2,   2, 1,    10000],
	'MoteStrain': 				[0.3,  0.2,   3, 1,    10000],
	'SonyAIBORobotSurface1': 	[0.3,  0.125, 2, 0.01, 10000],
	'SonyAIBORobotSurface2': 	[0.3,  0.125, 2, 0.01, 10000],
	'Symbols': 					[0.05, 0.175, 1, 0.1,  5000],
	'SyntheticControl': 		[0.15, 0.125, 3, 0.01, 5000],
	'Trace': 					[0.15, 0.125, 2, 0.1,  10000],
	'TwoLeadECG': 				[0.3,  0.075, 1, 0.1,  10000]
}

metadata = sorted(load_data_train_test(), key=lambda x: x['train']['n_samples']**2*x['train']['n_features']**3)
result_vectors = []

for dataset in metadata:
    if dataset['train']['name'] not in hyper_parameters_lts: continue

    print(dataset['train']['name'])

    # Load the training and testing dataset (features + label vector)
    train_df = pd.read_csv(dataset['train']['data_path'])
    test_df = pd.read_csv(dataset['test']['data_path'])
    X_train = train_df.drop('target', axis=1).values
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1).values
    y_test = test_df['target']

    map_dict = {}
    for j, c in enumerate(np.unique(y_train)):
        map_dict[c] = j
    y_train = y_train.map(map_dict) 
    y_test = y_test.map(map_dict)

    print(set(y_train), set(y_test))

    y_train = y_train.values
    y_test = y_test.values

    nr_shap, l, r, reg, max_it = hyper_parameters_lts[dataset['train']['name']]

    fit_lts(X_train, y_train, X_test, y_test, nr_shap, l, r, reg, max_it,
            'results/lts_vs_genetic/{}_learned_shapelets_{}.txt'.format(dataset['train']['name'], int(time.time())), 
            'results/lts_vs_genetic/{}_learned_shapelets_predictions_{}.csv'.format(dataset['train']['name'], int(time.time())), 
            'results/lts_vs_genetic/{}_learned_runtime_{}.csv'.format(dataset['train']['name'], int(time.time()))
    )


    fit_genetic(X_train, y_train, X_test, y_test,  
            'results/lts_vs_genetic/{}_genetic_shapelets_{}.txt'.format(dataset['train']['name'], int(time.time())), 
            'results/lts_vs_genetic/{}_genetic_shapelets_predictions_{}.csv'.format(dataset['train']['name'], int(time.time())),
            'results/lts_vs_genetic/{}_genetic_runtime_{}.csv'.format(dataset['train']['name'], int(time.time()))
    )
