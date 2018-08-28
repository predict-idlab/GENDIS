# In this script, we assess the gain or loss in predictive performance when
# we let LTS extract a shapelet set similar to GENDIS. For this script, the
# lts_vs_gendis script has to be run first.

# Standard Library
import os
import glob
from collections import defaultdict, Counter
import time

# Data wrangling
import pandas as pd
import numpy as np

# Calculate accuracies
from sklearn.metrics import accuracy_score

# Load the datasets & LTS
from tslearn.datasets import UCR_UEA_datasets
from tslearn.shapelets import ShapeletModel

# Python regexes
import re

# Parsing string representations of python objects
import ast

# Scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def parse_shapelets(shapelets):
    shapelets = shapelets.replace(']', '],')[:-2]
    shapelets = re.sub(r'\s+', ', ', shapelets)
    shapelets = re.sub(r',+', ',', shapelets)
    shapelets = shapelets.replace('],[', '], [')
    shapelets = shapelets.replace('[,', '[')
    shapelets = '[' + shapelets + ']'
    shapelets = re.sub(r',\s+]', ']', shapelets)
    return ast.literal_eval(shapelets)


def fit_lr(X_distances_train, y_train, X_distances_test, y_test, out_path):
    lr = GridSearchCV(
            LogisticRegression(random_state=1337), 
            {
              'penalty': ['l1', 'l2'], 
              'C': [10**i for i in range(-2, 6)] + [5**i for i in range(-2, 6)],
              'class_weight': [None, 'balanced']
            }
        )

    lr.fit(X_distances_train, y_train)
    
    hard_preds = lr.predict(X_distances_test)
    proba_preds = lr.predict_proba(X_distances_test)

    print("[LR] Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[LR] Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_lr_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_lr_proba.csv')


def fit_lts(X_train, y_train, X_test, y_test, shap_dict, reg, max_it, shap_out_path, pred_out_path, timing_out_path):
    # Fit LTS model, print metrics on test-set, write away predictions and shapelets
    clf = ShapeletModel(n_shapelets_per_size=shap_dict, 
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

    with open(shap_out_path, 'w+') as ofp:
        for shap in clf.shapelets_:
            ofp.write(str(np.reshape(shap, (-1))) + '\n')

    with open(timing_out_path, 'w+') as ofp:
        ofp.write(str(learning_time))

    X_distances_train = clf.transform(X_train)
    X_distances_test = clf.transform(X_test)

    fit_lr(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)

hyper_parameters_lts = {
    #'Adiac':                     [0.3,  0.2,   3, 0.01, 10000],
    #'Beef':                     [0.15, 0.125, 3, 0.01, 10000],
    #'BeetleFly':                 [0.15, 0.125, 1, 0.01, 5000],
    #'BirdChicken':                 [0.3,  0.075, 1, 0.1,  10000],
    #'ChlorineConcentration':    [0.3,  0.2,   3, 0.01, 10000],
    #'Coffee':                     [0.05, 0.075, 2, 0.01, 5000],
    #'DiatomSizeReduction':         [0.3,  0.175, 2, 0.01, 10000],
    'ECGFiveDays':                 [0.05, 0.125, 2, 0.01, 10000],
    #'FaceFour':                 [0.3,  0.175, 3, 1.0,  5000],
    #'GunPoint':                 [0.15, 0.2,   3, 0.1,  10000],
    #'ItalyPowerDemand':            [0.3,  0.2,   3, 0.01, 5000],
    #'Lightning7':                 [0.05, 0.075, 3, 1,    5000],
    #'MedicalImages':             [0.3,  0.2,   2, 1,    10000],
    'MoteStrain':                 [0.3,  0.2,   3, 1,    10000],
    'SonyAIBORobotSurface1':     [0.3,  0.125, 2, 0.01, 10000],
    #'SonyAIBORobotSurface2':     [0.3,  0.125, 2, 0.01, 10000],
    'Symbols':                     [0.05, 0.175, 1, 0.1,  5000],
    #'SyntheticControl':         [0.15, 0.125, 3, 0.01, 5000],
    #'Trace':                     [0.15, 0.125, 2, 0.1,  10000],
    'TwoLeadECG':                 [0.3,  0.075, 1, 0.1,  10000]
}

DIR = 'results/dependent_vs_independent/'
datasets = set([x.split('_')[0] for x in os.listdir(DIR) 
                if x != '.keep'])
data_loader = UCR_UEA_datasets()
genetic_sizes = defaultdict(list)
for dataset in datasets:

    if dataset not in hyper_parameters_lts: continue

    # Load train and test set
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # Map labels to [0, .., C-1]
    map_dict = {}
    for j, c in enumerate(np.unique(y_train)):
        map_dict[c] = j
    y_train = pd.Series(y_train).map(map_dict).values
    y_test = pd.Series(y_test).map(map_dict).values

    # Get the best hyper-parameters for LTS (from original paper)
    nr_shap, l, r, reg, max_it = hyper_parameters_lts[dataset]

    # Iterate over the shapelet files from GENDIS and store the lengths 
    files = glob.glob('results/lts_vs_genetic/{}_genetic_shapelets*.txt'.format(dataset))
    if len(files):
        sizes = []
        for f in files:
            shaps = parse_shapelets(open(f, 'r').read())
            genetic_sizes[dataset].append(len(shaps))
            for s in shaps:
                sizes.append(len(s))
            
        shap_dict_cntr = Counter(np.random.choice(sizes, size=int(np.mean(genetic_sizes[dataset]))))
        shap_dict = {}
        for c in shap_dict_cntr:
            shap_dict[int(c)] = int(shap_dict_cntr[c])

    fit_lts(X_train, y_train, X_test, y_test, dict(shap_dict), reg, max_it,
        'results/lts_smaller/{}_learned_shapelets_{}.txt'.format(dataset, int(time.time())), 
        'results/lts_smaller/{}_learned_shapelets_predictions_{}.csv'.format(dataset, int(time.time())), 
        'results/lts_smaller/{}_learned_runtime_{}.csv'.format(dataset, int(time.time()))
    )