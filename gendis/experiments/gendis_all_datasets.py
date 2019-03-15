import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
import os

from genetic import GeneticExtractor

from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV

from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import DTYPE
from sklearn.ensemble.forest import ForestClassifier
from sklearn.utils import resample, gen_batches, check_random_state
from sklearn.utils.extmath import fast_dot
from sklearn.decomposition import PCA

from _exceptions import NotFittedError

def random_feature_subsets(array, batch_size, random_state=1234):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = range(array.shape[1])
    random_state.shuffle(list(features))
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


class RotationTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo

        super(RotationTreeClassifier, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            presort=presort)

    def rotate(self, X):
        if not hasattr(self, 'rotation_matrix'):
            raise NotFittedError('The estimator has not been fitted')

        return fast_dot(X, self.rotation_matrix)

    def pca_algorithm(self):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'randomized':
            return RandomizedPCA(random_state=self.random_state)
        elif self.rotation_algo == 'pca':
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, int(np.sqrt(len(X[0]))),
                                       random_state=self.random_state)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=int(n_samples*0.75),
                                random_state=10*i)
            pca = self.pca_algorithm()
            pca.fit(x_sample[:, subset])
            self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._fit_rotation_matrix(X)
        super(RotationTreeClassifier, self).fit(self.rotate(X), y,
                                                sample_weight, check_input)

    def predict_proba(self, X, check_input=True):
        return  super(RotationTreeClassifier, self).predict_proba(self.rotate(X),
                                                                  check_input)

    def predict(self, X, check_input=True):
        return super(RotationTreeClassifier, self).predict(self.rotate(X),
                                                           check_input)

    def apply(self, X, check_input=True):
        return super(RotationTreeClassifier, self).apply(self.rotate(X),
                                                         check_input)

    def decision_path(self, X, check_input=True):
        return super(RotationTreeClassifier, self).decision_path(self.rotate(X),
                                                                 check_input)

class RotationForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super(RotationForestClassifier, self).__init__(
            base_estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("n_features_per_subset", "rotation_algo",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

def fit_lr(X_distances_train, y_train, X_distances_test, y_test, out_path):
    lr = GridSearchCV(
            LogisticRegression(), 
            {
              'penalty': ['l1', 'l2'], 
              'C': [10**i for i in range(-2, 6)] + [5**i for i in range(-2, 6)],
              'class_weight': [None, 'balanced']
            }
        )
    lr.fit(X_distances_train, y_train)
    
    hard_preds = lr.predict(X_distances_test)
    proba_preds = lr.predict_proba(X_distances_test)
    
    hard_preds_train = lr.predict(X_distances_train)
    proba_preds_train = lr.predict_proba(X_distances_train)

    print("[LR] TRAIN Accuracy = {}".format(accuracy_score(y_train, hard_preds_train)))
    print("[LR] TRAIN Logloss = {}".format(log_loss(y_train, proba_preds_train)))
    print("[LR] TEST Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[LR] TEST Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_lr_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_lr_proba.csv')

def fit_svm(X_distances_train, y_train, X_distances_test, y_test, out_path):
    svc = GridSearchCV(
            SVC(probability=True, kernel='linear'), 
            {
              'C': [10**i for i in range(-2, 6)] + [5**i for i in range(-2, 6)]
            }
        )
    svc.fit(X_distances_train, y_train)
    
    hard_preds = svc.predict(X_distances_test)
    proba_preds = svc.predict_proba(X_distances_test)
    
    hard_preds_train = svc.predict(X_distances_train)
    proba_preds_train = svc.predict_proba(X_distances_train)

    print("[SVM] TRAIN Accuracy = {}".format(accuracy_score(y_train, hard_preds_train)))
    print("[SVM] TRAIN Logloss = {}".format(log_loss(y_train, proba_preds_train)))
    print("[SVM] TEST Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[SVM] TEST Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_svm_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_svm_proba.csv')

def fit_voting(X_distances_train, y_train, X_distances_test, y_test, out_path):
    svm_linear = Pipeline(steps=[('scale', MinMaxScaler()), ('svm', SVC(probability=True, kernel='linear'))])
    svm_quadratic = Pipeline(steps=[('scale', MinMaxScaler()), ('svm', SVC(probability=True, kernel='poly', degree=2))])
    rf = RandomForestClassifier(n_estimators=500)
    knn = GridSearchCV(KNeighborsClassifier(weights='distance', metric='euclidean'), {'n_neighbors': [1, 3, 5, min(7, len(X_train) // 5), min(13, len(X_train) // 5), min(25, len(X_train) // 5)]})
    rot = RotationForestClassifier(n_estimators=50)
    # We use logreg instead of naive bayes and bayesian networks
    logreg = GridSearchCV(LogisticRegression(), {'penalty': ['l1', 'l2']})
    # Apply some hyper-parameter tuning, since WEKA applies pruning (c4.5 algorithm)
    dt = GridSearchCV(DecisionTreeClassifier(), {'min_samples_leaf': [1, 3, 5, min(7, len(X_train) // 5), min(13, len(X_train) // 5), min(25, len(X_train) // 5)]})
    nb = ComplementNB(norm=True)

    models = [
        #('DecisionTree', dt),
        ('RotationForest', rot),
        ('LinearSVC', svm_linear),
        ('QuadraticSVC', svm_quadratic),
        ('RandomForest', rf),
        ('NearestNeighbors', knn),
        #('LogisticRegression', logreg),
    ]

    accuracies = []
    for name, clf in models:
        cv_acc = np.mean(cross_val_score(clf, X_distances_train, y_train, cv=10, scoring='accuracy'))
        accuracies.append(cv_acc)

    acc_sum = sum(accuracies)
    norm_accuracies = []
    for acc in accuracies:
        norm_accuracies.append(acc/acc_sum)

    voting = VotingClassifier(models, weights=norm_accuracies, voting='soft')
    voting.fit(X_distances_train, y_train)
    hard_preds = voting.predict(X_distances_test)
    proba_preds = voting.predict_proba(X_distances_test)
    hard_preds_train = voting.predict(X_distances_train)
    proba_preds_train = voting.predict_proba(X_distances_train)

    print("[Voting] TRAIN Accuracy = {}".format(accuracy_score(y_train, hard_preds_train)))
    print("[Voting] TRAIN Logloss = {}".format(log_loss(y_train, proba_preds_train)))
    print("[Voting] TEST Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[Voting] TEST Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_voting_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_voting_proba.csv')

def fit_rf(X_distances_train, y_train, X_distances_test, y_test, out_path):
    rf = GridSearchCV(
        RandomForestClassifier(), 
        {
            'n_estimators': [10, 25, 50, 100, 500], 
            'max_depth': [None, 3, 7, 15]
        }
    )
    rf.fit(X_distances_train, y_train)
    
    hard_preds = rf.predict(X_distances_test)
    proba_preds = rf.predict_proba(X_distances_test)
    
    hard_preds_train = rf.predict(X_distances_train)
    proba_preds_train = rf.predict_proba(X_distances_train)

    print("[RF] TRAIN Accuracy = {}".format(accuracy_score(y_train, hard_preds_train)))
    print("[RF] TRAIN Logloss = {}".format(log_loss(y_train, proba_preds_train)))
    print("[RF] TEST Accuracy = {}".format(accuracy_score(y_test, hard_preds)))
    print("[RF] TEST Logloss = {}".format(log_loss(y_test, proba_preds)))

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    proba_preds = pd.DataFrame(proba_preds, columns=['proba_{}'.format(x) for x in set(list(y_train) + list(y_test))])

    hard_preds.to_csv(out_path.split('.')[0]+'_rf_hard.csv')
    proba_preds.to_csv(out_path.split('.')[0]+'_rf_proba.csv')

def gendis_discovery(X_train, y_train, X_test, y_test, shap_out_path, pred_out_path, timing_out_path):
    pipeline = Pipeline([
        ('extractor', GeneticExtractor(verbose=False, population_size=10, iterations=25, wait=5, plot=None)),
        ('classifier', LogisticRegression())
    ])

    # For some datasets, the number of samples for a class is lower than 3...
    min_samples = Counter(y_train).most_common()[-1][1]
    n_folds = min(min_samples, 3)

    ts_len = X_train.shape[1]
    grid_search = GridSearchCV(pipeline, {'extractor__max_len': [ts_len // 4, ts_len // 2, 3 * ts_len // 4, ts_len]}, cv=n_folds, scoring='neg_log_loss')
    grid_search.fit(X_train, y_train)
    best_length = grid_search.best_params_['extractor__max_len']

    print(best_length, X_train.shape[1])

    genetic_extractor = GeneticExtractor(verbose=True, population_size=100, iterations=100, wait=10, plot=None, max_len=best_length)
    start = time.time()
    genetic_extractor.fit(X_train, y_train)
    genetic_time = time.time() - start

    print('Genetic shapelet discovery took {}s'.format(genetic_time))

    with open(shap_out_path, 'w+') as ofp:
        for shap in genetic_extractor.shapelets:
            ofp.write(str(np.reshape(shap, (-1))) + '\n')

    with open(timing_out_path, 'w+') as ofp:
        ofp.write(str(genetic_time))

    X_distances_train = genetic_extractor.transform(X_train)
    X_distances_test = genetic_extractor.transform(X_test)

    #fit_lr(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    #fit_rf(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    #fit_svm(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    fit_voting(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)

data_loader = UCR_UEA_datasets()

datasets = ['GestureMidAirD3', 'GestureMidAirD2', 
            'GestureMidAirD1', 'Symbols', 'HandMovementDirection', 'Heartbeat', 'Yoga', 'OSULeaf', 'Ham', 'Meat', 'Fish', 'Beef', 'ShapeletSim', 
            'FordB', 'FordA', 'ShapesAll', 'Herring', 'Earthquakes', 'BirdChicken', 'BeetleFly', 'OliveOil', 'Car', 'InsectEPGSmallTrain', 
            'InsectEPGRegularTrain', 'Lightning2', 'AtrialFibrilation', 'SmallKitchenAppliances']

done = ['ShakeGestureWiimoteZ', 'PLAID', 'PickupGestureWiimoteZ', 'GesturePebbleZ2', 'GesturePebbleZ1', 'AllGestureWiimoteZ', 
        'AllGestureWiimoteY', 'AllGestureWiimoteX', 'PenDigits', 'SmoothSubspace', 'MelbournePedestrian', 'ItalyPowerDemand', 
        'Chinatown', 'JapaneseVowels', 'RacketSports', 'InsectWingbeat', 'LSST', 'Libras', 'Crop', 'FingerMovements', 'NATOPS', 
        'SyntheticControl', 'FaceDetection', 'SonyAIBORobotSurface2', 'Ering', 'SonyAIBORobotSurface1', 'ProximalPhalanxTW', 
        'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxOutlineAgeGroup', 'PhalangesOutlinesCorrect', 'MiddlePhalanxTW', 
        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'DistalPhalanxOutlineCorrect', 
        'DistalPhalanxOutlineAgeGroup', 'TwoLeadECG', 'MoteStrain', 'SpokenArabicDigits', 'ECG200', 'MedicalImages', 
        'BasicMotions', 'TwoPatterns', 'SwedishLeaf', 'CBF', 'BME', 'FacesUCR', 'FaceAll', 'ECGFiveDays', 'ECG5000', 'PowerCons', 
        'Plane', 'PEMS', 'ArticularyWordRecognition', 'UMD', 'GunPointOldVersusYoung', 'GunPointMaleVersusFemale', 'GunPointAgeSpan', 
        'GunPoint', 'Wafer', 'Handwriting', 'ChlorineConcentration', 'Adiac', 'CharacterTrajectories', 'Fungi', 'Epilepsy', 'Phoneme', 
        'Wine', 'Strawberry', 'ArrowHead', 'InsectWingbeatSound', 'WordSynonyms', 'FiftyWords', 'DuckDuckGeese', 'Trace', 
        'ToeSegmentation1', 'Coffee', 'DodgerLoopWeekend', 'DodgerLoopGame', 'DodgerLoopDay', 'CricketZ', 'CricketY', 'CricketX', 
        'FreezerSmallTrain', 'FreezerRegularTrain', 'UWaveGestureLibraryZ', 'UWaveGestureLibraryY', 'UWaveGestureLibraryX', 
        'UWaveGestureLibrary', 'Lightning7', 'ToeSegmentation2', 'DiatomSizeReduction', 'FaceFour']

if not os.path.isdir('results/genetic'): 
    os.makedirs('results/genetic')

for dataset in datasets:
    try:
        X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)
        print(sorted(data_loader.baseline_accuracy(dataset)[dataset].items(), key=lambda x: -x[1]))

        # Re-sample the test and train set with same sizes and strataified
        if X_test is None or len(X_test) == 0: continue
        nr_test_samples = len(X_test)
        X = np.vstack((X_train, X_test))
        y = np.vstack((np.reshape(y_train, (-1, 1)), np.reshape(y_test, (-1, 1))))
        y = pd.Series(np.reshape(y, (-1,)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=nr_test_samples)
        test_idx = y_test.index
        train_idx = y_train.index

        scaler = TimeSeriesScalerMeanVariance()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

        # Map labels to [0, .., C-1]
        map_dict = {}
        for j, c in enumerate(sorted(set(y_train))):
            map_dict[c] = j
        y_train = y_train.map(map_dict).values
        y_test = y_test.map(map_dict).values

        timestamp = int(time.time())

        pd.DataFrame(np.reshape(y_test, (-1, 1)), index=test_idx, columns=['label']).to_csv('results/lts_vs_genetic/{}_ground_truth_test_{}.csv'.format(dataset, timestamp))
        pd.DataFrame(np.reshape(y_train, (-1, 1)), index=train_idx, columns=['label']).to_csv('results/lts_vs_genetic/{}_ground_truth_train_{}.csv'.format(dataset, timestamp))

        gendis_discovery(X_train, y_train, X_test, y_test,  
                'results/lts_vs_genetic/{}_genetic_shapelets_{}.txt'.format(dataset, timestamp), 
                'results/lts_vs_genetic/{}_genetic_shapelets_predictions_{}.csv'.format(dataset, timestamp),
                'results/lts_vs_genetic/{}_genetic_runtime_{}.csv'.format(dataset, timestamp)
        )
        print(sorted(data_loader.baseline_accuracy(dataset)[dataset].items(), key=lambda x: -x[1]))
    except KeyError as e:
        print('Dataset {} failed...'.format(dataset))
    except Exception as e:
        raise
