# Standard Library
import os
from collections import defaultdict

# Data wrangling
import pandas as pd
import numpy as np

# Calculate accuracies
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the datasets
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Paired Wilcoxon Test
from scipy.stats import mannwhitneyu, rankdata, ttest_ind
from terminaltables import AsciiTable

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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

from catboost import CatBoostClassifier


import sys
sys.path.append('..')
sys.path.append('.')
from genetic import GeneticExtractor


import glob
import ast
import re

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import DTYPE
from sklearn.ensemble.forest import ForestClassifier
from sklearn.utils import resample, gen_batches, check_random_state
from sklearn.utils.extmath import safe_sparse_dot as fast_dot
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

    def pca_algorithm(self, n_components=None):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'randomized':
            return PCA(random_state=self.random_state, n_components=n_components)
        elif self.rotation_algo == 'pca':
            return PCA(svd_solver='randomized', n_components=n_components)
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, min(int(np.sqrt(len(X[0]))), len(X)),
                                       random_state=self.random_state)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=n_samples,
                                random_state=10*i, replace=True)
            pca = self.pca_algorithm(n_components=len(subset))
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

# Comment this out if you want to process results of dependent vs independent
DIR = 'results/genetic/'
method1 = 'genetic'
method1_name = 'GENDIS'

accuracies1 = defaultdict(list)
st_accuracies = pd.read_csv("AllSplits/ST.csv", index_col=0, header=None)
ls_accuracies = pd.read_csv("AllSplits/LS.csv", index_col=0, header=None)

data_loader = UCR_UEA_datasets()

def construct_voting_clf(X_train, X_test, y_train, y_test, shapelets, out_path):
    genetic_extractor = GeneticExtractor()
    genetic_extractor.shapelets = shapelets

    gendis_train_distances = genetic_extractor.transform(X_train)
    gendis_test_distances = genetic_extractor.transform(X_test)

    svm_linear = Pipeline(steps=[
        ('scale', MinMaxScaler()), 
        ('svm', GridSearchCV(
            SVC(probability=True, kernel='linear', decision_function_shape='ovo'),
            {'C': [10**i for i in range(-2, 6)] + [5**i for i in range(-2, 6)]},
            cv=2
        ))
    ])
    svm_quadratic = Pipeline(steps=[('scale', MinMaxScaler()), ('svm', SVC(probability=True, kernel='poly', degree=2))])
    nb = GaussianNB()
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=100),
        {'criterion': ['gini', 'entropy'], 'max_features': ['log2', None, 'sqrt']},
        cv=2
    )
    dt = DecisionTreeClassifier()
    knn = Pipeline(steps=[
        ('scale', MinMaxScaler()), 
        ('knn', GridSearchCV(
            KNeighborsClassifier(weights='distance', metric='euclidean'), 
            {'n_neighbors': [1, min(3, len(X_train) // 10), min(5, len(X_train) // 10), min(7, len(X_train) // 10), min(13, len(X_train) // 10), min(25, len(X_train) // 10)]},
            cv=2
        ))])
    rot = RotationForestClassifier(n_estimators=50)
    logreg = GridSearchCV(LogisticRegression(), 
                          {'penalty': ['l1', 'l2'], 'C':  [10**i for i in range(-2, 6)] + [5**i for i in range(-2, 6)]},
                          cv=2)

    models = [
        ('LinearSVC', svm_linear),
        ('RotationForest', rot),
        ('RandomForest', rf),
        ('NearestNeighbors', knn),
        ('LogisticRegression', logreg)
    ]

    final_models = []

    accuracies = []
    for name, clf in models:
        cv_acc = np.mean(cross_val_score(clf, gendis_train_distances, y_train, cv=2, scoring='accuracy'))
        clf.fit(gendis_train_distances, y_train)
        print(name, cv_acc, accuracy_score(y_test, clf.predict(gendis_test_distances)))
        accuracies.append(cv_acc)

    print(accuracies)

    voting = VotingClassifier(models, weights=accuracies, voting='hard')
    voting.fit(gendis_train_distances, y_train)
    hard_preds = voting.predict(gendis_test_distances)

    hard_preds = pd.DataFrame(hard_preds, columns=['prediction'])
    hard_preds.to_csv(out_path.split('.')[0]+'_voting_tuned_hard.csv')

    return accuracy_score(y_test, hard_preds)

datasets = set([x.split('_')[0] for x in os.listdir(DIR) if x != '.keep'])

table_data = [['Dataset', '#Classes', 'TS Length', '#Train', '#Test', 'Accuracy GENDIS Voting', 'Accuracy ST', 'Significance vs ST (MWU)', 'Difference']]
for dataset in datasets:

    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

    if X_train is None: continue

    X = np.vstack((X_train, X_test))
    y = np.vstack((np.reshape(y_train, (-1, 1)), np.reshape(y_test, (-1, 1))))

    glob_path = DIR + '{}_{}*{}_proba.csv'
    method1_files = glob.glob(glob_path.format(dataset, 'genetic', 'voting'))

    for file in method1_files:

        timestamp = file.split('_')[-3]
        preds = np.argmax(pd.read_csv(file, index_col=[0]).values, axis=1)
        ground_truth = pd.read_csv(DIR + '{}_ground_truth_test_{}.csv'.format(dataset, timestamp), index_col=0)
        print('-'*50)
        print(file)
        print('Current accuracy:', dataset, accuracy_score(ground_truth['label'], preds))

        test_idx = ground_truth.index
        train_idx = list(set(range(len(X))) - set(test_idx))

        ground_truth = ground_truth['label']

        map_dict = {}
        for j, c in enumerate(sorted(set(ground_truth))):
            map_dict[c] = j
        ground_truth = pd.Series(ground_truth).map(map_dict).values

        X_train_ = X[train_idx, :]
        X_test_ = X[test_idx, :]
        y_train_ = y[train_idx]
        y_test_ = y[test_idx]

        scaler = TimeSeriesScalerMeanVariance()
        X_train_ = scaler.fit_transform(X_train_)
        X_test_ = scaler.fit_transform(X_test_)

        X_train_ = np.reshape(X_train_, (X_train_.shape[0], X_train_.shape[1]))
        X_test_ = np.reshape(X_test_, (X_test_.shape[0], X_test_.shape[1]))

        # Map labels to [0, .., C-1]
        map_dict = {}
        for j, c in enumerate(np.unique(y_train_)):
            map_dict[c] = j
        y_train_ = pd.Series(y_train_.flatten()).map(map_dict).values
        y_test_ = pd.Series(y_test_.flatten()).map(map_dict).values
        
        shapelet_file = DIR + '{}_{}_shapelets_{}.txt'.format(dataset, 'genetic', timestamp)
        shapelets = []
        with open(shapelet_file, 'r') as ifp:
            for line in ifp.read().split(']\n'):
                if len(line):
                    proc_line = re.sub(r'\s+', ',', line)[:-1] + ']'
                    proc_line = proc_line.replace('[,', '[')
                    shapelets.append(np.array(ast.literal_eval(proc_line)))
        
        voting_acc = construct_voting_clf(X_train_, X_test_, y_train_, y_test_, shapelets, 
                                          'results/lts_vs_genetic/{}_genetic_shapelets_predictions_{}.csv'.format(dataset, timestamp))
        print(dataset, voting_acc, np.mean(st_accuracies.loc[dataset]))
        accuracies1[dataset].append(voting_acc)

    if len(accuracies1[dataset]) == 0: continue
    gendis_accuracy = np.mean(accuracies1[dataset])
    ls_accuracy = np.mean(ls_accuracies.loc[dataset])
    st_accuracy = np.mean(st_accuracies.loc[dataset])
    T, p = mannwhitneyu(accuracies1[dataset], st_accuracies.loc[dataset])
    if p > 0.05:
        st_mwu_significance = '\\'
    else:
        if gendis_accuracy > st_accuracy:
            st_mwu_significance = '+'
        else:
            st_mwu_significance = '-'

    T, p = mannwhitneyu(accuracies1[dataset], ls_accuracies.loc[dataset])
    if p > 0.05:
        ls_mwu_significance = '\\'
    else:
        if gendis_accuracy > ls_accuracy:
            ls_mwu_significance = '+'
        else:
            ls_mwu_significance = '-'

    table_data.append([dataset, len(set(ground_truth)), len(X_train[0]), len(X_train), len(X_test), gendis_accuracy, st_accuracy, st_mwu_significance, gendis_accuracy - st_accuracy])

    table_head = table_data[0]
    table_data = table_data[1:]
    table_data = sorted(table_data, key=lambda x: x[-1])
    table_data = [table_head] + table_data

    result_table = AsciiTable(table_data, 'GENDIS vs other shapelet methods')
    print(result_table.table)