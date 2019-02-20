import time
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from genetic import GeneticExtractor

from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# TODO: We need to tune the max_len parameter

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
        ('extractor', GeneticExtractor(verbose=False, population_size=10, iterations=25, wait=10, plot=None)),
        ('classifier', LogisticRegression())
    ])

    ts_len = X_train.shape[1]
    grid_search = GridSearchCV(pipeline, {'extractor__max_len': [ts_len // 4, ts_len // 2, 3 * ts_len // 4, ts_len]}, cv=3, scoring='neg_log_loss')
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

    fit_lr(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    fit_rf(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)
    fit_svm(X_distances_train, y_train, X_distances_test, y_test, pred_out_path)

data_loader = UCR_UEA_datasets()

datasets = ['ShakeGestureWiimoteZ', 'PLAID', 'PickupGestureWiimoteZ', 'GesturePebbleZ2', 'GesturePebbleZ1', 'AllGestureWiimoteZ', 
            'AllGestureWiimoteY', 'AllGestureWiimoteX', 'PenDigits', 'SmoothSubspace', 'MelbournePedestrian', 'ItalyPowerDemand', 
            'Chinatown', 'JapaneseVowels', 'RacketSports', 'InsectWingbeat', 'LSST', 'Libras', 'Crop', 'FingerMovements', 'NATOPS', 
            'SyntheticControl', 'FaceDetection', 'SonyAIBORobotSurface2', 'Ering', 'SonyAIBORobotSurface1', 'ProximalPhalanxTW', 
            'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxOutlineAgeGroup', 'PhalangesOutlinesCorrect', 'MiddlePhalanxTW', 
            'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'DistalPhalanxOutlineCorrect', 
            'DistalPhalanxOutlineAgeGroup', 'TwoLeadECG', 'MoteStrain', 'SpokenArabicDigits', 'ElectricDevices', 'ECG200', 'MedicalImages', 
            'BasicMotions', 'TwoPatterns', 'SwedishLeaf', 'CBF', 'BME', 'FacesUCR', 'FaceAll', 'ECGFiveDays', 'ECG5000', 'PowerCons', 
            'Plane', 'PEMS', 'ArticularyWordRecognition', 'UMD', 'GunPointOldVersusYoung', 'GunPointMaleVersusFemale', 'GunPointAgeSpan', 
            'GunPoint', 'Wafer', 'Handwriting', 'ChlorineConcentration', 'Adiac', 'CharacterTrajectories', 'Fungi', 'Epilepsy', 'Phoneme', 
            'Wine', 'Strawberry', 'ArrowHead', 'InsectWingbeatSound', 'WordSynonyms', 'FiftyWords', 'DuckDuckGeese', 'Trace', 
            'ToeSegmentation1', 'Coffee', 'DodgerLoopWeekend', 'DodgerLoopGame', 'DodgerLoopDay', 'CricketZ', 'CricketY', 'CricketX', 
            'FreezerSmallTrain', 'FreezerRegularTrain', 'UWaveGestureLibraryZ', 'UWaveGestureLibraryY', 'UWaveGestureLibraryX', 
            'UWaveGestureLibrary', 'Lightning7', 'ToeSegmentation2', 'DiatomSizeReduction', 'FaceFour', 'GestureMidAirD3', 'GestureMidAirD2', 
            'GestureMidAirD1', 'Symbols', 'HandMovementDirection', 'Heartbeat', 'Yoga', 'OSULeaf', 'Ham', 'Meat', 'Fish', 'Beef', 'ShapeletSim', 
            'FordB', 'FordA', 'ShapesAll', 'Herring', 'Earthquakes', 'BirdChicken', 'BeetleFly', 'OliveOil', 'Car', 'InsectEPGSmallTrain', 
            'InsectEPGRegularTrain', 'Lightning2', 'AtrialFibrilation', 'SmallKitchenAppliances', 'ScreenType', 'RefrigerationDevices', 
            'LargeKitchenAppliances', 'Computers', 'NonInvasiveFetalECGThorax2', 'NonInvasiveFetalECGThorax1', 'SelfRegulationSCP1', 'WormsTwoClass', 
            'Worms', 'UWaveGestureLibraryAll', 'StarlightCurves', 'Phoneme', 'MixedShapesSmallTrain', 'MixedShapes', 'Mallat', 'Haptics', 
            'SelfRegulationSCP2', 'Cricket', 'EOGVerticalSignal', 'EOGHorizontalSignal', 'ACSF1', 'SemgHandSubjectCh2', 'SemgHandMovementCh2', 
            'SemgHandGenderCh2', 'CinCECGtorso', 'EthanolLevel', 'EthanolConcentration', 'InlineSkate', 'PigCVP', 'PigArtPressure', 'PigAirwayPressure', 
            'StandWalkJump', 'HandOutlines', 'Rock', 'MotorImagery', 'HouseTwenty', 'EigenWorms']

for dataset in datasets:
    try:
        X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)
        print(sorted(data_loader.baseline_accuracy(dataset)[dataset].items(), key=lambda x: -x[1]))

        # TODO: Concatenate X and y's and re-split them (stratified)
        if X_test is None or len(X_test) == 0: continue
        nr_test_samples = len(X_test)
        X = np.vstack((X_train, X_test))
        y = np.vstack((np.reshape(y_train, (-1, 1)), np.reshape(y_test, (-1, 1))))
        y = np.reshape(y, (-1,))
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=nr_test_samples)

        scaler = TimeSeriesScalerMeanVariance()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

        # Map labels to [0, .., C-1]
        map_dict = {}
        for j, c in enumerate(np.unique(y_train)):
            map_dict[c] = j
        y_train = pd.Series(y_train).map(map_dict).values
        y_test = pd.Series(y_test).map(map_dict).values

        timestamp = int(time.time())

        pd.DataFrame(y_test, columns=['label']).to_csv('results/lts_vs_genetic/{}_ground_truth_test_{}.csv'.format(dataset, timestamp))
        pd.DataFrame(y_train, columns=['label']).to_csv('results/lts_vs_genetic/{}_ground_truth_train_{}.csv'.format(dataset, timestamp))

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
