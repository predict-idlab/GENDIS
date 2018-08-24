from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from collections import Counter
import operator
import sys
import os
sys.path.append('..')
sys.path.append('../other')
import util
from sax import SAXExtractor
import other_util
from tslearn.datasets import UCR_UEA_datasets
import warnings; warnings.filterwarnings('ignore')


class ShapeletTransformer(BaseEstimator, TransformerMixin):
    """ An example transformer that returns the element-wise square root..
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    input_shape : tuple
        The shape the data passed to :meth:`fit`
    """
    def __init__(self, method=None, min_len=None, max_len=None, 
                 nr_shapelets=1, metric='ig'):
        self.extractor = method
            
        self.shapelets = []
        self.min_len = min_len
        self.max_len = max_len
        self.nr_shapelets = nr_shapelets
        self.metric = metric

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)

        self.input_shape_ = X.shape

        self.shapelets = self.extractor.extract(
            X, y, 
            min_len=self.min_len, 
            max_len=self.max_len, 
            nr_shapelets=self.nr_shapelets, 
            metric=self.metric
        )

        # Return the transformer
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_transformed : array of int of shape = [n_samples, n_features]
            The array containing the element-wise square roots of the values
            in `X`
        """
        # Check is fit had been called
        check_is_fitted(self, ['shapelets'])

        # Input validation
        X = check_array(X)

        feature_vectors = np.zeros((len(X), len(self.shapelets)))
        for smpl_idx, sample in enumerate(X):
            for shap_idx, shapelet in enumerate(self.shapelets):
                feature_vectors[smpl_idx, shap_idx] = util.sdist_no_norm(shapelet.flatten(), sample)

        return feature_vectors


class ShapeletTree(object):
    def __init__(self, right=None, left=None, shapelet=None, threshold=None, class_probabilities={}):
        self.right = right
        self.left = left
        self.shapelet = shapelet
        self.threshold = threshold
        self.class_probabilities = class_probabilities

    def evaluate(self, time_serie, proba=True):
        if self.is_leaf():
            if proba:
                return self.class_probabilities
            else:
                return max(self.class_probabilities.items(), key=operator.itemgetter(1))[0]
        else:
            dist = util.sdist(self.shapelet, time_serie)
            if dist <= self.threshold:
                return self.left.evaluate(time_serie, proba=proba)
            else:
                return self.right.evaluate(time_serie, proba=proba)

    def predict(self, X):
        return [ self.evaluate(ts, proba=False) for ts in X ]

    def predict_proba(self, X):
        return [ self.evaluate(ts, proba=True) for ts in X ]

    def is_leaf(self):
        return self.threshold is None

    def extract_all_shapelets(self):
        if self.is_leaf():
            return None
        else:
            left_shap = self.left.extract_all_shapelets()
            right_shap = self.right.extract_all_shapelets()
            all_shapelets = [self.shapelet]
            if left_shap is not None:
                all_shapelets += left_shap
            if right_shap is not None:
                all_shapelets += right_shap
            return all_shapelets


class ShapeletTreeClassifier(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, method=None, max_depth=None, min_samples_split=1, min_len=None, max_len=None, metric='ig'):
        self.extractor = method

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.metric = metric
        self.min_len = min_len
        self.max_len = max_len

        self.tree = None

    def _calc_probs(self, y):
        probs = Counter(y)
        total = sum(probs.values())
        for k in probs: probs[k] /= total
        return probs


    def _extract_tree(self, X, y, depth=0):
        if (self.max_depth is None or depth > self.max_depth) and len(np.unique(y)) > 1:
            # Extract 1 shapelet, using the specified `extractor`
            map_dict = {}
            for j, c in enumerate(np.unique(y)):
                map_dict[c] = j
            y_mapped = np.vectorize(map_dict.get)(y)

            shapelet = self.extractor.extract(
                X, y_mapped, 
                min_len=self.min_len, 
                max_len=self.max_len, 
                nr_shapelets=1, 
                metric=self.metric
            )[0]

            # Get the best threshold distance for this shapelet
            L = []
            X_left, y_left, X_right, y_right = [], [], [], []
            for k in range(len(X)):
                D = X[k, :]
                dist = util.sdist(shapelet, D)
                L.append((dist, y[k]))
            threshold = other_util.get_threshold(L)

            # Create a new internal node
            node = ShapeletTree(right=None, left=None, shapelet=shapelet, threshold=threshold, class_probabilities=self._calc_probs(y))

            # Partition the data
            X_left, y_left, X_right, y_right = [], [], [], []
            for ts, label in zip(X, y):
                if util.sdist(shapelet, ts) <= threshold:
                    X_left.append(ts)
                    y_left.append(label)
                else:
                    X_right.append(ts)
                    y_right.append(label)

            X_left = np.array(X_left)
            y_left = np.array(y_left)
            X_right = np.array(X_right)
            y_right = np.array(y_right)

            # Recursive call to create the left and right child of the internal node
            if len(X_left) >= self.min_samples_split:
                node.left = self._extract_tree(X_left, y_left, depth=depth+1)
            else:
                 node.left = ShapeletTree(right=None, left=None, shapelet=None, threshold=None, class_probabilities=self._calc_probs(y_left))
            if len(X_right) >= self.min_samples_split:
                node.right = self._extract_tree(X_right, y_right, depth=depth+1)
            else:
                 node.right = ShapeletTree(right=None, left=None, shapelet=None, threshold=None, class_probabilities=self._calc_probs(y_right))
            return node

        else:
            return ShapeletTree(right=None, left=None, shapelet=None, threshold=None, class_probabilities=self._calc_probs(y))

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.tree = self._extract_tree(X, y)
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['tree'])

        # Input validation
        X = check_array(X)

        return self.tree.predict(X)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        return self.tree.predict_proba(X)


def estimate_min_max(X, y, extractor, min_perc=25, max_perc=75, min_len=3, 
                     max_len=None, iterations=10):
    shapelet_lengths = []
    for _ in range(iterations):
        rand_idx = np.random.choice(range(len(X)), size=10, replace=False)
        X_sub = X[rand_idx, :]
        y_sub = y[rand_idx]
        shapelet_lengths += [len(x) for x in extractor.extract(X_sub, y_sub, 
                                                               min_len=min_len, 
                                                               max_len=max_len, 
                                                               nr_shapelets=10)]
        
    _min = int(np.percentile(shapelet_lengths, min_perc))
    _max = max(_min + 1, int(np.percentile(shapelet_lengths, max_perc)))
    return _min, _max

def extract_shapelets_with_tree(X_train, y_train, extractor, min_len, max_len):
    shap_tree_clf = ShapeletTreeClassifier(method=extractor, min_len=min_len,
                                           max_len=max_len, metric=other_util.calculate_ig)
    shap_tree_clf.fit(X_train, y_train)
    return shap_tree_clf.tree.extract_all_shapelets()

datasets = [
    "ItalyPowerDemand",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "MoteStrain",
    "TwoLeadECG",
    "ECGFiveDays",
    #"CBF",
    #"GunPoint",
    #"ECG200",
    #"DiatomSizeReduction",
    #"Coffee",
    #"SyntheticControl",
    #"ArrowHead",
    #"FaceFour",
    #"Plane",
    #"ToeSegmentation1",
    #"Symbols",
    #"Wine",
    #"ShapeletSim",
    #"ToeSegmentation2",
    #"BeetleFly",
    #"BirdChicken",
    #"MiddlePhalanxTW",
    #"DistalPhalanxTW",
    #"MiddlePhalanxOutlineAgeGroup",
    #"ProximalPhalanxOutlineAgeGroup",
    #"DistalPhalanxOutlineAgeGroup",
    #"ProximalPhalanxTW",
    #"FacesUCR",
    #"Beef",
    #"MedicalImages",
    #"Lightning7",
    #"OliveOil",
    #"ProximalPhalanxOutlineCorrect",
    #"DistalPhalanxOutlineCorrect",
    #"MiddlePhalanxOutlineCorrect",
    #"Trace",
    #"Meat",
    #"SwedishLeaf",
    #"Herring",
    #"ECG5000",
    #"Car",
    #"FaceAll",
    #"InsectWingbeatSound",
    #"Adiac",
    #"Lightning2",
    #"Ham",
    #"ChlorineConcentration",
    #"WordSynonyms",
    #"PhalangesOutlinesCorrect",
    #"TwoPatterns",
    #"Fish",
    #"OSULeaf",
    #"Mallat",
    #"Wafer",
    #"FiftyWords",
    #"CricketZ",
    #"CricketX",
    #"CricketY",
    #"Strawberry",
    #"Yoga",
    #"CinCECGtorso",
    #"Earthquakes",
    #"Computers",
    #"WormsTwoClass",
    #"Worms",
    #"UWaveGestureLibraryZ",
    #"UWaveGestureLibraryY",
    #"UWaveGestureLibraryX",
    #"Haptics",
    #"ShapesAll",
    #"Phoneme",
    #"LargeKitchenAppliances",
    #"ScreenType",
    #"SmallKitchenAppliances",
    #"RefrigerationDevices",
    #"AALTDChallenge",
    #"InlineSkate",
    #"ElectricDevices",
    #"UWaveGestureLibraryAll",
    #"StarLightCurves",
    #"NonInvasiveFatalECGThorax1",
    #"NonInvasiveFetalECGThorax1",
    #"NonInvasiveFatalECGThorax2",
    #"NonInvasiveFetalECGThorax2",
    #"FordA",
    #"FordB",
    #"HandOutlines"
]

data_loader = UCR_UEA_datasets()

for dataset in datasets:
    print(dataset)
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))

    # Map labels to [0, .., C-1]
    map_dict = {}
    for j, c in enumerate(np.unique(y_train)):
        map_dict[c] = j
    y_train = pd.Series(y_train).map(map_dict).values
    y_test = pd.Series(y_test).map(map_dict).values

    # Create the extractor & determine min and max shapelet length heuristically
    extractor = SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100, 
                              iterations=5, mask_size=3)
    _min, _max = estimate_min_max(X_train, y_train, extractor)

    # Extract shapelets using Shapelet Tree and fit RF
    tree_shapelets = extract_shapelets_with_tree(X_train, y_train, extractor,
                                                 _min, _max)
    shap_transformer = ShapeletTransformer(method=extractor, min_len=_min, 
                                           max_len=_max, metric=other_util.calculate_ig)
    shap_transformer.shapelets = [np.array(x) for x in tree_shapelets]

    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = GridSearchCV(
      RandomForestClassifier(random_state=1337),
      {'n_estimators': [5, 10, 50, 100, 250, 500]}
    )
    rf.fit(X_distances_train, y_train)

    # Write away the predictions + a plot from all shapelets
    rf_preds = pd.DataFrame(rf.predict_proba(X_distances_test))
    rf_preds.to_csv('results/dependent_vs_independent/{}_rf_preds_tree.csv'.format(dataset['train']['name']))
    with open('results/dependent_vs_independent/{}_shaps_tree.txt'.format(dataset['train']['name']), 'w') as ofp:
        for shapelet in tree_shapelets:
            ofp.write(str(shapelet)+'\n')

    # Do the same, but extract the shapelets in a single pass (features//2 shapelets)
    shap_transformer = ShapeletTransformer(method=extractor, min_len=_min, metric=other_util.calculate_ig,
                                           max_len=_max, nr_shapelets=dataset['train']['n_features']//2)
    shap_transformer.fit(X_train, y_train)
    shap_transformer.shapelets = [np.array(x) for x in shap_transformer.shapelets]
    X_distances_train = shap_transformer.transform(X_train)
    X_distances_test = shap_transformer.transform(X_test)

    rf = GridSearchCV(
      RandomForestClassifier(random_state=1337),
      {'n_estimators': [5, 10, 50, 100, 250, 500]}
    )
    rf.fit(X_distances_train, y_train)

    rf_preds = pd.DataFrame(rf.predict_proba(X_distances_test))
    rf_preds.to_csv('results/dependent_vs_independent/{}_rf_preds_transform.csv'.format(dataset['train']['name']))
    with open('results/dependent_vs_independent/{}_shaps_transform.txt'.format(dataset['train']['name']), 'w') as ofp:
        for shapelet in shap_transformer.shapelets:
            ofp.write(str(shapelet)+'\n')