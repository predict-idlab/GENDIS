import sys
sys.path.append('..')
from tslearn.generators import random_walk_blobs
import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

try:
	from genetic import GeneticExtractor
except:
	from gendis.genetic import GeneticExtractor

try:
	from pairwise_dist import _pdist
except:
	from gendis.pairwise_dist import _pdist

def f1_fitness(X, y, shapelets, verbose=False, cache=None):
    """Calculate the fitness of an individual/shapelet set"""
    D = np.zeros((len(X), len(shapelets)))

    # First check if we already calculated distances for a shapelet
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache_val = cache.get(shap_hash)
        if cache_val is not None:
            D[:, shap_ix] = cache_val

    # Fill up the 0 entries
    _pdist(X, [shap.flatten() for shap in shapelets], D)

    # Fill up our cache
    for shap_ix, shap in enumerate(shapelets):
        shap_hash = hash(tuple(shap.flatten()))
        cache.set(shap_hash, D[:, shap_ix])

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    lr.fit(D, y)
    preds = lr.predict(D)
    cv_score = f1_score(y, preds, average='micro')

    return (cv_score, sum([len(x) for x in shapelets]))

def test_f1_fitness():
	X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
	X = np.reshape(X, (X.shape[0], X.shape[1]))
	extractor = GeneticExtractor(iterations=5, n_jobs=1, population_size=10, fitness=f1_fitness)
	extractor.fit(X, y)