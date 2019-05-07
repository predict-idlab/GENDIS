from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import numpy as np

try:
    from gendis.pairwise_dist import _pdist
except:
    from pairwise_dist import _pdist

def logloss_fitness(X, y, shapelets, cache=None, verbose=False):
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
    preds = lr.predict_proba(D)
    cv_score = -log_loss(y, preds)

    return (cv_score, sum([len(x) for x in shapelets]))