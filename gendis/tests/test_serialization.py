import sys
sys.path.append('..')
from gendis.genetic import GeneticExtractor
from tslearn.generators import random_walk_blobs
import numpy as np
import os

def test_serialization():
	X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
	X = np.reshape(X, (X.shape[0], X.shape[1]))
	extractor = GeneticExtractor(iterations=5, n_jobs=1, population_size=10)
	distances = extractor.fit_transform(X, y)
	extractor.save('temp.p')
	new_extractor = GeneticExtractor.load('temp.p')
	new_distances = new_extractor.transform(X)
	np.testing.assert_array_equal(distances, new_distances)
	os.remove('temp.p')