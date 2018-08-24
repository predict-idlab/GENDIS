import sys
sys.path.append('..')
from gendis.genetic import GeneticExtractor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def teast_pipeline():
	X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
	X = np.reshape(X, (X.shape[0], X.shape[1]))
	extractor = GeneticExtractor(iterations=5, n_jobs=1, population_size=10)
	lr = LogisticRegression()
	pipeline = Pipeline([
		('shapelets', extractor),
		('log_reg', lr)
	])
	pipeline.fit(X, y)
	