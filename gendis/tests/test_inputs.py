import sys
sys.path.append('..')
from gendis.genetic import GeneticExtractor

import pandas as pd
import numpy as np

def test_accept_pd_DataFrame():
	X = [
		[0]*8, [0]*8, [0]*8, [0]*8,
		[1]*8, [1]*8, [1]*8, [1]*8,
	]
	y = [0, 0, 0, 0, 1, 1, 1, 1]

	pd_X = pd.DataFrame(X)
	pd_y = pd.Series(y)

	genetic = GeneticExtractor(population_size=5, iterations=5)
	genetic.fit(pd_X, pd_y)

def test_accept_list():
	X = [
		[0]*8, [0]*8, [0]*8, [0]*8,
		[1]*8, [1]*8, [1]*8, [1]*8,
	]
	y = [0, 0, 0, 0, 1, 1, 1, 1]

	genetic = GeneticExtractor(population_size=5, iterations=5)
	genetic.fit(X, y)

def test_accept_np_array():
	X = [
		[0]*8, [0]*8, [0]*8, [0]*8,
		[1]*8, [1]*8, [1]*8, [1]*8,
	]
	y = [0, 0, 0, 0, 1, 1, 1, 1]

	np_X = []
	for x in X:
		np_X.append(np.array(x))
	np_X = np.array(np_X)
	np_y = np.array(y)

	genetic = GeneticExtractor(population_size=5, iterations=5)
	genetic.fit(np_X, np_y)

def test_accept_variable_length_arrays():
	X = [
		[0]*10, [0]*6, [0]*8, [0]*6,
		[1]*8, [1]*7, [1]*8, [1]*5,
	]
	y = [0, 0, 0, 0, 1, 1, 1, 1]

	genetic = GeneticExtractor(population_size=5, iterations=5)
	genetic.fit(X, y)

def test_accept_float_labels():
	X = [
		[0]*8, [0]*8, [0]*8, [0]*8,
		[1]*8, [1]*8, [1]*8, [1]*8,
	]
	y = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]

	genetic = GeneticExtractor(population_size=5, iterations=5)
	genetic.fit(X, y)

def test_accept_string_labels():
	X = [
		[0]*8, [0]*8, [0]*8, [0]*8,
		[1]*8, [1]*8, [1]*8, [1]*8,
	]
	y = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

	genetic = GeneticExtractor(population_size=5, iterations=5)
	genetic.fit(X, y)