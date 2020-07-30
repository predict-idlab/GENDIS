# Standard lib
from collections import defaultdict, Counter, OrderedDict
import array
import time

# "Standard" data science libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Serialization
import pickle

# Evolutionary algorithms framework
from deap import base, creator, algorithms, tools

# Parallelization
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

# ML
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

# Data padding
from tslearn.utils import to_time_series_dataset

# Custom fitness function
try:
    from fitness import logloss_fitness
except:
    from gendis.fitness import logloss_fitness

# Pairwise distances
try:
    from pairwise_dist import _pdist
except:
    from gendis.pairwise_dist import _pdist

# Custom genetic operators
try:
    from operators import random_shapelet, kmeans
    from operators import add_shapelet, remove_shapelet, mask_shapelet
    from operators import (merge_crossover, point_crossover, 
                           shap_point_crossover)
except:
    from gendis.operators import random_shapelet, kmeans
    from gendis.operators import add_shapelet, remove_shapelet, mask_shapelet
    from gendis.operators import (merge_crossover, point_crossover, 
                                  shap_point_crossover)

from inspect import signature

# Ignore warnings
import warnings; warnings.filterwarnings('ignore')

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class GeneticExtractor(BaseEstimator, TransformerMixin):
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    population_size : int
        The number of individuals in our population. Increasing this parameter
        increases both the runtime per generation, as the probability of
        finding a good solution.

    iterations : int
        The maximum number of generations the algorithm may run.

    wait : int
        If no improvement has been found for `wait` iterations, then stop

    add_noise_prob : float
        The chance that gaussian noise is added to a random shapelet from a
        random individual every generation

    add_shapelet_prob : float
        The chance that a shapelet is added to a random shapelet set every gen

    remove_shapelet_prob : float
        The chance that a shapelet is deleted to a random shap set every gen

    crossover_prob : float
        The chance that of crossing over two shapelet sets every generation

    normed : boolean
        Whether we first have to normalize before calculating distances

    n_jobs : int
        The number of threads to use

    verbose : boolean
        Whether to print some statistics in every generation

    plot : object
        Whether to plot the individuals every generation (if the population 
        size is <= 20), or to plot the fittest individual

    Attributes
    ----------
    shapelets : array-like
        The fittest shapelet set after evolution
    label_mapping: dict
        A dictionary that maps the labels to the range [0, ..., C-1]

    Example
    -------
    An example showing genetic shapelet extraction on a simple dataset:

    >>> from tslearn.generators import random_walk_blobs
    >>> from genetic import GeneticExtractor
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> np.random.seed(1337)
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, noise_level=0.1)
    >>> X = np.reshape(X, (X.shape[0], X.shape[1]))
    >>> extractor = GeneticExtractor(iterations=5, population_size=10)
    >>> distances = extractor.fit_transform(X, y)
    >>> lr = LogisticRegression()
    >>> _ = lr.fit(distances, y)
    >>> lr.score(distances, y)
    1.0
    """
    def __init__(self, population_size=50, iterations=25, verbose=False, 
                 normed=False, mutation_prob=0.1, wait=10, plot=None, 
                 max_shaps=None, crossover_prob=0.4, n_jobs=1, max_len=None, 
                 fitness=None, init_ops=[random_shapelet, kmeans], 
                 cx_ops=[merge_crossover, point_crossover, 
                         shap_point_crossover], 
                 mut_ops=[add_shapelet, remove_shapelet, mask_shapelet]):
        # Hyper-parameters
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.plot = plot
        self.wait = wait
        self.n_jobs = n_jobs
        self.normed = normed
        self.max_len = max_len
        self.max_shaps = max_shaps
        self.init_ops = init_ops
        self.cx_ops = cx_ops
        self.mut_ops = mut_ops

        if fitness is None:
            self.fitness = logloss_fitness
        else:
            # Do some initial checks
            assert callable(fitness)
            assert len(signature(fitness).parameters) == 5
            self.fitness = fitness

        # Attributes
        self.label_mapping = {}
        self.shapelets = []
        self._min_length = 0

    def _convert_X(self, X):
        if isinstance(X, list):
            for i in range(len(X)):
                X[i] = np.array(X[i])
            X = np.array(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Pad the timeseries such that they all have the same length
        X = to_time_series_dataset(X)
        if X.shape[2] > 1:
            raise ValueError('Multivariate timeseries are currently '
                             'not yet supported')
        X = np.reshape(X, (X.shape[0], X.shape[1]))

        return X

    def _convert_y(self, y):
        # Map labels to [0, ..., C-1]
        for j, c in enumerate(np.unique(y)):
            self.label_mapping[c] = j

        # Use pandas map function and convert to numpy
        y = np.reshape(pd.Series(y).map(self.label_mapping).values, (-1, 1))

        return y

    def fit(self, X, y):
        """Extract shapelets from the provided timeseries and labels.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        y : array-like, shape = [n_samples]
            The target values.
        """
        X = self._convert_X(X)
        y = self._convert_y(y)
        self._min_length = np.min(np.sum(~np.isnan(X), axis=1))
        self.min_shap_len = 4
        
        if self._min_length <= 4:
            raise Exception('Time series should be of at least length 4!')

        if self.max_len is None:
            if len(X[0]) > 20:
                self.max_len = len(X[0]) // 2
            else:
                self.max_len = len(X[0])

        if self.max_shaps is None:
            self.max_shaps = int(np.sqrt(self._min_length)) + 1

        # Sci-kit learn check for label vector.
        check_array(y)

        # We will try to maximize the negative logloss of LR in CV.
        # In the case of ties, we pick the one with least number of shapelets
        weights = (1.0, -1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)

        # Individual are lists (of shapelets (list))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        cache = LRUCache(2048)

        # Keep a history of the evolution
        self.history = []

        def create_individual(n_shapelets=None):
            """Generate a random shapelet set"""
            if n_shapelets is None:
                n_shapelets = np.random.randint(2, self.max_shaps)

            init_op = np.random.choice(self.init_ops)
            return init_op(X, n_shapelets, self.min_shap_len, self._min_length)

        # Register all operations in the toolbox
        toolbox = base.Toolbox()

        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)


        # Register all our operations to the DEAP toolbox
        toolbox.register("merge", merge_crossover)
        deap_cx_ops = []
        for i, cx_op in enumerate(self.cx_ops):
            toolbox.register("cx{}".format(i), cx_op)
            deap_cx_ops.append(getattr(toolbox, ("cx{}".format(i))))
        deap_mut_ops = []
        for i, mut_op in enumerate(self.mut_ops):
            toolbox.register("mutate{}".format(i), mut_op)
            deap_mut_ops.append(getattr(toolbox, ("mutate{}".format(i))))
        toolbox.register("create", create_individual)
        toolbox.register("individual",  tools.initIterate, creator.Individual, 
                         toolbox.create)
        toolbox.register("population", tools.initRepeat, list, 
                         toolbox.individual)
        toolbox.register("evaluate", 
                         lambda shaps: self.fitness(X, y, shaps, 
                                                    verbose=self.verbose, 
                                                    cache=cache))
        # Small tournaments to ensure diversity
        toolbox.register("select", tools.selTournament, tournsize=3)  

        # Set up the statistics. We will measure the mean, std dev and max
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("q25", lambda x: np.quantile(x, 0.25))
        stats.register("q75", lambda x: np.quantile(x, 0.75))

        # Initialize the population and calculate their initial fitness values
        start = time.time()
        pop = toolbox.population(n=self.population_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Keep track of the best iteration, in order to do stop after `wait`
        # generations without improvement
        it, best_it = 1, 1
        best_ind = []
        best_score = float('-inf')

        # Set up a matplotlib figure and set the axes
        height = int(np.ceil(self.population_size/4))
        if self.plot is not None and self.plot != 'notebook':
            if self.population_size <= 20:
                f, ax = plt.subplots(4, height, sharex=True)
            else:
                plt.figure(figsize=(15, 5))
                plt.xlim([0, len(X[0])])

        # The genetic algorithm starts here
        while it <= self.iterations and it - best_it < self.wait:
            gen_start = time.time()

            # Clone the population into offspring
            offspring = list(map(toolbox.clone, pop))

            # Plot the fittest individual of our population
            if self.plot is not None:
                if self.population_size <= 20:
                    if self.plot == 'notebook':
                        f, ax = plt.subplots(4, height, sharex=True)
                    for ix, ind in enumerate(offspring):
                        ax[ix//height][ix%height].clear()
                        for s in ind:
                            ax[ix//height][ix%height].plot(range(len(s)), s)
                    plt.pause(0.001)
                    if self.plot == 'notebook': 
                        plt.show()

                else:
                    plt.clf()
                    for shap in best_ind:
                        plt.plot(range(len(shap)), shap)
                    plt.pause(0.001)

            # Iterate over all individuals and apply CX with certain prob
            start = time.time()
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                for cx_op in deap_cx_ops:
                    if np.random.random() < self.crossover_prob:
                        cx_op(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

            # Apply mutation to each individual
            start = time.time()
            for idx, indiv in enumerate(offspring):
                for mut_op in deap_mut_ops:
                    if np.random.random() < self.mutation_prob:
                        mut_op(indiv, toolbox)
                        del indiv.fitness.values

            # Update the fitness values
            start = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population and update hall of fame, statistics & history
            start = time.time()
            new_pop = toolbox.select(offspring, self.population_size - 1)
            fittest_ind = tools.selBest(pop + offspring, 1)
            pop[:] = new_pop + fittest_ind
            it_stats = stats.compile(pop)
            self.history.append([it, it_stats])

            # Print our statistics
            if self.verbose:
                if it == 1:
                    # Print the header of the statistics
                    print('it\t\tavg\t\tstd\t\tmax\t\ttime')

                print('{}\t\t{}\t\t{}\t\t{}\t{}'.format(
                    it, 
                    np.around(it_stats['avg'], 4), 
                    np.around(it_stats['std'], 3), 
                    np.around(it_stats['max'], 6),
                    np.around(time.time() - gen_start, 4), 
                ))

            # Have we found a new best score?
            if it_stats['max'] > best_score:
                best_it = it
                best_score = it_stats['max']
                best_ind = tools.selBest(pop + offspring, 1)
                self.fitness(X, y, best_ind[0], verbose=True, cache=cache)

                # Overwrite self.shapelets everytime so we can
                # pre-emptively stop the genetic algorithm
                best_shapelets = []
                for shap in best_ind[0]:
                    best_shapelets.append(shap.flatten())
                self.shapelets = best_shapelets


            it += 1

        best_shapelets = []
        for shap in best_ind[0]:
            best_shapelets.append(shap.flatten())
        self.shapelets = best_shapelets

    def transform(self, X):
        """After fitting the Extractor, we can transform collections of 
        timeseries in matrices with distances to each of the shapelets in
        the evolved shapelet set.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        Returns
        -------
        D : array-like, shape = [n_ts, n_shaps]
            The matrix with distances
        """
        X = self._convert_X(X)

        # Check is fit had been called
        check_is_fitted(self, ['shapelets'])

        # Construct (|X| x |S|) distance matrix
        D = np.zeros((len(X), len(self.shapelets)))
        _pdist(X, [shap.flatten() for shap in self.shapelets], D)

        return D

    def fit_transform(self, X, y):
        """Combine both the fit and transform method in one.

        Parameters
        ----------
        X : array-like, shape = [n_ts, ]
            The training input timeseries. Each timeseries must be an array,
            but the lengths can be variable

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        D : array-like, shape = [n_ts, n_shaps]
            The matrix with distances
        """
        # First call fit, then transform
        self.fit(X, y)
        D = self.transform(X)
        return D

    def save(self, path):
        """Write away all hyper-parameters and discovered shapelets to disk"""
        pickle.dump(self, open(path, 'wb+'))

    @staticmethod
    def load(path):
        """Instantiate a saved GeneticExtractor"""
        return pickle.load(open(path, 'rb'))