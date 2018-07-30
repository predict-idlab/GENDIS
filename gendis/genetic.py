# Standard lib
from collections import defaultdict, Counter
import array
import time

# "Standard" data science libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Motif extraction
from mstamp_stomp import mstamp as mstamp_stomp

# Evolutionary algorithms framework
from deap import base, creator, algorithms, tools

# Time series operations
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.barycenters import euclidean_barycenter

# Parallelization
from pathos.multiprocessing import ProcessingPool as Pool

# ML
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss

# Util functions
import util

# Ignore warnings
import warnings; warnings.filterwarnings('ignore')

class GeneticExtractor():
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    population_size : int
        The number of individuals in our population. Increasing this parameter
        increases both the runtime per generation, as the probability of
        finding a good solution.

    iterations : int
        The maximum number of generations the algorithm may run.

    add_noise_prob : float
        The chance that gaussian noise is added to a random shapelet from a
        random individual every generation

    add_shapelet_prob : float
        The chance that a shapelet is added to a random shapelet set every gen

    remove_shapelet_prob : float
        The chance that a shapelet is deleted to a random shapelet set every gen

    crossover_prob : float
        The chance that of crossing over two shapelet sets every generation

    normed : boolean
        Whether we first have to normalize before calculating distances

    n_jobs : int
        The number of threads to use

    verbose : boolean
        Whether to print some statistics in every generation

    plot : boolean
        Whether to plot the individuals every generation (if the population 
        size is smaller than or equal to 20), or to plot the fittest individual

    Attributes
    ----------
    shapelets : array-like
        The fittest shapelet set after evolution

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
    >>> extractor = GeneticExtractor(iterations=5, n_jobs=1, population_size=10)
    >>> distances = extractor.fit_transform(X, y)
    >>> lr = LogisticRegression()
    >>> _ = lr.fit(distances, y)
    >>> lr.score(distances, y)
    1.0
    """
    def __init__(self, population_size=50, iterations=25, verbose=False, 
                 normed=False, add_noise_prob=0.3, add_shapelet_prob=0.3, 
                 wait=10, plot=None, remove_shapelet_prob=0.3, 
                 crossover_prob=0.66, n_jobs=4):
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.add_noise_prob = add_noise_prob
        self.add_shapelet_prob = add_shapelet_prob
        self.remove_shapelet_prob = remove_shapelet_prob
        self.crossover_prob = crossover_prob
        self.plot = plot
        self.wait = wait
        self.n_jobs = n_jobs
        self.normed = normed

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
        # If y is a 1D list, convert it to a 2D column np array
        if type(y) is list or len(y.shape) == 1:
            y = np.reshape(y, (-1, 1))

        # Sci-kit learn checks
        check_array(X)
        check_array(y)

        # Determine the minimum and maximum shapelet length
        min_len = 4
        max_len = min([len(x) for x in X])

        # We will try to maximize the negative logloss of LR in CV.
        # In the case of ties, we pick the one with least number of shapelets
        weights = (1.0, -1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)

        # Individual are lists (of shapelets (list))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def random_shapelet(n_shapelets):
            """Extract a random subseries from the training set"""
            shaps = []
            for _ in range(n_shapelets):
                rand_row = np.random.randint(X.shape[0])
                rand_length = np.random.randint(min_len, max_len)
                rand_col = np.random.randint(X.shape[1] - rand_length)
                shaps.append(X[rand_row, rand_col:rand_col+rand_length])
            if n_shapelets > 1:
                return np.array(shaps)
            else:
                return np.array(shaps[0])

        def kmeans(n_shapelets, shp_len, n_draw=1000):
            """Sample subseries from the timeseries and apply K-Means on them"""
            # Sample `n_draw` subseries of length `shp_len`
            n_ts, sz = X.shape
            indices_ts = np.random.choice(n_ts, size=n_draw, replace=True)
            start_idx = np.random.choice(sz - shp_len + 1, size=n_draw, 
                                         replace=True)
            end_idx = start_idx + shp_len

            subseries = np.zeros((n_draw, shp_len))
            for i in range(n_draw):
                subseries[i] = X[indices_ts[i], start_idx[i]:end_idx[i]]

            tskm = TimeSeriesKMeans(n_clusters=n_shapelets, metric="euclidean", 
                                    verbose=False)
            return tskm.fit(subseries).cluster_centers_

        def motif(n_shapelets, n_draw=100):
            """Extract some motifs from sampled timeseries"""
            shaps = []
            for _ in range(n_shapelets):
                rand_length = np.random.randint(min_len, max_len)
                subset_idx = np.random.choice(range(len(X)), 
                                              size=n_draw, 
                                              replace=True)
                ts = X[subset_idx, :].flatten()
                matrix_profile, _ = mstamp_stomp(ts, rand_length)
                motif_idx = matrix_profile[0, :].argsort()[-1]
                shaps.append(ts[motif_idx:motif_idx + rand_length])
            if n_shapelets > 1:
                return np.array(shaps)
            else:
                return np.array(shaps[0])

        def create_individual(n_shapelets=None):
            """ Generate a random shapelet set """
            if n_shapelets is None:
                n_shapelets = np.random.randint(2, int(np.sqrt(X.shape[1])) + 1)
            
            rand = np.random.random()
            if n_shapelets > 1:
                if rand < 1./3.:
                    rand_length = np.random.randint(min_len, max_len)
                    return kmeans(n_shapelets, rand_length)
                elif 1./3. < rand < 2./3.:
                    return motif(n_shapelets)
                else:
                    return random_shapelet(n_shapelets)
            else:
                if rand < 0.5:
                    return motif(n_shapelets)
                else:
                    return random_shapelet(n_shapelets)



        def cost(shapelets):
            """ Calculate the fitness of an individual/shapelet set"""
            start = time.time()
            D = np.zeros((len(X), len(shapelets)))
            for k in range(len(X)):
                ts = X[k, :]
                for j in range(len(shapelets)):
                    if self.normed:
                        dist = util.sdist(shapelets[j].flatten(), ts)
                    else:
                        dist = util.sdist_no_norm(shapelets[j].flatten(), ts)
                    D[k, j] = dist

                
            lr = LogisticRegression()
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1337)
            preds = cross_val_predict(lr, D, y, method='predict_proba', cv=skf) 
            cv_score = -log_loss(y, preds)

            return (cv_score, sum([len(x) for x in shapelets]))

        def add_noise(shapelets):
            """Add random noise to a random shapelet"""
            rand_shapelet = np.random.randint(len(shapelets))
            tools.mutGaussian(shapelets[rand_shapelet], 
                              mu=0, sigma=0.1, indpb=0.15)

            return shapelets,

        def add_shapelet(shapelets):
            """Add a shapelet to the individual"""
            shapelets.append(create_individual(n_shapelets=1))

            return shapelets,

        def remove_shapelet(shapelets):
            """Remove a random shapelet from the individual"""
            if len(shapelets) > 1:
                rand_shapelet = np.random.randint(len(shapelets))
                shapelets.pop(rand_shapelet)

            return shapelets,

        def merge_crossover(ind1, ind2):
            """ Merge shapelets from one set with shapelets from the other """
            # Construct a pairwise similarity matrix using GAK
            _all = list(ind1) + list(ind2)
            similarity_matrix = cdist_gak(ind1, ind2, sigma=sigma_gak(_all))

            # Iterate over shapelets in `ind1` and merge them with shapelets
            # from `ind2`
            for row_idx in range(similarity_matrix.shape[0]):
                # Remove all elements equal to 1.0
                mask = similarity_matrix[row_idx, :] != 1.0
                non_equals = similarity_matrix[row_idx, :][mask]
                if len(non_equals):
                    # Get the timeseries most similar to the one at row_idx
                    max_col_idx = np.argmax(non_equals)
                    ts1 = list(ind1[row_idx]).copy()
                    ts2 = list(ind2[max_col_idx]).copy()
                    # Merge them and remove nans
                    ind1[row_idx] = euclidean_barycenter([ts1, ts2])
                    ind1[row_idx] = ind1[row_idx][~np.isnan(ind1[row_idx])]

            # Apply the same for the elements in ind2
            for col_idx in range(similarity_matrix.shape[1]):
                mask = similarity_matrix[:, col_idx] != 1.0
                non_equals = similarity_matrix[:, col_idx][mask]
                if len(non_equals):
                    max_row_idx = np.argmax(non_equals)
                    ts1 = list(ind1[max_row_idx]).copy()
                    ts2 = list(ind2[col_idx]).copy()
                    ind2[col_idx] = euclidean_barycenter([ts1, ts2])
                    ind2[col_idx] = ind2[col_idx][~np.isnan(ind2[col_idx])]

            return ind1, ind2

        def point_crossover(ind1, ind2):
            """ Apply one- or two-point crossover on the shapelet sets """
            if len(ind1) > 1 and len(ind2) > 1:
                if np.random.random() < 0.5:
                    ind1, ind2 = tools.cxOnePoint(list(ind1), list(ind2))
                else:
                    ind1, ind2 = tools.cxTwoPoint(list(ind1), list(ind2))
            
            return ind1, ind2

        # Register all operations in the toolbox
        toolbox = base.Toolbox()

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)


        # Register all our operations to the DEAP toolbox
        toolbox.register("merge", merge_crossover)
        toolbox.register("cx", point_crossover)
        toolbox.register("mutate", add_noise)
        toolbox.register("add", add_shapelet)
        toolbox.register("remove", remove_shapelet)
        toolbox.register("individual",  tools.initIterate, creator.Individual, 
                         create_individual)
        toolbox.register("population", tools.initRepeat, list, 
                         toolbox.individual)
        toolbox.register("evaluate", cost)
        # Small tournaments to ensure diversity
        toolbox.register("select", tools.selTournament, tournsize=3)  

        # Set up the statistics. We will measure the mean, std dev and max
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        # Initialize the population and calculate their initial fitness values
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
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                try:
                    if np.random.random() < self.crossover_prob:
                        toolbox.merge(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                    if np.random.random() < self.crossover_prob:
                        toolbox.cx(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                except:
                    raise

            # Apply mutation to each individual
            for idx, indiv in enumerate(offspring):
                if np.random.random() < self.add_noise_prob:
                    toolbox.mutate(indiv)
                    del indiv.fitness.values
                if np.random.random() < self.add_shapelet_prob:
                    toolbox.add(indiv)
                    del indiv.fitness.values
                if np.random.random() < self.remove_shapelet_prob:
                    toolbox.remove(indiv)
                    del indiv.fitness.values

            # Update the fitness values         
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population and update hall of fame & statistics
            new_pop = toolbox.select(offspring, self.population_size - 1)
            fittest_ind = tools.selBest(pop + offspring, 1)
            pop[:] = new_pop + fittest_ind
            it_stats = stats.compile(pop)

            # Print our statistics
            if self.verbose:
                if it == 1:
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

            it += 1

        self.shapelets = np.array(best_ind[0])

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
        # Check is fit had been called
        check_is_fitted(self, ['shapelets'])

        # Input validation
        check_array(X)

        # Construct (|X| x |S|) distance matrix
        D = np.zeros((len(X), len(self.shapelets)))
        for smpl_idx, sample in enumerate(X):
            for shap_idx, shapelet in enumerate(self.shapelets):
                if self.normed:
                    dist = util.sdist(shapelet.flatten(), sample)
                else:
                    dist = util.sdist_no_norm(shapelet.flatten(), sample)
                D[smpl_idx, shap_idx] = dist

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
        # If y is a 1D list, convert it to a 2D column np array
        if type(y) is list or len(y.shape) == 1:
            y = np.reshape(y, (-1, 1))

        # Input validation
        check_array(X)
        check_array(y)
        
        # First call fit, then transform
        self.fit(X, y)
        D = self.transform(X)
        return D