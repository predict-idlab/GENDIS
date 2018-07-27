class GeneticExtractor():
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    n_population : int, default=300
        Number of population for the genetic algorithm.

    crossover_proba : float, default=0.5
        Probability of crossover for the genetic algorithm.

    mutation_proba : float, default=0.2
        Probability of mutation for the genetic algorithm.

    n_generations : int, default=40
        Number of generations for the genetic algorithm.

    crossover_independent_proba : float, default=0.1
        Independent probability of crossover for the genetic algorithm.

    mutation_independent_proba : float, default=0.05
        Independent probability of mutation for the genetic algorithm.

    tournament_size : int, default=3
        Tournament size for the genetic algorithm.

    caching : boolean, default=False
        If True, scores of the genetic algorithm are cached.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    generation_scores_ : array of shape [n_generations]
        The maximum cross-validation score for each generation.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    An example showing genetic shapelet extraction:

    >>> from tslearn.generators import random_walk_blobs
    >>> from genetic import GeneticExtractor
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, d=2, n_blobs=2)
    >>> extractor = GeneticExtractor()
    >>> extractor.fit(X, y)

    """
    def __init__(self, population_size=25, iterations=50, verbose=True,
                 add_noise_prob=0.3, add_shapelet_prob=0.3, wait=10, plot=True,
                 remove_shapelet_prob=0.3, crossover_prob=0.66, n_jobs=4):
        """

        """
        np.random.seed(1337)
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


    def fit(self, ts, labels):
        """Fit the GeneticSelectionCV model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        ts : array-like, shape = [n_ts]
            The training input timeseries. 

        labels : array-like, shape = [n_samples]
            The target values.
        """
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
                rand_row = np.random.randint(ts.shape[0])
                rand_length = np.random.randint(self.min_len, self.max_len)
                rand_col = np.random.randint(ts.shape[1] - rand_length)
                shaps.append(ts[rand_row, rand_col:rand_col+rand_length])
            return shaps

        def kmeans(n_shapelets, shp_len, n_draw=2500):
            """Sample subseries from the timeseries and apply K-Means on them"""
            # Sample a `n_draw` subseries of length `shp_len`
            n_ts, sz = X.shape
            indices_ts = np.random.choice(n_ts, size=n_draw, replace=True)
            start_idx = np.random.choice(sz - shp_len + 1, size=n_draw, 
                                         replace=True)
            end_idx = start_idx + shp_len

            subseries = np.zeros((n_draw, shp_len))
            for i in range(n_draw):
                subseries[i] = X[indices_ts[i], start_idx:end_idx]

            tskm = TimeSeriesKMeans(n_clusters=n_shapelets, metric="euclidean", 
                                    verbose=False)
            return tskm.fit(subseries).cluster_centers_

        def motif(n_shapelets, n_draw=2500):
            """Extract some motifs from sampled timeseries"""
            shaps = []
            for _ in range(n_shapelets):
                rand_length = np.random.randint(self.min_len, self.max_len)
                subset_idx = np.random.choice(range(len(ts)), 
                                              size=int(0.75*len(ts)), 
                                              replace=True)
                ts = ts[subset_idx, :].flatten()
                matrix_profile, _ = mstamp_stomp(ts, rand_length)
                motif_idx = matrix_profile[0, :].argsort()[-1]
                shaps.append(ts[motif_idx:motif_idx + rand_length])
            return shaps

        def create_individual(n_shapelets=1):
            rand = np.random.random()
            if rand < 1./3.:
                rand_length = np.random.randint(self.min_len, self.max_len)
                return kmeans(n_shapelets, rand_length)
            elif 1./3. < rand < 2./3.:
                return motif(n_shapelets)
            else:
                return random_shapelet(n_shapelets)


        def cost(shapelets):
            """
            .
            """
            start = time.time()
            X = np.zeros((len(ts), len(shapelets)))
            for k in range(len(ts)):
                D = ts[k, :]
                for j in range(len(shapelets)):
                    dist = util.sdist_no_norm(shapelets[j].flatten(), D)
                    X[k, j] = dist

                
            lr = LogisticRegression()
            cv_score = -log_loss(
                self.labels, 
                cross_val_predict(
                    lr, X, self.labels, method='predict_proba', 
                    cv=StratifiedKFold(n_splits=3, shuffle=True, 
                                       random_state=1337)
                )
            )

            return (cv_score, sum([len(x) for x in shapelets]))

        def add_noise(shapelets):
            """Add random noise to a random shapelet"""
            rand_shapelet = np.random.randint(len(shapelets))
            tools.mutGaussian(shapelets[rand_shapelet], 
                              mu=0, sigma=0.1, indpb=0.15)

            return shapelets,

        def add_shapelet(shapelets):
            """Add a shapelet to the individual"""
            shapelets.append(create_individual(n_shaps=1))

            return shapelets,

        def remove_shapelet(shapelets):
            """Remove a random shapelet from the individual"""
            if len(shapelets) > 1:
                rand_shapelet = np.random.randint(len(shapelets))
                shapelets.pop(rand_shapelet)

            return shapelets,

        def merge_crossover(ind1, ind2):
            """ """
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
                    ind1[row_idx] = euclidean_barycenter([ts1, t2])
                    ind1[row_idx] = ind1[row_idx][~np.isnan(ind1[row_idx])]

            # Apply the same for the elements in ind2
            for col_idx in range(similarity_matrix.shape[1]):
                mask = similarity_matrix[:, col_idx] != 1.0
                non_equals = similarity_matrix[:, col_idx][mask]
                if len(non_equals):
                    max_row_idx = np.argmax(non_equals)
                    ts1 = list(ind1[max_row_idx]).copy()
                    ts2 = list(ind2[col_idx]).copy()
                    ind2[col_idx] = euclidean_barycenter([ts1, t2])
                    ind2[col_idx] = ind2[col_idx][~np.isnan(ind2[col_idx])]

            return ind1, ind2

        def point_crossover(ind1, ind2):
            """ """
            if len(ind1) > 1 and len(ind2) > 1:
                if np.random.random() < 0.5:
                    ind1, ind2 = tools.cxOnePoint(list(ind1), list(ind2))
                else:
                    ind1, ind2 = tools.cxTwoPoint(list(ind1), list(ind2))
            
            return ind1, ind2

        # TODO: Make a comment why this is here
        set_config(assume_finite=True)

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
        if self.verbose:
            print('it\t\tavg\t\tstd\t\tmax\t\ttime')

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
        if self.plot:
            if self.population_size <= 20:
                f, ax = plt.subplots(4, height, sharex=True)
            else:
                plt.figure(figsize=(15, 5))
                plt.xlim([0, len(timeseries[0])])

        # The genetic algorithm starts here
        while it <= self.iterations and it - best_it < self.wait:
            gen_start = time.time()

            # Clone the population into offspring
            offspring = list(map(toolbox.clone, pop))

            # Plot the fittest individual of our population
            if self.plot:
                if self.population_size <= 20:
                    for ix, ind in enumerate(offspring):
                        ax[ix//height][ix%height].clear()
                        for s in ind:
                            ax[ix//height][ix%height].plot(range(len(s)), s)
                    plt.pause(0.001)

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
                    pass

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

        return best_ind

    def transform(self, ts):
        pass

    def fit_transform(self, ts, labels):
        pass