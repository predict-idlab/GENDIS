import other_util
import numpy as np
from deap import base, creator, algorithms, tools
import time

class SingleGeneticExtractor():
    # TODO: Implement co-evolution, where we evolve multiple populations.
    # TODO: One population per specified length. Else, the population converges
    # TODO: to similar shapelets of same length. Or alternatively: speciation!
    def __init__(self, population_size=25, iterations=50, verbose=True,
                 mutation_prob=0.25, crossover_prob=0.25, wait=5):
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.wait = wait
        #np.random.seed(1337)


    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric=other_util.calculate_ig):
        # TODO: If nr_shapelets > 1, then represent individuals by
        # TODO: `nr_shapelets` shapelets (instead of taking top-k from hof)
        if min_len is None:
            min_len = 4
        if max_len is None:
            max_len = timeseries.shape[1]

        weights = (1.0, 1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax, score=None)

        def random_shapelet():
            rand_row_idx = np.random.randint(timeseries.shape[0])
            rand_length = np.random.choice(range(min_len, max_len), size=1)[0]
            rand_col_start_idx = np.random.randint(timeseries.shape[1] - rand_length)
            return timeseries[
                rand_row_idx, 
                rand_col_start_idx:rand_col_start_idx+rand_length
            ]

        def cost(shapelet):
            L = []
            for k in range(len(timeseries)):
                D = timeseries[k, :]
                dist = other_util.sdist_no_norm(shapelet, D)
                L.append((dist, labels[k]))
            return metric(L)

        def mutation(pcls, shapelet):
            return tools.mutGaussian(shapelet, mu=0, sigma=0.1, indpb=0.1)[0]


        toolbox = base.Toolbox()
        toolbox.register("mate_one", tools.cxOnePoint)
        toolbox.register("mate_two", tools.cxTwoPoint)
        #toolbox.register("mutate", mutation, creator.Individual)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

        toolbox.register("individual",  tools.initIterate, creator.Individual, random_shapelet)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", cost)
        toolbox.register("select", tools.selRoulette)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        hof = tools.HallOfFame(nr_shapelets)

        pop = toolbox.population(n=self.population_size)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        it, best_it = 1, 1
        best_score = float('-inf')
        print('it\t\tavg\t\tstd\t\tmax\t\ttime')

        while it <= self.iterations and it - best_it < self.wait:
            #print(Counter([len(x) for x in pop]))
            start = time.time()

            # Apply selection and cross-over the selected individuals
            # TODO: Move this to a cross-over for-loop and just select 2 individuals in each it
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    if np.random.random() < 0.5:
                        toolbox.mate_one(child1, child2)
                    else:
                        toolbox.mate_two(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation to each individual
            for idx, indiv in enumerate(offspring):
                if np.random.random() < self.mutation_prob:
                    toolbox.mutate(indiv)
                    del indiv.fitness.values

            # Update the fitness values            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population and update hall of fame
            pop[:] = offspring
            it_stats = stats.compile(pop)
            hof.update(pop)
            print('{}\t\t{}\t\t{}\t\t{}\t{}'.format(
                it, 
                np.around(it_stats['avg'], 4), 
                np.around(it_stats['std'], 3), 
                np.around(it_stats['max'], 6),
                np.around(time.time() - start, 4), 
            ))

            if it_stats['max'] > best_score:
                best_it = it
                best_score = it_stats['max']

            it += 1

        return hof