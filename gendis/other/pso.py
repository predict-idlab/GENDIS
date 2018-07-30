import other_util
import numpy as np
from deap import base, creator, algorithms, tools

class ParticleSwarmExtractor():
    def __init__(self, particles=50, iterations=25, verbose=True, wait=5,
                 smin=-0.25, smax=0.25, phi1=1, phi2=1):
        self.particles = particles
        self.iterations = iterations
        self.verbose = verbose
        self.wait = wait
        self.smin = smin
        self.smax = smax
        self.phi1 = phi1
        self.phi2 = phi2

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric=other_util.calculate_ig):
        if min_len is None:
            min_len = 4
        if max_len is None:
            max_len = timeseries.shape[1]

        def random_shapelet():
            rand_row_idx = np.random.randint(timeseries.shape[0])
            rand_length = np.random.choice(range(min_len, max_len))
            rand_col_start_idx = np.random.randint(timeseries.shape[1] - rand_length)
            return timeseries[
                rand_row_idx, 
                rand_col_start_idx:rand_col_start_idx+rand_length
            ]

        def generate(smin, smax, n):
            parts = []
            for _ in range(n):
                rand_shap = random_shapelet()
                part = creator.Particle(rand_shap)
                part.speed = np.random.uniform(smin, smax, len(rand_shap))
                part.smin = smin
                part.smax = smax
                parts.append(part)
            return parts

        def updateParticle(part, best, phi1, phi2):
            u1 = np.random.uniform(0, phi1, len(part))
            u2 = np.random.uniform(0, phi2, len(part))
            #TODO: recheck this out (what if particles have variable lengths??)
            if len(part) < len(best):
                d, pos = other_util.sdist_with_pos(part, best)
                v_u1 = u1 * (part.best - part)
                v_u2 = u2 * (best[pos:pos+len(part)] - part)
                # These magic numbers are found in http://www.ijmlc.org/vol5/521-C016.pdf
                part.speed = 0.729*part.speed + np.minimum(np.maximum(1.49445 * (v_u1 + v_u2), part.smin), part.smax)
                part += part.speed
            else:
                d, pos = other_util.sdist_with_pos(best, part)
                v_u1 = (u1 * (part.best - part))[pos:pos+len(best)]
                v_u2 = u2[pos:pos+len(best)] * (best - part[pos:pos+len(best)])
                # These magic numbers are found in http://www.ijmlc.org/vol5/521-C016.pdf
                part.speed[pos:pos+len(best)] = 0.729*part.speed[pos:pos+len(best)] + np.minimum(np.maximum(1.49445 * (v_u1 + v_u2), part.smin), part.smax)
                part[pos:pos+len(best)] += part.speed[pos:pos+len(best)]


        def cost(shapelet):
            L = []
            for k in range(len(timeseries)):
                D = timeseries[k, :]
                dist = other_util.sdist(shapelet, D)
                L.append((dist, labels[k]))
            return metric(L)

        weights = (1.0,)
        if metric == 'ig':
            weights = (1.0, 1.0)
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, 
                       speed=list, smin=None, smax=None, best=None)

        toolbox = base.Toolbox()
        toolbox.register("population", generate, smin=self.smin, smax=self.smax)
        toolbox.register("update", updateParticle, phi1=self.phi1, phi2=self.phi2)
        toolbox.register("evaluate", cost)

        pop = toolbox.population(n=self.particles)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        GEN = 10000
        best = None
        it_wo_improvement = 0
        g = 0

        for g in range(GEN):
            it_wo_improvement += 1
            for part in pop:
                part.fitness.values = toolbox.evaluate(part)
                if part.best is None or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if best is None or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    it_wo_improvement = 0
            for part in pop:
                toolbox.update(part, best)

            # Gather all the fitnesses in one list and print the stats
            logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
            print(logbook.stream)

            if it_wo_improvement >= self.wait:
                break

        return [best]