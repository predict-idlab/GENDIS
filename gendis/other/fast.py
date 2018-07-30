import util

class FastExtractor():
    def __init__(self, pruning=False, cache_size=10):
        self.pruning = pruning
        self.cache_size = cache_size

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric=util.calculate_ig):
        super(FastExtractor, self).extract(timeseries, labels, min_len,
                                           max_len, nr_shapelets, metric)
        shapelets = []
        for j in trange(len(timeseries), desc='timeseries', position=0):
            S = timeseries[j, :]
            stats = {}
            # Pre-compute all metric arrays, which will allow us to
            # calculate the distance between two timeseries in constant time
            for k in range(len(timeseries)):
                metrics = util.calculate_metric_arrays(S, timeseries[k, :])
                stats[(j, k)] = metrics

            for l in range(min_len, max_len):  
                # Keep a history to calculate an upper bound, this could
                # result in pruning,LRUCache thus avoiding the construction of the
                # orderline L (which is an expensive operation)
                H = LRUCache(size=self.cache_size)
                for i in range(len(S) - l + 1):
                    if self.pruning:
                        # Check if we can prune
                        prune = False
                        for w in range(len(H.values)):
                            L_prime, S_prime = H.values[w]
                            R = util.sdist(S[i:i+l], S_prime)
                            if util.upper_ig(L_prime.copy(), R) < max_gain:
                                prune = True
                                break
                        if prune: continue

                    # Extract a shapelet from S, starting at index i with length l
                    L = []  # An orderline with the distances to shapelet & labels
                    for k in range(len(timeseries)):
                        S_x, S_x2, S_y, S_y2, M = stats[(j, k)]
                        L.append((
                            util.sdist_metrics(i, l, S_x, S_x2, S_y, S_y2, M),
                            labels[k]
                        ))
                    score = metric(L)
                    shapelets.append(([list(S[i:i+l])] + list(score) + [j, i, l]))

                    if self.pruning:
                        H.put((L, S[i:i+l]))

        shapelets = sorted(shapelets)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets
