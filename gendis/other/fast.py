import other_util
from tqdm import trange

class LRUCache():
    def __init__(self, size=5):
        self.values = []
        self.size = size

    def put(self, value):
        while len(self.values) >= self.size:
            self.values.remove(self.values[0])

        self.values.append(value)

class FastExtractor():
    def __init__(self, pruning=False, cache_size=10):
        self.pruning = pruning
        self.cache_size = cache_size

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric=other_util.calculate_ig):
        if min_len is None:
            min_len = 4
        if max_len is None:
            max_len = timeseries.shape[1]
            
        shapelets = []
        for j in trange(len(timeseries), desc='timeseries', position=0):
            S = timeseries[j, :]
            stats = {}
            # Pre-compute all metric arrays, which will allow us to
            # calculate the distance between two timeseries in constant time
            for k in range(len(timeseries)):
                metrics = other_util.calculate_metric_arrays(S, timeseries[k, :])
                stats[(j, k)] = metrics

            for l in range(min_len, max_len):  
                # Keep a history to calculate an upper bound, this could
                # result in pruning,LRUCache thus avoiding the construction of 
                # the orderline L (which is an expensive operation)
                H = LRUCache(size=self.cache_size)
                for i in range(len(S) - l + 1):
                    if self.pruning:
                        # Check if we can prune
                        prune = False
                        for w in range(len(H.values)):
                            L_prime, S_prime = H.values[w]
                            R = other_util.sdist(S[i:i+l], S_prime)
                            if other_util.upper_ig(L_prime.copy(), R) < max_gain:
                                prune = True
                                break
                        if prune: continue

                    # Extract a shapelet from S, start at index i with length l
                    L = []  # An orderline with distances to shapelet & labels
                    for k in range(len(timeseries)):
                        S_x, S_x2, S_y, S_y2, M = stats[(j, k)]
                        L.append((
                            other_util.sdist_metrics(i, l, S_x, S_x2, S_y, S_y2, M),
                            labels[k]
                        ))
                    score = metric(L)
                    shapelets.append(([list(S[i:i+l])] + list(score) + [j, i, l]))

                    if self.pruning:
                        H.put((L, S[i:i+l]))

        shapelets = sorted(shapelets, key=lambda x: x[1:], reverse=True)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets
