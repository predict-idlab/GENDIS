import other_util
from tqdm import trange


class BruteForceExtractor():
    def __init__(self):
        pass

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric=other_util.calculate_ig):
        if min_len is None:
            min_len = 4
        if max_len is None:
            max_len = timeseries.shape[1]

        shapelets = []
        for j in trange(len(timeseries), desc='timeseries', position=0):
            # We will extract shapelet candidates from S
            S = timeseries[j, :]
            for l in range(min_len, max_len):  
                for i in range(len(S) - l + 1):
                    candidate = S[i:i+l]
                    # Compute distances to all other timeseries
                    L = []  # The orderline, to calculate entropy, only for IG
                    for k in range(len(timeseries)):
                        D = timeseries[k, :]
                        dist = other_util.sdist(candidate, D)
                        L.append((dist, labels[k]))
                    score = metric(L)
                    shapelets.append((list(candidate), list(score), [j, i, l]))

        shapelets = sorted(shapelets, key=lambda x: x[1:], reverse=True)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets
