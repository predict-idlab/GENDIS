import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import other_util
import pickle
from scipy.stats import norm
import other_util
from collections import Counter, defaultdict
from tqdm import trange

def calculate_breakpoints(n):
    """Calculate the n-1 different points in [-1, 1] which are the boundaries
    of different regions in N(0, 1) such that the probability of each
    region is equal."""
    split_points = []
    fraction = float(1)/float(n)
    for i in range(1, n):
        split_points.append(norm.ppf(i*fraction))
    return split_points


def calculate_distance_table(alphabet_size):
    """Calculate a matrix of possible distances of two words in an alphabet
    resulting from a SAX-transformation"""
    break_points = calculate_breakpoints(alphabet_size)
    distance_table = np.zeros((alphabet_size, alphabet_size))
    for r in range(alphabet_size):
        for c in range(alphabet_size):
            if abs(r - c) <= 1:
                distance_table[r, c] = 0
            else:
                d1 = break_points[max(r, c) - 1]
                d2 = break_points[min(r, c)]
                distance_table[r, c] = d1 - d2
    return distance_table


def _paa(ts, nr_windows):
    window_means = []
    window_size = int(np.floor(float(len(ts)) / float(nr_windows)))
    for i in range(nr_windows - 1):
        window = ts[i*window_size:(i+1)*window_size]
        window_means.append(np.mean(window))
    window_means.append(np.mean(ts[i*window_size:]))
    return window_means


def sax_distance(x_sax, y_sax, ts_length, nr_windows, alphabet_size, 
                 distance_table=None):
    if distance_table is None:
        distance_table = calculate_distance_table(alphabet_size)
    
    total_distance = 0
    for x, y in zip(x_sax, y_sax):
        total_distance += distance_table[x, y]
    return np.sqrt(ts_length / nr_windows) * total_distance


def transform_ts(ts, nr_windows, alphabet_size, symbol_map):
    """Transform a timeseries in their SAX representation"""
    #  if the standard deviation of the sequence before normalization is below 
    # an epsilon ε, we simply assign the entire word to 
    # the middle-ranged alphabet (e.g. 'cccccc' if a = 5)
    sequence = []
    window_means = _paa(ts, nr_windows)
    for mean in window_means:
        for interval in symbol_map:
            if interval[0] <= mean < interval[1]:
                sequence.append(symbol_map[interval])
    return np.array(sequence)


def get_symbol_map(alphabet_size):
    split_points = calculate_breakpoints(alphabet_size)
    symbol_map = {}
    symbol_map[(float('-inf'), split_points[0])] = 0
    for i, j in enumerate(range(len(split_points) - 1)):
        symbol_map[(split_points[j], split_points[j + 1])] = i + 1
    symbol_map[(split_points[-1], float('inf'))] = len(split_points)
    return symbol_map


def transform(timeseries, nr_windows, alphabet_size):
    """Transform a collection of timeseries in their SAX representation"""
    symbol_map = get_symbol_map(alphabet_size)

    transformed_ts = [transform_ts(ts, nr_windows, alphabet_size, symbol_map) 
                      for ts in timeseries]
    return np.array(transformed_ts)



class SAXExtractor():
    def __init__(self, alphabet_size=4, sax_length=16, nr_candidates=100, 
                 iterations=5, mask_size=3):
        super(SAXExtractor, self).__init__()
        self.alphabet_size = alphabet_size
        self.sax_length = sax_length
        self.nr_candidates = nr_candidates
        self.iterations = iterations
        self.mask_size = mask_size

    def _random_mask(self, sax_timeseries, mask_size=5):
        """In order to calculate similarity between different timeseries
        in the SAX domain, we apply random masks and check whether the 
        remainder of the timeseries are equal to eachother.

        Parameters:
        -----------
        * sax_timeseries (3D np.array: timeseries x sax_words x word_length)
             The timeseries to mask
        * mask_size (int)
             How many elements should be masked
        """
        random_idx = np.random.choice(
            range(sax_timeseries.shape[2]),
            size=sax_timeseries.shape[2] - mask_size,
            replace=False
        )
        return sax_timeseries[:, :, random_idx]


    def _create_score_table(self, sax_timeseries, labels, iterations=10, 
                            mask_size=5):
        unique_labels = list(set(labels))
        score_table = np.zeros((
            sax_timeseries.shape[0], 
            sax_timeseries.shape[1],
            len(unique_labels)
        ))

        for it in range(iterations):
            masked_timeseries = self._random_mask(sax_timeseries, mask_size)
            hash_table = defaultdict(list)
            for ts_idx in range(masked_timeseries.shape[0]):
                for sax_idx in range(masked_timeseries.shape[1]):
                    key = tuple(list(masked_timeseries[ts_idx, sax_idx]))
                    hash_table[key].append((ts_idx, sax_idx))
            
            for bucket in hash_table:
                for (ts_idx1, sax_idx) in hash_table[bucket]:
                    unique_idx = set([x[0] for x in hash_table[bucket]])
                    for idx in unique_idx:
                        score_table[
                            ts_idx1, 
                            sax_idx, 
                            unique_labels.index(labels[idx])
                        ] += 1

        return score_table

    def extract(self, timeseries, labels, min_len=None, max_len=None, 
                nr_shapelets=1, metric=other_util.calculate_ig):
        if min_len is None:
            min_len = sax_length
        if max_len is None:
            max_len = timeseries.shape[1]

        unique_classes = set(labels)
        classes_cntr = Counter(labels)

        shapelets = []
        for l in trange(min_len, max_len, desc='length', position=0):
            # To select the candidates, all subsequences of length l from   
            # all time series are created using the sliding window technique, 
            # and we create their corresponding SAX word and keep them in SAXList 
            sax_words = np.zeros((
                len(timeseries), 
                timeseries.shape[1] - l + 1,
                self.sax_length
            ))
            for ts_idx, ts in enumerate(timeseries):
                # Extract all possible subseries, by using a sliding window
                # with shift=1
                subseries = []
                for k in range(len(ts) - l + 1):
                    subseries.append(other_util.z_norm(ts[k:k+l]))
                # Transform all the subseries and add them to the sax_words
                transformed_timeseries = transform(subseries, self.sax_length, 
                                                   self.alphabet_size)
                sax_words[ts_idx] = transformed_timeseries
            
            score_table = self._create_score_table(sax_words, labels, 
                                                   iterations=self.iterations,
                                                   mask_size=self.mask_size)
            max_score_table = np.ones_like(score_table)
            for c in unique_classes:
                max_score_table[:, :, c] = classes_cntr[c] * self.iterations
            rev_score_table = max_score_table - score_table

            # TODO: Can we replace this simple power calculation by a more
            # powerful metric to heuristically measure the quality
            power = []
            for ts_idx in range(score_table.shape[0]):
                for sax_idx in range(score_table.shape[1]):
                    min_val, max_val = float('inf'), float('-inf')
                    total = 0
                    for class_idx in range(score_table.shape[2]):
                        score = score_table[ts_idx, sax_idx, class_idx]
                        rev_score = rev_score_table[ts_idx, sax_idx, class_idx]
                        diff = score - rev_score
                        if diff > max_val:
                            max_val = diff
                        if diff < min_val:
                            min_val = diff
                        total += abs(diff)

                    v = (total-abs(max_val)-abs(min_val)) + abs(max_val-min_val)
                    power.append((v, (ts_idx, sax_idx)))
            
            top_candidates = sorted(power, key=lambda x: -x[0])[:self.nr_candidates]
            for score, (ts_idx, sax_idx) in top_candidates:
                candidate = timeseries[ts_idx][sax_idx:sax_idx+l]
                L = []  # The orderline, to calculate entropy
                for k in range(len(timeseries)):
                    D = timeseries[k, :]
                    dist = other_util.sdist(candidate, D)
                    L.append((dist, labels[k]))
                score = metric(L)
                shapelets.append(([list(candidate)] + list(score) + [ts_idx, sax_idx, l]))

        shapelets = sorted(shapelets, key=lambda x: x[1:], reverse=True)
        best_shapelets = [x[0] for x in shapelets[:nr_shapelets]]
        return best_shapelets