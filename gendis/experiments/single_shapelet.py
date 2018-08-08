import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../other')
sys.path.append('..')

import util
import other_util
from brute_force import BruteForceExtractor
from genetic_single import SingleGeneticExtractor

from tslearn.shapelets import ShapeletModel

np.random.seed(2018)

# Generate a simple dataset based on sinusoidal waves
Fs = 8000
f = 5
sample = 1640

fig, ax = plt.subplots(2, 3)

x = np.arange(sample)

ts1 = np.sin(2 * np.pi * f * x / Fs)[::40]
ts2 = -np.abs(np.sin(2 * np.pi * f * x / Fs + np.pi))[::40]
ts3 = np.sin(2 * np.pi * f * x / Fs + np.pi)[::40]
ts4 = np.abs(np.sin(2 * np.pi * f * x / Fs + np.pi))[::40]

ax[0][1].plot(range(len(ts1)), ts1, c='r')
ax[0][1].axis('off')
ax[0][2].plot(range(len(ts2)), ts2, c='r')
ax[0][2].axis('off')
ax[1][1].plot(range(len(ts3)), ts3, c='b')
ax[1][1].axis('off')
ax[1][2].plot(range(len(ts4)), ts4, c='b')
ax[1][2].axis('off')

ax[0][0].axis('off')
ax[0][0].annotate('Class 0', (0, 0.5), fontsize=16)

ax[1][0].axis('off')
ax[1][0].annotate('Class 1', (0, 0.5), fontsize=16)

plt.savefig('results/shap_artificial.svg')

# Split into X matrix and y vector
X = np.array([
    np.array(ts1),
    np.array(ts2),
    np.array(ts3),
    np.array(ts4)
])

y = np.array([
    0, 
    0, 
    1, 
    1
])

# Discover shapelet using:
#   * brute-force (will not find optimal one)
#   * genetic algorithm
#   * learning time series shapelets

ge = SingleGeneticExtractor(verbose=True, population_size=100, iterations=1000, wait=100)
ge.fit(X, y)
print(ge.shapelets)
gen_shapelet = ge.shapelets[0]

bfe = BruteForceExtractor()
bf_shapelet = bfe.extract(X, y)[0]

clf = ShapeletModel(n_shapelets_per_size={len(ts1): 1}, 
                    max_iter=5000, verbose_level=0, batch_size=1,
                    optimizer='sgd', weight_regularizer=0)
clf.fit(
        np.reshape(
            X, 
            (X.shape[0], X.shape[1], 1)
        ), 
        y
    )
lts_shapelet = clf.shapelets_[0]


# Plot the shapelets and orderline
fig, ax = plt.subplots(3, 3)

ax[0][0].axis('off')
ax[0][0].annotate('Brute Force', (0, 0.5), fontsize=16)

ax[0][1].axis('off')
ax[0][1].plot(range(len(bf_shapelet)), bf_shapelet)

dist_ts1 = util.sdist_no_norm(bf_shapelet, ts1)
dist_ts2 = util.sdist_no_norm(bf_shapelet, ts2)
dist_ts3 = util.sdist_no_norm(bf_shapelet, ts3)
dist_ts4 = util.sdist_no_norm(bf_shapelet, ts4)
ax[0][2].scatter([dist_ts1, dist_ts2], [0, 0], c='r', alpha=0.5, s=100)
ax[0][2].scatter([dist_ts3, dist_ts4], [0, 0], c='b', marker='x', alpha=0.5, s=100)
ax[0][2].plot([-0.1, max([dist_ts1, dist_ts2, dist_ts3, dist_ts4]) + 0.1], [0, 0], c='k', alpha=0.25)
ax[0][2].axis('off')


ax[1][0].axis('off')
ax[1][0].annotate('Genetic', (0, 0.5), fontsize=16)

ax[1][1].axis('off')
ax[1][1].plot(range(len(gen_shapelet)), gen_shapelet)

dist_ts1 = util.sdist_no_norm(gen_shapelet, ts1)
dist_ts2 = util.sdist_no_norm(gen_shapelet, ts2)
dist_ts3 = util.sdist_no_norm(gen_shapelet, ts3)
dist_ts4 = util.sdist_no_norm(gen_shapelet, ts4)
ax[1][2].scatter([dist_ts1, dist_ts2], [0, 0], c='r', alpha=0.5, s=100)
ax[1][2].scatter([dist_ts3, dist_ts4], [0, 0], c='b', marker='x', alpha=0.5, s=100)
ax[1][2].plot([-0.1, max([dist_ts1, dist_ts2, dist_ts3, dist_ts4]) + 0.1], [0, 0], c='k', alpha=0.25)
ax[1][2].axis('off')


ax[2][0].axis('off')
ax[2][0].annotate('LTS', (0, 0.5), fontsize=16)

ax[2][1].axis('off')
ax[2][1].plot(range(len(lts_shapelet)), lts_shapelet)

dist_ts1 = util.sdist_no_norm(lts_shapelet, ts1)
dist_ts2 = util.sdist_no_norm(lts_shapelet, ts2)
dist_ts3 = util.sdist_no_norm(lts_shapelet, ts3)
dist_ts4 = util.sdist_no_norm(lts_shapelet, ts4)
ax[2][2].scatter([dist_ts1, dist_ts2], [0, 0], c='r', alpha=0.5, s=100)
ax[2][2].scatter([dist_ts3, dist_ts4], [0, 0], c='b', marker='x', alpha=0.5, s=100)
ax[2][2].plot([-0.1, max([dist_ts1, dist_ts2, dist_ts3, dist_ts4]) + 0.1], [0, 0], c='k', alpha=0.25)
ax[2][2].axis('off')

plt.savefig('results/extracted_shapelets.svg')