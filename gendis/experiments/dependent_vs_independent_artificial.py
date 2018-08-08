import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../other')
from sax import SAXExtractor
import other_util as util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import warnings; warnings.filterwarnings('ignore')

np.random.seed(2018)

def calculate_distance_matrix(X, shapelets):
    D = np.zeros((len(X), len(shapelets)))
    for smpl_idx, sample in enumerate(X):
        for shap_idx, shapelet in enumerate(shapelets):
            dist = util.sdist_no_norm(shapelet.flatten(), sample)
            D[smpl_idx, shap_idx] = dist
    return D

def extract_shapelets_with_tree(X, y, min_len, max_len, shapelets=[]):
    if len(set(y)) <= 1:
        return []

    label_map = {}
    for i, lab in enumerate(set(y)):
        label_map[lab] = i
    y = pd.Series(y).map(label_map).values

    # Extract the best shapelet
    sax = SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100, 
                       iterations=5, mask_size=3)
    shapelet = sax.extract(X, y, min_len=min_len, max_len=max_len)[0]

    # Partition X and call function recursively
    L = []
    X_left, y_left, X_right, y_right = [], [], [], []
    for k in range(len(X)):
        D = X[k, :]
        dist = util.sdist_no_norm(shapelet, D)
        L.append((dist, y[k]))
    threshold = util.get_threshold(L)

    for ts, label in zip(X, y):
        if util.sdist_no_norm(shapelet, ts) <= threshold:
            X_left.append(ts)
            y_left.append(label)
        else:
            X_right.append(ts)
            y_right.append(label)

    X_left = np.array(X_left)
    X_right = np.array(X_right)
    y_left = np.array(y_left)
    y_right = np.array(y_right)

    shapelets = [shapelet]
    shapelets += extract_shapelets_with_tree(X_left, y_left, min_len, max_len)
    shapelets += extract_shapelets_with_tree(X_right, y_right, min_len, max_len)
    return shapelets


##############################################################################
#                  1. Generate imbalanced artificial data                    #
##############################################################################

ts1 = np.array([0, 1, 0, -1, 0])
ts2 = np.array([1, 1, 0, 1, 0])
ts3 = np.array([-1, -1, 0, -1, 0])

X_train, X_test = [], []
y_train, y_test = [], []

for _ in range(25):
    X_train.append(ts1 + np.random.normal(scale=0.01, size=len(ts1)))
    y_train.append(0)

for _ in range(5):
    X_train.append(ts2 + np.random.normal(scale=0.05, size=len(ts2)))
    y_train.append(1)

for _ in range(5):
    X_train.append(ts3 + np.random.normal(scale=0.05, size=len(ts3)))
    y_train.append(2)

for _ in range(25):
    X_test.append(ts1 + np.random.normal(scale=0.01, size=len(ts1)))
    y_test.append(0)

for _ in range(5):
    X_test.append(ts2 + np.random.normal(scale=0.05, size=len(ts2)))
    y_test.append(1)

for _ in range(5):
    X_test.append(ts3 + np.random.normal(scale=0.05, size=len(ts3)))
    y_test.append(2)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

##############################################################################
#                          2. Plot train and test set                        #
##############################################################################

cmap = {
    0: 'c',
    1: 'g',
    2: 'r',
    3: 'g',
    4: 'r',
    5: 'y',
    6: 'b',
    7: 'w',
}

class_map = {}
for i, c in enumerate(set(y_train)):
    class_map[c] = i
    
added_labels = set()

plt.figure(figsize=(15, 5))
for ts, label in zip(X_train, y_train):
    if label in added_labels:
        plt.plot(range(len(ts)), ts, c=cmap[label], alpha=0.25)
    else:
        plt.plot(range(len(ts)), ts, c=cmap[label], alpha=0.25, 
        		 label=class_map[label])
        added_labels.add(label)

for ts, label in zip(X_test, y_test):
    plt.plot(range(len(ts), 2*len(ts)), ts, c=cmap[label], alpha=0.25)

plt.axis('off')
plt.legend(prop={'size': 14}, loc='center')
plt.title('Generated train and test set')
plt.annotate('train', (1.75, -1.25), fontsize=16)
plt.annotate('test', (6.75, -1.25), fontsize=16)
plt.ylim([-1.5, 1.5])
plt.savefig("results/data.svg", bbox_inches='tight')

##############################################################################
#                            3. Dependent discovery                          #
##############################################################################

dependent_shapelets = extract_shapelets_with_tree(X_train, y_train, 3, 5)
dependent_shapelets = np.array([np.array(x) for x in dependent_shapelets])

##############################################################################
#                           4. Independent discovery                         #
##############################################################################

sax = SAXExtractor(alphabet_size=4, sax_length=16, nr_candidates=100,
                   iterations=5, mask_size=3)
independent_shapelets = sax.extract(X_train, y_train, min_len=3, max_len=5, 
                                    nr_shapelets=2)
independent_shapelets = np.array([np.array(x) for x in independent_shapelets])

##############################################################################
#                       5. Plot discovered shapelets                         #
##############################################################################

plt.figure(figsize=(15, 5))
for i, shap in enumerate(dependent_shapelets):
    if i == 0:
        plt.plot(range(len(shap)), shap, c='r', alpha=0.33, label='dependent')
    else:
        plt.plot(range(len(shap)), shap, c='r', alpha=0.33)
for i, shap in enumerate(independent_shapelets):
    if i == 0:
        plt.plot(range(len(shap)), shap, c='b', alpha=0.33, label='independent')
    else:
        plt.plot(range(len(shap)), shap, c='b', alpha=0.33)
plt.legend(prop={'size': 14})
plt.axis('off')
plt.title('Independent vs dependent shapelets')
plt.savefig('results/shapelets.svg')

##############################################################################
#                  6. Generate scatter plot of distances                     #
##############################################################################

dependent_features = calculate_distance_matrix(X_test, dependent_shapelets)
independent_features = calculate_distance_matrix(X_test, independent_shapelets)

print(independent_features)

plt.figure()
for i, c in zip(range(3), ['r', 'g', 'b']):
    filtered_features = dependent_features[y_test == i]
    plt.scatter([x[0] for x in filtered_features], 
                [x[1] for x in filtered_features], 
                c=c, label='Class {}'.format(i))
plt.legend(prop={'size': 14})
plt.xlabel('Distance $S_1$')
plt.ylabel('Distance $S_2$')
plt.title('Scatter plot of distances to dependent shapelets')
plt.savefig('results/scatter_dependent.svg')

plt.figure()
for i, c in zip(range(3), ['r', 'g', 'b']):
    filtered_features = independent_features[y_test == i]
    plt.scatter([x[0] for x in filtered_features], 
                [x[1] for x in filtered_features], 
                c=c, label='Class {}'.format(i))
plt.legend(prop={'size': 14})
plt.xlabel('Distance $S_1$')
plt.ylabel('Distance $S_2$')
plt.title('Scatter plot of distances to independent shapelets')
plt.savefig('results/scatter_independent.svg')

##############################################################################
#                       7. Fit Logistic Regression                           #
##############################################################################

lr = GridSearchCV(LogisticRegression(), 
                  {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
lr.fit(calculate_distance_matrix(X_train, dependent_shapelets), y_train)
print('Dependent approach: {}'.format(lr.score(dependent_features, y_test)))

lr = GridSearchCV(LogisticRegression(), 
                  {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
lr.fit(calculate_distance_matrix(X_train, independent_shapelets), y_train)
print('Independent approach: {}'.format(lr.score(independent_features, y_test)))