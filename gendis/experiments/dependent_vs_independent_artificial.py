import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import sys
sys.path.append('../other')
from sax import SAXExtractor
import other_util as util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import warnings; warnings.filterwarnings('ignore')

plt.rc('text', usetex=True)
plt.rc('text.latex', unicode=True)

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

cmap = plt.get_cmap('viridis')

class_map = {}
for i, c in enumerate(set(y_train)):
    class_map[c] = 'Class '+str(i)
    
added_labels = set()

plt.figure(figsize=(15, 5))
for ts, label in zip(X_train, y_train):
    if label in added_labels:
        plt.plot(range(len(ts)), ts, c=cmap(label / (len(set(y_train))  - 1)), alpha=0.5)
    else:
        plt.plot(range(len(ts)), ts, c=cmap(label / (len(set(y_train))  - 1)), alpha=0.5, 
        		 label=class_map[label])
        added_labels.add(label)

for ts, label in zip(X_test, y_test):
    plt.plot(range(len(ts), 2*len(ts)), ts, c=cmap(label / (len(set(y_train))  - 1)), alpha=0.5)

plt.axis('off')
plt.legend(prop={'size': 24}, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -.075))
plt.annotate('train', (2, 1.25), fontsize=24, ha='center', va='center')
plt.annotate('test', (7, 1.25), fontsize=24, ha='center', va='center')
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
        plt.plot(range(len(shap)), shap, c=cmap(0.0), alpha=0.5, label='dependent')
    else:
        plt.plot(range(len(shap)), shap, c=cmap(0.0), alpha=0.5)
for i, shap in enumerate(independent_shapelets):
    if i == 0:
        plt.plot(range(len(shap)), shap, c=cmap(1.0), alpha=0.5, label='independent')
    else:
        plt.plot(range(len(shap)), shap, c=cmap(1.0), alpha=0.5)
plt.legend(prop={'size': 14})
plt.axis('off')
plt.title('Independent vs dependent shapelets')
plt.savefig('results/shapelets.svg')

##############################################################################
#                  6. Generate scatter plot of distances                     #
##############################################################################

dependent_features = calculate_distance_matrix(X_test, dependent_shapelets)
independent_features = calculate_distance_matrix(X_test, independent_shapelets)

f, ax = plt.subplots(2, 2, figsize=(6, 6), gridspec_kw = {'height_ratios':[4, 1]})

for k, c in zip(range(3), ['#edf8b1', '#7fcdbb', '#2c7fb8']):
    filtered_features = dependent_features[y_test == k]
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].scatter([x[0] for x in filtered_features], 
                        [x[1] for x in filtered_features], 
                        s=100,
                        c=cmap(k / 2), label='Class {} -- dependent'.format(k))

for k, c in zip(range(3), ['#edf8b1', '#7fcdbb', '#2c7fb8']):
    filtered_features = independent_features[y_test == k]
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].scatter([x[0] for x in filtered_features], 
                        [x[1] for x in filtered_features], 
                        s=100,
                        c=cmap(k / 2), marker='x', label='Class {} -- independent'.format(k))

# Upper-left
ax[0][0].set_xlim(0, 0.1)
ax[0][0].set_xticks(np.arange(0, 0.11, step=0.1))
ax[0][0].set_ylim(1.75, 2.75)
ax[0][0].set_yticks(np.arange(1.75, 2.75, step=0.25))
ax[0][0].spines['bottom'].set_visible(False)
ax[0][0].spines['right'].set_visible(False)
ax[0][0].tick_params(bottom=False, labelbottom='off')

# Bottom-left
ax[1][0].set_xlim(0, 0.1)
ax[1][0].set_xticks(np.arange(0, 0.11, step=0.1))
ax[1][0].set_ylim(0, 0.25)
ax[1][0].set_yticks(np.arange(0, 0.26, step=0.25))
ax[1][0].spines['top'].set_visible(False)
ax[1][0].spines['right'].set_visible(False)

# Upper-right
ax[0][1].set_xlim(1.9, 2.1)
ax[0][1].set_xticks(np.arange(1.9, 2.11, step=0.1))
ax[0][1].set_ylim(1.75, 2.75)
ax[0][1].set_yticks(np.arange(1.75,2.75, step=0.25))
ax[0][1].spines['bottom'].set_visible(False)
ax[0][1].spines['left'].set_visible(False)
ax[0][1].tick_params(left=False, labelleft='off', bottom=False, labelbottom='off')

# Bottom-right
ax[1][1].set_xlim(1.9, 2.1)
ax[1][1].set_xticks(np.arange(1.9, 2.11, step=0.1))
ax[1][1].set_ylim(0, 0.25)
ax[1][1].set_yticks(np.arange(0, 0.25, step=0.25))
ax[1][1].spines['top'].set_visible(False)
ax[1][1].spines['left'].set_visible(False)
ax[1][1].tick_params(left=False, labelleft='off')

d = .015  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax[0][0].transAxes, color='k', clip_on=False)
ax[0][0].plot((-d, +d), (-d, +d), **kwargs)       
ax[0][0].plot((1 - d, 1 + d), (1-d, 1+d), **kwargs)

kwargs = dict(transform=ax[0][1].transAxes, color='k', clip_on=False)
ax[0][1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  
ax[0][1].plot((-d, +d), (1 - d, 1 + d), **kwargs) 

kwargs = dict(transform=ax[1][0].transAxes, color='k', clip_on=False)
ax[1][0].plot((-d, +d), (1 - d*4, 1 + d*4), **kwargs)  
ax[1][0].plot((1 - d, 1 + d), (-d*4, +d*4), **kwargs)     

kwargs = dict(transform=ax[1][1].transAxes, color='k', clip_on=False)
ax[1][1].plot((1 - d, 1 + d), (1 - d*4, 1 + d*4), **kwargs)  
ax[1][1].plot((-d, +d), (-d*4, +d*4), **kwargs)     

f.text(0.,0.5, "Distance $S_2$", ha="center", va="center", fontsize=16, rotation=90)
f.text(0.5,0., "Distance $S_1$", ha="center", va="center", fontsize=16)

ax[0][0].legend(prop={'size': 13}, ncol=1)

plt.tight_layout()
plt.savefig('results/distances_scatter.svg', bbox_inches='tight')

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