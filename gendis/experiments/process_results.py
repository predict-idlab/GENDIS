# Standard Library
import os
import glob
from collections import defaultdict

# Data wrangling
import pandas as pd
import numpy as np

# Calculate accuracies
from sklearn.metrics import accuracy_score

# Load the datasets
from tslearn.datasets import UCR_UEA_datasets

# Print fancy table and create scatter plot
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from adjustText import adjust_text

# Paired Wilcoxon Test
from scipy.stats import wilcoxon

# Process result files in directory and create summarizing statistics + plots

# Comment this out if you want to process results of LTS vs GENDIS
#DIR = 'results/lts_vs_genetic/'
#method1 = 'genetic'
#method2 = 'learned'
#method1_name = 'GENDIS'
#method2_name = 'LTS'

# Comment this out if you want to process results of dependent vs independent
DIR = 'results/dependent_vs_independent/'
method1 = 'tree'
method2 = 'transform'
method1_name = 'dependent'
method2_name = 'independent'

accuracies1 = defaultdict(list)
accuracies2 = defaultdict(list)

data_loader = UCR_UEA_datasets()

# Iterate over files in directory, process the predictions (_proba) and save 
# them in a dict. Afterwards, print a table with aggregated results & create 
# scatter plot (stat test if |values| > 1)
datasets = set([x.split('_')[0] for x in os.listdir(DIR) 
                if x != '.keep'])

for dataset in datasets:
    glob_path = DIR + '{}_{}*lr_proba.csv'
    method1_files = glob.glob(glob_path.format(dataset, method1))
    method2_files = glob.glob(glob_path.format(dataset, method2))

    # First, we load the ground truth, needed to calculate accuracy
    _, _, _, ground_truth = data_loader.load_dataset(dataset)

    # Map classes to [0, ..., C-1]
    map_dict = {}
    for j, c in enumerate(set(ground_truth)):
        map_dict[c] = j
    ground_truth = pd.Series(ground_truth).map(map_dict).values

    # Iterate over files of method 1 and calculate accuracy
    for file in method1_files:
        preds = np.argmax(pd.read_csv(file, index_col=[0]).values, axis=1)
        accuracies1[dataset].append(accuracy_score(ground_truth, preds))

    # Iterate over files of method 2 and calculate accuracy
    for file in method2_files:
        preds = np.argmax(pd.read_csv(file, index_col=[0]).values, axis=1)
        accuracies2[dataset].append(accuracy_score(ground_truth, preds))

# Now create a fancy table with:
#	* The mean accuracy per dataset
#	* The standard deviation of that accuracy (if measures > 1)
#	* p-value from paired Wilcoxon test (if measures > 1)
comparison_table = [['dataset', 'acc. mean {}'.format(method1_name), 
				     'acc. mean {}'.format(method2_name), 
				 	 'acc. std. method1', 'acc. std. method2', 'p-value']]	 
method1_accs, method2_accs, p_values, names = [], [], [], []
for dataset in accuracies1:
	if dataset not in accuracies2: continue

	row = [
		dataset, 
		np.around(np.mean(accuracies1[dataset]), 4), 
		np.around(np.mean(accuracies2[dataset]), 4)
	]
	method1_accs.append(np.mean(accuracies1[dataset]))
	method2_accs.append(np.mean(accuracies2[dataset]))
	names.append(dataset)
	if len(accuracies1[dataset]) > 1 and  len(accuracies1[dataset]) == len(accuracies2[dataset]):
		row.append(np.around(np.std(accuracies1[dataset]), 2))
		row.append(np.around(np.std(accuracies2[dataset]), 2))
		row.append(np.around(wilcoxon(accuracies1[dataset], accuracies2[dataset])[1], 3))
		p_values.append(wilcoxon(accuracies1[dataset], accuracies2[dataset])[1])
	else:
		row.append('/')
		row.append('/')
		row.append('/')
	comparison_table.append(row)

method1_accs = np.array(method1_accs)
method2_accs = np.array(method2_accs)
p_values = np.array(p_values)

comparison_table = AsciiTable(comparison_table)
print(comparison_table.table)

# Now create a scatter plot that depicts the predictive performance of
# LTS in function of the performance of GENDIS. All points in the 
# right-bottom half is where GENDIS outperforms LTS, and vice versa for the
# left-upper half of the plot. If the measures of a dataset is larger than
# 1, we perform a statistical hypothesis test. If the difference is
# significant, we mark it with a star marker.
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rc('text.latex', unicode=True)

plt.figure(figsize=(15,15))

if len(p_values) > 0:
	plt.scatter(method1_accs[p_values <= 0.05], method2_accs[p_values <= 0.05], c='k', marker='*', s=225)
	plt.scatter(method1_accs[p_values > 0.05], method2_accs[p_values > 0.05], c='k', marker='x', s=225)
else:
	plt.scatter(method1_accs, method2_accs)

limits = [np.min(np.append(method1_accs, method2_accs)) - 0.05, np.max(np.append(method1_accs, method2_accs)) + 0.05]
f = lambda x: x
range_x = np.linspace(limits[0], limits[1], 100)

plt.plot(range_x, f(range_x), dashes=[5,5], c='k')

plt.fill_between(range_x, f(range_x), alpha=0.3, color='#FFEDA0')
plt.fill_between(range_x, f(range_x), limits[1], alpha=0.3, color='#F03B20')

texts = []
for name, x, y in zip(names, method1_accs, method2_accs):
    texts.append(plt.annotate(name, (x, y), fontsize=20, alpha=0.8))

#adjust_text(texts, only_move={'text': 'xy'}, arrowprops=dict(arrowstyle="-", color='k', lw=0.5, alpha=0.4))

plt.text(np.min(np.append(method1_accs, method2_accs)) + 0.05, np.max(np.append(method1_accs, method2_accs)) - 0.05, r'\texttt{\textsc{'+method2_name+r'}}\textbf{ is better}', ha='center', va='center', fontsize=24)
plt.text(np.max(np.append(method1_accs, method2_accs)) - 0.05, np.min(np.append(method1_accs, method2_accs)) + 0.05, r'\texttt{\textsc{'+method1_name+r'}}\textbf{ is better}', ha='center', va='center', fontsize=24)

plt.xlim(limits)
plt.ylim(limits)

plt.xlabel(r'\textbf{Acc.} (\texttt{\textsc{'+method1_name+'}})', fontsize=24)
plt.ylabel(r'\textbf{Acc.} (\texttt{\textsc{'+method2_name+'}})', fontsize=24)

plt.savefig('results/scatter_plot.svg')

plt.show()