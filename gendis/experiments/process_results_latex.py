# Standard Library
import os
import glob
from collections import defaultdict

# Data wrangling
import pandas as pd
import numpy as np

# Calculate accuracies
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the datasets
from tslearn.datasets import UCR_UEA_datasets

# Paired Wilcoxon Test
from scipy.stats import mannwhitneyu, rankdata, ttest_ind, wilcoxon
from tabulate import tabulate

import Orange
import matplotlib.pyplot as plt

import ast
import re

# Comment this out if you want to process results of dependent vs independent
DIR = 'results/genetic/'
method1 = 'genetic'
method1_name = 'GENDIS'

accuracies1 = defaultdict(list)
st_accuracies = pd.read_csv("AllSplits/ST.csv", index_col=0, header=None)
ls_accuracies = pd.read_csv("AllSplits/LS.csv", index_col=0, header=None)
fs_accuracies = pd.read_csv("AllSplits/FS.csv", index_col=0, header=None)

gendis_vals = []
st_vals = []
ls_vals = []

data_loader = UCR_UEA_datasets()

# Iterate over files in directory, process the predictions (_proba) and save 
# them in a dict. Afterwards, print a table with aggregated results & create 
# scatter plot (stat test if |values| > 1)
datasets = list(set([x.split('_')[0] for x in os.listdir(DIR) 
                     if x != '.keep']))

table_data = [['Dataset', '\#Classes', 'TS Length', '\#Train', '\#Test',
               'GENDIS', 'ST', 'LTS', 'FS', '\#Shapelets']]
for dataset in datasets:
    print(dataset)
    glob_path = DIR + '{}_{}*{}''_proba.csv'
    method1_files = glob.glob(glob_path.format(dataset, method1, 'voting'))
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

    accuracies = []
    nr_shaps = []

    # Iterate over files of method 1 and calculate accuracy
    for file in method1_files:
        # Get the corresponding ground_truth
        timestamp = file.split('_')[-3]
        preds = np.argmax(pd.read_csv(file, index_col=[0]).values, axis=1)

        # WARNING: I fucked up... I stored y_train at 'test' and y_test and 'train'
        # It should be fixed when the new results arrive, so this needs to be reverted to test!
        ground_truth = pd.read_csv(DIR + '{}_ground_truth_test_{}.csv'.format(dataset, timestamp))['label']
        if len(ground_truth) != len(preds):
            ground_truth = pd.read_csv(DIR + '{}_ground_truth_train_{}.csv'.format(dataset, timestamp))['label']
        if len(ground_truth) != len(preds):
            print('Skipping {}...'.format(dataset))
            continue
        
        # Map classes to [0, ..., C-1]
        map_dict = {}
        for j, c in enumerate(sorted(set(ground_truth))):
            map_dict[c] = j
        ground_truth = pd.Series(ground_truth).map(map_dict).values

        accuracies.append((accuracy_score(ground_truth, preds), timestamp))

        shapelet_file = DIR + '{}_{}_shapelets_{}.txt'.format(dataset, 'genetic', timestamp)
        shapelets = []
        with open(shapelet_file, 'r') as ifp:
            for line in ifp.read().split(']\n'):
                if len(line):
                    line = line.replace('...', '')
                    proc_line = re.sub(r'\s+', ',', line)[:-1] + ']'
                    proc_line = proc_line.replace('[,', '[')
                    shapelets.append(np.array(ast.literal_eval(proc_line)))

        nr_shaps.append(len(shapelets))

    gendis_accuracy = np.mean([x[0] for x in accuracies])
    ls_accuracy = np.mean(ls_accuracies.loc[dataset])
    st_accuracy = np.mean(st_accuracies.loc[dataset])
    fs_accuracy = np.mean(fs_accuracies.loc[dataset])

    table_row = [dataset, len(set(ground_truth)), len(X_train[0]), len(X_train), len(X_test)]

    best_algorithm = np.argmax([gendis_accuracy, st_accuracy, ls_accuracy, fs_accuracy])

    if best_algorithm == 0:
        p_values = [ttest_ind([x[0] for x in accuracies], ls_accuracies.loc[dataset])[1],
                    ttest_ind([x[0] for x in accuracies], st_accuracies.loc[dataset])[1],
                    ttest_ind([x[0] for x in accuracies], fs_accuracies.loc[dataset])[1]]

        if max(p_values) < 0.05:
            table_row.append('\\textbf{{{}}}'.format(np.around(100*gendis_accuracy, 1)))
        else:
            table_row.append(np.around(100*gendis_accuracy, 1))
    else:
        table_row.append(np.around(100*gendis_accuracy, 1))

    if best_algorithm == 1:
        p_values = [ttest_ind(st_accuracies.loc[dataset], [x[0] for x in accuracies])[1],
                    ttest_ind(st_accuracies.loc[dataset], ls_accuracies.loc[dataset])[1],
                    ttest_ind(st_accuracies.loc[dataset], fs_accuracies.loc[dataset])[1]]

        if max(p_values) < 0.05:
            table_row.append('\\textbf{{{}}}'.format(np.around(100*st_accuracy, 1)))
        else:
            table_row.append(np.around(100*st_accuracy, 1))
    else:
        table_row.append(np.around(100*st_accuracy, 1))

    if best_algorithm == 2:
        p_values = [ttest_ind(ls_accuracies.loc[dataset], [x[0] for x in accuracies])[1],
                    ttest_ind(ls_accuracies.loc[dataset], st_accuracies.loc[dataset])[1],
                    ttest_ind(ls_accuracies.loc[dataset], fs_accuracies.loc[dataset])[1]]

        if max(p_values) < 0.05:
            table_row.append('\\textbf{{{}}}'.format(np.around(100*ls_accuracy, 1)))
        else:
            table_row.append(np.around(100*ls_accuracy, 1))
    else:
        table_row.append(np.around(100*ls_accuracy, 1))

    if best_algorithm == 3:
        p_values = [ttest_ind(fs_accuracies.loc[dataset], [x[0] for x in accuracies])[1],
                    ttest_ind(fs_accuracies.loc[dataset], st_accuracies.loc[dataset])[1],
                    ttest_ind(fs_accuracies.loc[dataset], ls_accuracies.loc[dataset])[1]]

        if max(p_values) < 0.05:
            table_row.append('\\textbf{{{}}}'.format(np.around(100*fs_accuracy, 1)))
        else:
            table_row.append(np.around(100*fs_accuracy, 1))
    else:
        table_row.append(np.around(100*fs_accuracy, 1))

    table_row.append(int(np.mean(nr_shaps)))

    table_data.append(table_row)

table_data_df = pd.DataFrame(table_data[1:], columns=['Dataset', '#Classes', 'TS Length', '#Train', '#Test', 
               									      'GENDIS', 'ST', 'LTS', 'FS', '#Shapelets'])
table_data_df = table_data_df.dropna()
table_data_df['GENDIS'] = table_data_df['GENDIS'].apply(lambda x: str(x).strip('\\textbf{').strip('}')).astype(float)
table_data_df['ST'] = table_data_df['ST'].apply(lambda x: str(x).strip('\\textbf{').strip('}')).astype(float)
table_data_df['LTS'] = table_data_df['LTS'].apply(lambda x: str(x).strip('\\textbf{').strip('}')).astype(float)
table_data_df['FS'] = table_data_df['FS'].apply(lambda x: str(x).strip('\\textbf{').strip('}')).astype(float)
table_data_df[['Dataset', 'GENDIS', 'ST', 'LTS', 'FS']].to_csv('results/results.csv', index=False)

table_data = [table_data[0]] + sorted(table_data[1:], key=lambda x: x[0])

chunk_size = len(table_data) // 2
for i in range(2):
	tabulate.LATEX_ESCAPE_RULES={}
	if i < 2:
		table_chunk = [table_data[0]] + table_data[i*chunk_size + 1:(i + 1)*chunk_size + 1]
	else:
		table_chunk = [table_data[0]] + table_data[i*chunk_size + 1:]

	result_table = tabulate(table_chunk, tablefmt='latex_raw')
	print(result_table)