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
from terminaltables import AsciiTable

import Orange
import matplotlib.pyplot as plt

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
datasets = set([x.split('_')[0] for x in os.listdir(DIR) 
                if x != '.keep'])

ranks_per_technique = defaultdict(list)

for technique in ['voting']:
    ranks_gendis = []
    ranks_ls = []
    ranks_st = []
    ranks_fs = []
    table_data = [['Dataset', '#Classes', 'TS Length', '#Train', '#Test', 'Accuracy GENDIS {}'.format(technique), 'Accuracy ST', 'T-Test ST', 'Difference ST', 'Accuracy LS', 'T-Test LS', 'Difference LS']]
    for dataset in datasets:
        print(dataset)
        glob_path = DIR + '{}_{}*{}''_proba.csv'
        method1_files = glob.glob(glob_path.format(dataset, method1, technique))
        X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

        if X_train is None: 
            print('Training data could not be loaded')
            continue

        if not method1_files:
            print('No files found for this dataset')
            continue

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

            accuracies1[dataset].append((accuracy_score(ground_truth, preds), timestamp))

        gendis_accuracy = np.mean([x[0] for x in accuracies1[dataset]])
        ls_accuracy = np.mean(ls_accuracies.loc[dataset])
        st_accuracy = np.mean(st_accuracies.loc[dataset])
        T, p = ttest_ind([x[0] for x in accuracies1[dataset]], st_accuracies.loc[dataset])
        if p > 0.05:
            st_mwu_significance = '\\'
        else:
            if gendis_accuracy > st_accuracy:
                st_mwu_significance = '+'
            else:
                st_mwu_significance = '-'

        T, p = ttest_ind([x[0] for x in accuracies1[dataset]], ls_accuracies.loc[dataset])
        if p > 0.05:
            ls_mwu_significance = '\\'
        else:
            if gendis_accuracy > ls_accuracy:
                ls_mwu_significance = '+'
            else:
                ls_mwu_significance = '-'

        # T, p = ttest_ind(accuracies1[dataset], st_accuracies.loc[dataset])
        # if p > 0.05:
        #     st_tt_significance = '\\'
        # else:
        #     if gendis_accuracy > st_accuracy:
        #         st_tt_significance = '+'
        #     else:
        #         st_tt_significance = '-'
                
        # T, p = ttest_ind(accuracies1[dataset], ls_accuracies.loc[dataset])
        # if p > 0.05:
        #     ls_tt_significance = '\\'
        # else:
        #     if gendis_accuracy > ls_accuracy:
        #         ls_tt_significance = '+'
        #     else:
        #         ls_tt_significance = '-'


        table_data.append([dataset, len(set(ground_truth)), len(X_train[0]), len(X_train), len(X_test), np.around(gendis_accuracy, 4), 
                           np.around(st_accuracy, 4), st_mwu_significance, np.around(gendis_accuracy - st_accuracy, 4), 
                           np.around(ls_accuracy, 4), ls_mwu_significance, np.around(gendis_accuracy - ls_accuracy, 4)])

        gendis_vals.append(gendis_accuracy)
        st_vals.append(st_accuracy)
        ls_vals.append(ls_accuracy)

        #results = data_loader.baseline_accuracy(dataset)[dataset]
        #results_shapelets = {}
        #for method in ['LS', 'ST', 'FS']:
        #    results_shapelets[method] = results[method]
        #results = results_shapelets
        results = {}
        results['LS'] = np.mean(ls_accuracies.loc[dataset])
        results['FS'] = np.mean(fs_accuracies.loc[dataset])
        results['ST'] = np.mean(st_accuracies.loc[dataset])
        results['GENDIS'] = np.mean([x[0] for x in accuracies1[dataset]])
        sorted_results = sorted(results.items(), key=lambda x: -x[1])

        for item, rank in zip(sorted_results, len(sorted_results) - rankdata([x[1] for x in sorted_results])):
            if 'GENDIS' in item[0]: ranks_gendis.append((dataset, rank))
            if item[0] == 'LS': ranks_ls.append((dataset, rank))
            if item[0] == 'ST': ranks_st.append((dataset, rank))
            if item[0] == 'FS': ranks_fs.append((dataset, rank))

            if 'GENDIS' in item[0]:
                print(rank, item[0], item[1], sorted(accuracies1[dataset]))
            else:
                print(rank, item[0], item[1])

            ranks_per_technique[item[0]].append(rank)

        print()
        print()

    print('GENDIS {} Ranks:'.format(technique), ranks_gendis, np.mean([x[1] for x in ranks_gendis]))
    print('LS Ranks:', ranks_ls, np.mean([x[1] for x in ranks_ls]), [x[1] < y[1] for x, y in zip(ranks_gendis, ranks_ls)])
    print('ST Ranks:', ranks_st, np.mean([x[1] for x in ranks_st]), [x[1] < y[1] for x, y in zip(ranks_gendis, ranks_st)])
    print('FS Ranks:', ranks_fs, np.mean([x[1] for x in ranks_fs]), [x[1] < y[1] for x, y in zip(ranks_gendis, ranks_fs)])

    table_head = table_data[0]
    table_data = table_data[1:]
    table_data = sorted(table_data, key=lambda x: x[8])
    table_data = [table_head] + table_data

    result_table = AsciiTable(table_data, 'GENDIS vs other shapelet methods')
    print(result_table.table)


print(np.mean(gendis_vals), np.std(gendis_vals))
print(np.mean(st_vals), np.std(st_vals))
print(np.mean(ls_vals), np.std(ls_vals))
print(wilcoxon(gendis_vals, st_vals))
print(wilcoxon(gendis_vals, ls_vals))
print(wilcoxon(ls_vals, st_vals))


avg_ranks = [np.mean(x) for x in ranks_per_technique.values()]
names = list(ranks_per_technique.keys())

print(avg_ranks, len(ranks_per_technique['GENDIS']) )


from statsmodels.stats.libqsturng import qsturng

# Sources for the formula for cd:
# https://mlr.mlr-org.com/reference/generateCritDifferencesData.html
# https://scikit-learn-general.narkive.com/ebgsIj5T/critical-difference-diagram

plt.figure()
alpha = 0.1
n_methods = len(avg_ranks)
n_datasets = len(ranks_per_technique['GENDIS'])
q_alpha = qsturng(1 - alpha, n_methods, np.inf) / np.sqrt(2)
cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
Orange.evaluation.graph_ranks(avg_ranks, names, cd=cd)
plt.gcf().set_size_inches(10, 3)
plt.subplots_adjust(left=0.25)
plt.savefig('results/cd_diagram_1.svg', format='svg')

plt.figure()
alpha = 0.05
n_methods = len(avg_ranks)
n_datasets = len(ranks_per_technique['GENDIS'])
q_alpha = qsturng(1 - alpha, n_methods, np.inf) / np.sqrt(2)
cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
Orange.evaluation.graph_ranks(avg_ranks, names, cd=cd)
plt.gcf().set_size_inches(10, 3)
plt.subplots_adjust(left=0.25)
plt.savefig('results/cd_diagram_05.svg', format='svg')

plt.figure()
alpha = 0.01
n_methods = len(avg_ranks)
n_datasets = len(ranks_per_technique['GENDIS'])
q_alpha = qsturng(1 - alpha, n_methods, np.inf) / np.sqrt(2)
cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
Orange.evaluation.graph_ranks(avg_ranks, names, cd=cd)
plt.gcf().set_size_inches(10, 3)
plt.subplots_adjust(left=0.25)
plt.savefig('results/cd_diagram_01.svg', format='svg')


"""
CBF: (0.9333333333333333, '1552145432')
Coffee: (0.8928571428571429, '1552292141')
Trace: (0.99, '1552316958')
TwoLeadECG: (0.8630377524143986, '1552096875')
DistalPhalanxOutlineCorrect: (0.782608695652174, '1552088336')
Plane: (0.9619047619047619, '1552171277')
SonyAIBORobotSurface2: (0.7869884575026233, '1552035609')
ToeSegmentation1: (0.8859649122807017, '1552271601')
ArrowHead: (0.7085714285714285, '1552225440')
DistalPhalanxOutlineAgeGroup: (0.7769784172661871, '1552094529'),
"""