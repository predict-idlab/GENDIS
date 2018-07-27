import os
import pandas as pd

"""Iterate over all the csv files in the directory,
load it in a DataFrame with pandas and create a list of 
dicts:
[{
	'name': str,
	'n_samples': int,
	'n_features': int,
	'n_classes': int,
	'data': pd.DataFrame
}]"""

def load_data():
	data = []
	this_directory = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'all'
	for file in os.listdir(this_directory):
		try:
			if file.endswith('csv'):
				df = pd.read_csv(this_directory+os.sep+file)	
				if 'target' in df.columns:
					data.append({
						'name': file.split('.')[0],
						'n_samples': len(df),
						'n_features': len(df.columns) - 1,
						'n_classes': len(set(df['target'])),
						'data': df
					})
		except:
			print('Loading {} failed...'.format(file))
	return data

def load_data_train_test():
	data = []
	this_directory = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'partitioned'
	for file in os.listdir(this_directory):
		try:
			if os.path.isdir(this_directory+os.sep+file):
				train_df = pd.read_csv(this_directory+os.sep+file+os.sep+file+'_train.csv')
				test_df = pd.read_csv(this_directory+os.sep+file+os.sep+file+'_test.csv')
				if 'target' in train_df.columns:
					data.append({
						'train': {
							'name': file.split('_train.csv')[0],
							'n_samples': len(train_df),
							'n_features': len(train_df.columns) - 1,
							'n_classes': len(set(train_df['target'])),
							'data_path': this_directory + os.sep + file + os.sep + file + '_train.csv'
						},
						'test': {
							'name': file.split('_test.csv')[0],
							'n_samples': len(test_df),
							'n_features': len(test_df.columns) - 1,
							'n_classes': len(set(test_df['target'])),
							'data_path': this_directory + os.sep + file + os.sep + file + '_test.csv'
						}
					})
		except:
			print('Loading {} failed...'.format(file))
	return data
