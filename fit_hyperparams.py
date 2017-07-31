#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from sklearn import cross_validation, svm
from sklearn.grid_search import GridSearchCV
from utils import datetime_for_filename
import pandas as pd
import numpy as np
import glob
#import cv2
import scipy
import random
import datetime
import os
import yaml
import sys
import pickle
from estimators import NN, XGBoost, TestClassifier
#import warnings
#warnings.filterwarnings('ignore')

def main(config_file, i_model):
	with open(config_file, 'r') as f:
		config = yaml.load(f)

	# Load file of training image names and correct labels.
	train_set = pd.read_csv(config['train_set'])
	train_labels = train_set['invasive'].values
	print('Loading features...')
	if i_model == 0:
		raise NotImplementedError('No hyperparams to tune for NN model.')
		model = NN()
		with open(config['train_features_nn'], 'rb') as f:
			train_features = pickle.load(f)
	elif i_model == 1:
		raise NotImplementedError('No hyperparams to tune for xgboost model.')
		model = XGBoost
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
	elif i_model == 2:
		model = svm.SVC()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		tuned_parameters = {'gamma': [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'C': [00.1, 0.1, 1, 10, 100, 1000]}
		category_in_hyperparams_file = 'svm'
	else:
		raise NotImplementedError('No hyperparams to tune for test model.')
		model = TestClassifier()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)

	print('Finding hyperparameters...')
	clf = GridSearchCV(model, tuned_parameters, cv=5)
	clf.fit(train_features, train_labels)
	print('Found best hyperparams:')
	print(clf.best_params_)

	# Write chosen hyperparams to file.
	with open(config['hyperparams_file'], 'r') as f:
		hyperparams = yaml.load(f)
	# Put grid search best params in hyperparams dict.
	for key in clf.best_params_:
		hyperparams[category_in_hyperparams_file][key] = clf.best_params_[key]
	# Save hyperparams.
	with open(config['hyperparams_file'], 'w') as f:
		yaml.dump(hyperparams, f)
	print('Wrote best params to ' + str(config['hyperparams_file']))

if __name__ == "__main__":
	main(sys.argv[1], int(sys.argv[2]))
