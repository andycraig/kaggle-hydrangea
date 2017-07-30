#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from sklearn import cross_validation, svm, GridSearchCV
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
		tuned_parameters = {'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
		scores = ['precision', 'recall']
		category_in_hyperparams_file = 'svm'
	else:
		raise NotImplementedError('No hyperparams to tune for test model.')
		model = TestClassifier()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)

	print('Finding hyperparameters...')
	clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
	clf.fit(X_train, y_train)

	# Write chosen hyperparams to file.
	with open(config['hyperparams_file'], 'r') as f:
		hyperparams = yaml.load(f)
	# Put grid search best params in hyperparams dict.
	for key in clf.best_params_:
		hyperparams[category_in_hyperparams_file][key] = clf.best_params[key]
	# Save hyperparams.
	with open(config['hyperparams_file'], 'w') as f:
		yaml.write(hyperparams, f)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
