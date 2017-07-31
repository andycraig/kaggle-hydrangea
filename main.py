#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from sklearn import cross_validation, svm
from sklearn.ensemble import BaggingClassifier
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

def main(config_file, i_model, fold):
	with open(config_file, 'r') as f:
		config = yaml.load(f)
	with open(config_file['hyperparams_file'], 'r') as f:
		hyperparams = yaml.load(f)

	# Load file of training image names and correct labels.
	train_set = pd.read_csv(config['train_set'])
	train_labels = train_set['invasive'].values
	# Load file of test image names and dummy labels.
	test_set = pd.read_csv(config['test_set'])

	print('Loading features...')
	if i_model == 0:
		model = NN()
		with open(config['train_features_nn'], 'rb') as f:
			train_features = pickle.load(f)
		with open(config['test_features_nn'], 'rb') as f:
			test_features = pickle.load(f)
	elif i_model == 1:
		model = XGBoost()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)
	elif i_model == 2:
		model = svm.SVC(**hyperparams['svm'])
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)
	else:
		model = TestClassifier()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)


	if fold != None:
		mask_fold_train = train_set['fold'] == fold
		mask_fold_val = not mask_fold_train
	else:
		mask_fold_train = np.ones([len(train_labels), 1], dtype=bool)

	print('Fitting...')
	n_estimators = 8
	max_samples = 1.0 * (n_estimators - 1) / n_estimators
	clf = BaggingClassifier(model,
							n_estimators=n_estimators,
							max_samples=max_samples,
							max_features=1)
	clf.fit(X=train_features[mask_fold_train], y=train_labels[mask_fold_train])

	if fold != None:
		model_col_name = 'M' + str(i_model)
		train_set[model_col_name].loc[mask_fold_val] = clf.predict_proba(train_features[mask_fold_val])
		train_set.to_csv(config['train_set'], index=False)
		print('Added predictions for model ' + str(i_model) + ', fold ' + str(fold) + ' to column ' + model_col_name + ' of ' + config['train_set'])

	print('Predicting...')
	y_pred = clf.predict_proba(test_features)
	now = datetime.datetime.now()
	test_set['invasive'] = y_pred
	submission_file = config['submit_prefix'] + '_' + str(i_model) + '_' + datetime_for_filename() + '.csv'
	test_set[['name','invasive']].to_csv(submission_file, index=None)
	print("Saved submission file to ", submission_file)

if __name__ == "__main__":
	if len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], fold=sys.argv[3])
	else:
		main(sys.argv[1], sys.argv[2], fold=None)
