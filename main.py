#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from sklearn import cross_validation, svm
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
	# Load file of test image names and dummy labels.
	test_set = pd.read_csv(config['test_set'])

	y = train_set['invasive'].values

	print('Loading features...')
	if i_model == 0:
		model = NN()
		with open(config['train_features_nn'], 'rb') as f:
			train_features = pickle.load(f)
		with open(config['test_features_nn'], 'rb') as f:
			test_features = pickle.load(f)
	elif i_model == 1:
		model = XGBoost
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)
	elif i_model == 2:
		model = svm.SVC()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)
	else:
		model = TestClassifier()
		train_features = pd.read_csv(config['train_features_gbt'], header=None)
		test_features = pd.read_csv(config['test_features_gbt'], header=None)

	print('Fitting...')
	n_estimators = 8
	max_samples = 1.0 * (n_estimators - 1) / n_estimators
	clf = BaggingClassifier(model,
							n_estimators=n_estimators,
							max_samples=max_samples,
							max_features=1)
	clf.fit(X=train_features, y=train_labels)
	print('Predicting...')
	y_pred = clf.predict(test_features)

	now = datetime.datetime.now()
	test_set['invasive'] = y_pred
	submission_file = config['submit_prefix'] + '_' + str(i_model) + '_' + datetime_for_filename() + '.csv'
	test_set[['name','invasive']].to_csv(submission_file, index=None)
	print("Saved submission file to ", submission_file)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
