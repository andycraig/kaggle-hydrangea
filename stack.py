#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy
import random
import datetime
import os
import yaml
import sys

def main(config_file, i_model, fold):
	with open(config_file, 'r') as f:
		config = yaml.load(f)

	# Load file of training image names and correct labels.
	train_set = pd.read_csv(config['train_set'])
	train_labels = train_set['invasive'].values
	# Load file of test image names and dummy labels.
	test_set = pd.read_csv(config['test_set'])

	# Fit stacking model predicting 'invasive' from model columns.
	model_cols = ['M' + str(x) for x in range(config['n_models'])]
	S = LogisticRegression()
	S.fit(X=train_set[model_cols], y=train_labels)
	# Make predictions based on model columns of test set.
	pred = S.predict(X=test_set[model_cols])

	# Write these predictions to submit file.
