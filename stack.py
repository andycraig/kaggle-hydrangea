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
from utils import datetime_for_filename

def main(config_file):
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
	predictions = S.predict(X=test_set[model_cols])
	test_set['invasive'] = predictions
	# Write these predictions to submit file.
	submit_file = config['submit_prefix'] + '_' + datetime_for_filename() + '.csv'
	test_set.to_csv(submit_file, header=True, index=None)
	print("Saved submit file to " + submit_file)

if __name__ == "__main__":
	main(sys.argv[1])
