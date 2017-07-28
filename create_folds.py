#!/usr/bin/env python3

import pandas as pd
import random
import numpy as np
import yaml
import sys

def main(config_file):
	with open(config_file, 'r') as f:
		config = yaml.load(f)

	# Set seed.
	random.seed(4)
	np.random.seed(4)

	# Load training CSV.
	df = pd.read_csv(config['train_set_clean'])
	n_train = len(df)

	# Generate folds.
	n_folds = 5
	# Make sufficient copies of [0 ... n_folds-1] so that there are n_train elements.
	unpermuted_fold_ids = (list(range(n_folds)) * int(np.ceil(n_train / n_folds)))[0:n_train]
	permuted_fold_ids = np.random.permutation(unpermuted_fold_ids)

	# Write folds into CSV.
	df['fold'] = permuted_fold_ids
	df.to_csv(config['train_set'], index=False)

if __name__ == "__main__":
	main(sys.argv[1])
