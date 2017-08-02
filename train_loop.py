#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import os, sys, yaml
from train import main
#import warnings
#warnings.filterwarnings('ignore')

if __name__ == "__main__":
	with open(sys.argv[1], 'r') as f:
		config = yaml.load(f)

	# Train each fold.
	print(config['n_folds'])
	for i_fold in range(config['n_folds']):
		main(sys.argv[1], int(sys.argv[2]), i_fold)

	# Train all.
	main(sys.argv[1], int(sys.argv[2]), fold=None)
