#!/usr/bin/env python3

import pandas as pd
import random
import numpy as np

# Set seed.
random.seed(4)
np.random.seed(4)

# Load training CSV.
df = pd.read_csv('../data/train/train_labels.csv.bak')
n_train = len(df)

# Generate folds.
n_folds = 5
# Make sufficient copies of [0 ... n_folds-1] so that there are n_train elements.
unpermuted_fold_ids = (list(range(n_folds)) * int(np.ceil(n_train / n_folds)))[0:n_train]
permuted_fold_ids = np.random.permutation(unpermuted_fold_ids)

# Write folds into CSV.
df['fold'] = permuted_fold_ids
df.to_csv('../data/train/train_labels.csv', index=False)