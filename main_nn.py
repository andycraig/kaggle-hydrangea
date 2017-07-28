#!/usr/bin/env python3

from train import train_nn
from utils import datetime_for_filename
import pandas as pd
import numpy as np
import pickle

# Load file of training image names and correct labels.
train_set = pd.read_csv('../data/train/train_labels.csv')
# Load file of test image names and dummy labels.
test_set = pd.read_csv('../sample_submission.csv')

print("Loading train images...")
with open('../data/train/train_nn.pickle', 'rb') as f:
	train_imgs = pickle.load(f)
print("Loading test images...")
with open('../data/test/test_nn.pickle', 'rb') as f:
	test_imgs = pickle.load(f)

train_labels = np.array(train_set['invasive'])

# Do the training.
preds_test, train_losses, test_losses = train_nn(train_imgs=train_imgs, train_labels=train_labels, test_imgs=test_imgs)


# Print some output.
print('Mean train loss: ' + str(np.mean(train_losses)))
print('Mean test loss: ' + str(np.mean(test_losses)))
# Save submission file.
test_set['invasive'] = preds_test
submission_file = '../submit_nn_' + datetime_for_filename() + '.csv'
test_set.to_csv(submission_file, index=None)
print("Saved submission file to ", submission_file)
