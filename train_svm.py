#!/usr/bin/env python3

from utils import load_from_pickle_if_possible
from utils import datetime_for_filename
from PIL import Image
#import tqdm
import sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn import cross_validation
from sklearn import svm
import yaml
import sys

max_n_imgs = np.inf # np.inf to use all data.
n_folds = 8 # Originally 8.
img_x, img_y, n_channels = 128, 128, 3

# Load file of training image names and correct labels.
train_set = pd.read_csv('../data/train/train_labels.csv')
# Load file of test image names and dummy labels.
test_set = pd.read_csv('../sample_submission.csv')

n_train = min(max_n_imgs, len(train_set))
n_test = min(max_n_imgs, len(test_set))
print("Using ", n_train, "/", len(train_set), " to train.")
print("Using ", n_test, "/", len(test_set), " to test.")
train_set = train_set.iloc[:n_train]
test_set = test_set.iloc[:n_test]

# Load the train and test images.
# Helper function.
def load_img_matrix(n_imgs, folder_name, names_and_labels):
	imgs_matrix = np.zeros([n_imgs, img_x * img_y * n_channels])
	for i, img_name in enumerate(names_and_labels['name'].iloc[:]):
		img_src = '../data/' + folder_name + '/img/' + str(img_name) + '.jpg'
		img = Image.open(img_src).resize((img_x, img_y)) # was 128x128
		# Resize and change pixel range from 0-255 to 0-1.
		imgs_matrix[i,:] = np.array(img.getdata()).reshape([img_x * img_y * n_channels]) / 255
	return imgs_matrix

# Load train images from pickle if possible.
print("Loading train images...")
train_imgs = load_from_pickle_if_possible('../data/train/train_svm.pickle',
					lambda : load_img_matrix(n_train, 'train', train_set))
print("Loading test images...")
test_imgs = load_from_pickle_if_possible('../data/test/test_svm.pickle',
					lambda : load_img_matrix(n_test, 'test', test_set))

train_labels = np.array(train_set['invasive'].iloc[:])

# Define n-fold cross validation splits and evaluation metric.
print("Training with ", n_folds, " folds.")
kf = cross_validation.KFold(n_train, n_folds=n_folds, shuffle=True)
eval_fun = sklearn.metrics.roc_auc_score

# Initialise prediction placeholders.
preds_test = np.zeros(len(test_imgs), dtype=np.float)
train_losses, test_losses = [], []

for i, (i_train, i_test) in enumerate(kf):
	print("Starting iteration ", i)
	# Elements returned by kf.split are arrays of length 1 that contain
	# the array of indices, so pull out the arrays.
	#i_train = i_train_list_of_array[0]
	#i_test = i_test_list_of_array[0]
	# Get the train/validation elements for this split.
	x_tr = train_imgs[i_train,:]
	y_tr = train_labels[i_train]
	x_val = train_imgs[i_test,:]
	y_val = train_labels[i_test]

	# Train the SVM.
	model = svm.SVC()
	model.fit(x_tr, y_tr)

	# Add the losses for this fold.
	train_losses.append(eval_fun(y_tr, model.predict(x_tr)))
	test_losses.append(eval_fun(y_val, model.predict(x_val)))

	# Add the prediction contribution from this fold.
	preds_test += model.predict(test_imgs) / n_folds

	print(str(i) + ': Train: ' + str(train_losses[-1]) + ' Val: ' + str(test_losses[-1]))

# Print some output.
print('Mean train loss: ' + str(np.mean(train_losses)))
print('Mean test loss: ' + str(np.mean(test_losses)))
# Save submission file.
test_set['invasive'] = preds_test
submission_file = '../submit_svm_' + datetime_for_filename() + '.csv'
test_set.to_csv(submission_file, index=None)
print("Saved submission file to ", submission_file)
