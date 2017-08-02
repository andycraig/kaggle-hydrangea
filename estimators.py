# Estimators

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import sklearn
from model_nn import get_model
import pandas as pd
import numpy as np
from tensorflow.contrib import keras
from sklearn import cross_validation
try:
	import xgboost as xgb
except ImportError as e:
	print("Couldn't find module xgboost. Won't be able to use xgboost.")

img_x = 128
img_y = 128
n_channels = 3

# Test classifier
class TestClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, demo_param='demo'):
		self.demo_param = demo_param

	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y
		 # Return the classifier
		return self

	def predict(self, X):

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
		return self.y_[closest]

# NN classifier
class NN(BaseEstimator, ClassifierMixin):

	def __init__(self, demo_param='demo'):
		self.demo_param = demo_param

	def fit(self, X, y):
		# Check that X and y have correct shape.
		X, y = check_X_y(X, y)
		# Store the classes seen during fit.
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y
		# self.X_ is an n x (128*128*3) matrices.
		# For CNN, we want an nx128x128x3 matrix.
		self.X_4Dmatrix = self.X_.reshape([-1, img_x, img_y, n_channels])
		datagen = keras.preprocessing.image.ImageDataGenerator(
			# featurewise_center = True,
			rotation_range = 30,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			# zca_whitening = True,
			shear_range = 0.2,
			zoom_range = 0.2,
			horizontal_flip = True,
			vertical_flip = True,
			fill_mode = 'nearest')
		datagen.fit(self.X_4Dmatrix)

		self.model = get_model()
		earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
		# Do the fit.
		batch_size = 64
		epochs = 1000
		self.model.fit_generator(datagen.flow(self.X_4Dmatrix, self.y_, batch_size=batch_size),
			callbacks=[earlystop],
			steps_per_epoch=(len(y) / batch_size),
			epochs=epochs,
			verbose=2)
		# Return the classifier
		return self

	def predict(self, X):

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		# Convert X (list of matrices) to a matrix before sending to model.predict().
		predictions = self.model.predict(X.reshape([len(X), img_x, img_y, n_channels]))
		return predictions

# XGBoost classifier
class XGBoost(BaseEstimator, ClassifierMixin):

	def __init__(self, demo_param='demo'):
		self.demo_param = demo_param

	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y
		# Return the classifier
		xgtrain = xgb.DMatrix(self.X_, label=self.y_)

		params = {
			'eta': 0.05, #0.03
			'silent': 1,
			'verbose_eval': True,
			'verbose': False,
			'seed': 4
		}
		params['objective'] = 'binary:logistic'
		params['eval_metric'] = "auc"
		params['min_child_weight'] = 15
		params['cosample_bytree'] = 0.8
		params['cosample_bylevel'] = 0.9
		params['max_depth'] = 4
		params['subsample'] = 0.9
		params['max_delta_step'] = 10
		params['gamma'] = 1
		params['alpha'] = 0
		params['lambda'] = 1
		#params['base_score'] =  0.63

		watchlist = [(xgtrain,'train')]
		self.model = xgb.train(list(params.items()), xgtrain, 5000, watchlist,
						early_stopping_rounds=25, verbose_eval = 50)

		return self

	def predict(self, X):

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		xgtest = xgb.DMatrix(X)
		predictions = self.model.predict(xgtest,ntree_limit=self.model.best_ntree_limit)
		return predictions
