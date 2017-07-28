import sklearn
import xgboost as xgb
from model import get_model
import pandas as pd
import numpy as np
from tensorflow.contrib import keras
from sklearn import cross_validation

def train_gbt(train, test, test_features, train_features, train_labels, n_folds = 8):
	n_train = len(train)
	y_pred = np.zeros(test_features.shape[0])
	xgtest = xgb.DMatrix(test_features)
	score = 0
	kf = cross_validation.KFold(n_train, n_folds=n_folds, shuffle=False)

	for i, (trn_index, val_index) in enumerate(kf):

	    xgtrain = xgb.DMatrix(train_features[trn_index], label=train_labels[trn_index])
	    xgvalid = xgb.DMatrix(train_features[val_index], label=train_labels[val_index])

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

	    watchlist = [ (xgtrain,'train'), (xgvalid, 'valid') ]
	    model = xgb.train(list(params.items()), xgtrain, 5000, watchlist,
	                      early_stopping_rounds=25, verbose_eval = 50)
						  score += model.best_score / n_folds

	    y_pred += model.predict(xgtest,ntree_limit=model.best_ntree_limit) / n_folds
		return y_pred, score

def train_nn(train_imgs, train_labels, test_imgs,
 			n_folds=8,
			epochs = 1000 # Originally 1000.
			batch_size = 64 # Originally 64.
			max_steps_per_epoch = np.inf # Originally np.inf):
	n_train = len(train_labels)
	# Define n-fold cross validation splits and evaluation metric.
	print("Training with ", n_folds, " folds.")
	kf = cross_validation.KFold(, n_folds=n_folds, shuffle=True)
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
		x_tr = train_imgs[i_train,:,:,:]
		y_tr = train_labels[i_train]
		x_val = train_imgs[i_test,:,:,:]
		y_val = train_labels[i_test]
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
		datagen.fit(x_tr)

		model = get_model()
		earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
		# Do the fit.
		model.fit_generator(datagen.flow(x_tr, y_tr, batch_size=batch_size),
			validation_data=(x_val, y_val), callbacks=[earlystop],
			steps_per_epoch=min(max_steps_per_epoch, len(train_imgs) / batch_size),
			epochs=epochs,
			verbose=2)

		# Add the losses for this fold.
		train_losses.append(eval_fun(y_tr, model.predict(x_tr)[:,0]))
		test_losses.append(eval_fun(y_val, model.predict(x_val)[:,0]))
		print(str(i) + ': Train: ' + str(train_losses[-1]) + ' Val: ' + str(test_losses[-1]))

		# Add the prediction contribution from this fold.
		preds_test += model.predict(test_imgs)[:,0] / n_folds

		return preds_test, train_losses, test_losses
