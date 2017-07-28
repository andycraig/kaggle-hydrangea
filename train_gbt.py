#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from PIL import Image, ImageStat
#from tqdm import tqdm
from sklearn import cross_validation
#import xgboost as xgb
import pandas as pd
import numpy as np
import glob
#import cv2
import scipy
import random
import datetime
import os
#import warnings
#warnings.filterwarnings('ignore')

n_folds = 8

random.seed(4)
np.random.seed(4)

def get_features(path):
    try:
        ft = []
        img = Image.open(path)
        im_stats_ = ImageStat.Stat(img)
        ft += im_stats_.sum
        ft += im_stats_.mean
        ft += im_stats_.rms
        ft += im_stats_.var
        ft += im_stats_.stddev
        img = np.array(img)[:,:,:3]
        ft += [scipy.stats.kurtosis(img[:,:,0].ravel())]
        ft += [scipy.stats.kurtosis(img[:,:,1].ravel())]
        ft += [scipy.stats.kurtosis(img[:,:,2].ravel())]
        ft += [scipy.stats.skew(img[:,:,0].ravel())]
        ft += [scipy.stats.skew(img[:,:,1].ravel())]
        ft += [scipy.stats.skew(img[:,:,2].ravel())]
        # bw = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        # ft += list(cv2.HuMoments(cv2.moments(bw)).flatten())

		# Histograms:
		# m.histogram() # This might be the bands concatenated.
		# Get the bands:
		#im.split() ⇒ sequence
		#Returns a tuple of individual image bands from an image. For example, splitting an “RGB” image creates three new images each containing a copy of one of the original bands (red, green, blue).
        # ft += list(cv2.calcHist([bw],[0],None,[64],[0,256]).flatten()) #bw
        # ft += list(cv2.calcHist([img],[0],None,[64],[0,256]).flatten()) #r
        # ft += list(cv2.calcHist([img],[1],None,[64],[0,256]).flatten()) #g
        # ft += list(cv2.calcHist([img],[2],None,[64],[0,256]).flatten()) #b
        # m, s = cv2.meanStdDev(img) #mean and standard deviation
        # ft += list(m.ravel())
        # ft += list(s.ravel())

		#Laplacian:
		#scipy.ndimage.filters.laplace¶
        # ft += [cv2.Laplacian(bw, cv2.CV_64F).var()]
        # ft += [cv2.Laplacian(img, cv2.CV_64F).var()]

        # Sobel:
        #image = Image.open('your_image.png')
        #image = image.filter(ImageFilter.FIND_EDGES)

        # ft += [cv2.Sobel(bw,cv2.CV_64F,1,0,ksize=5).var()]
        # ft += [cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=5).var()]
        # ft += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]
        # ft += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]
    except:
        print(path)
    return ft

def load_img(paths):
    imf_d = {}
    for f in paths:
        imf_d[f] = get_features(f)
    fdata = [imf_d[f] for f in paths]
    return fdata

print('Loading Train Data')
in_path = '../data/'
train = pd.read_csv(in_path + 'train/train_labels.csv')
train['path'] = train['name'].map(lambda x: in_path + 'train/img/' + str(x) + '.jpg')
train_csv = '../data/train/xtrain1.csv'
try:
	print("Trying to load existing features file...")
	xtrain = pd.read_csv(train_csv)
	print("Done.")
except FileNotFoundError as e:
	print("Could not find. Loading images and calculating features...")
	xtrain = load_img(train['path']); print('train...')
	pd.DataFrame.from_dict(xtrain).to_csv(train_csv, index=False)
	xtrain = pd.read_csv(train_csv)
n_train = len(xtrain)

print('Loading Test Data')
test_jpg = glob.glob(in_path + 'test/img/*.jpg')
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['name','path']
test_csv = '../data/test/xtest1.csv'
try:
	xtest = pd.read_csv(test_csv)
except FileNotFoundError as e:
	print("Trying to load existing features file...")
	xtest = load_img(test['path']); print('test...')
	pd.DataFrame.from_dict(xtest).to_csv(test_csv, index=False)
xtest = pd.read_csv(test_csv)

xtrain = xtrain.values
xtest = xtest.values
y = train['invasive'].values

print('xgb fitting ...')
xgb_test = pd.DataFrame(test[['name']], columns=['name'])
y_pred = np.zeros(xtest.shape[0])
xgtest = xgb.DMatrix(xtest)
score = 0
kf = cross_validation.KFold(n_train, n_folds=n_folds, shuffle=False)

print('Training and making predictions')
for i, (trn_index, val_index) in enumerate(kf):

    xgtrain = xgb.DMatrix(xtrain[trn_index], label=y[trn_index])
    xgvalid = xgb.DMatrix(xtrain[val_index], label=y[val_index])

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

    y_pred += model.predict(xgtest,ntree_limit=model.best_ntree_limit) / n_folds
    score += model.best_score / n_folds

print('Mean AUC:',score)

now = datetime.datetime.now()
xgb_test['invasive'] = y_pred
submission_file = '../submit_svm_' + datetime_for_filename() + '.csv'
xgb_test[['name','invasive']].to_csv(submission_file, index=None)
print("Saved submission file to ", submission_file)
