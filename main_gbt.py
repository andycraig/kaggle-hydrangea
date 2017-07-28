#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Based on
# https://www.kaggle.com/the1owl/fractals-of-nature-blend-0-90050

from train import train_gbt
from utils import preprocess_gbt
from PIL import Image, ImageStat
#from tqdm import tqdm
from sklearn import cross_validation
import pandas as pd
import numpy as np
import glob
#import cv2
import scipy
import random
import datetime
import os
import yaml
#import warnings
#warnings.filterwarnings('ignore')

config = yaml.load('config.yaml')

# Load file of training image names and correct labels.
train_set = pd.read_csv(config['test_set'])
# Load file of test image names and dummy labels.
test_set = pd.read_csv(config['train_set'])

print('Loading training data features...')
train_features = pd.read_csv(config['train_features_gbt'])
print("Done.")
print('Loading test data features...')
test_features = pd.read_csv(config['test_features_gbt']).values
print('Done.')

y = train_set['invasive'].values

print('xgb fitting ...')
y_pred, score = train_gbt(train_features=train_features, test_features=test_features, train_labels=y)

print('Mean AUC:',score)

now = datetime.datetime.now()
xgb_test['invasive'] = y_pred
submission_file = '../submit_svm_' + datetime_for_filename() + '.csv'
xgb_test[['name','invasive']].to_csv(submission_file, index=None)
print("Saved submission file to ", submission_file)
