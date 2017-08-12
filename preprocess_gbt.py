#!/usr/bin/env python3

from PIL import Image
from PIL import ImageStat
import scipy
import scipy.stats
import yaml
import sys
import pandas as pd
import numpy as np
import cv2

def main(config_file):
	with open(config_file, 'r') as f:
		config = yaml.load(f)

	# Load file of training image names and correct labels.
	train_set = pd.read_csv(config['train_set'])
	# Load file of test image names and dummy labels.
	test_set = pd.read_csv(config['test_set'])
	# Preprocess GBT features.

	print("Preprocessing GBT training set features...")
	train_features_gbt = preprocess_gbt(config['train_imgs'], train_set)
	np.savetxt(config['train_features_gbt'], train_features_gbt, delimiter=',')
	print("Saved training features to " + config['train_features_gbt'])
	test_features_gbt = preprocess_gbt(config['test_imgs'], test_set)
	np.savetxt(config['test_features_gbt'], test_features_gbt, delimiter=',')
	print("Saved test features to " + config['test_features_gbt'])

def get_features(path):
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
	bw = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
	ft += list(cv2.HuMoments(cv2.moments(bw)).flatten())
	ft += list(cv2.calcHist([bw],[0],None,[64],[0,256]).flatten()) #bw
	ft += list(cv2.calcHist([img],[0],None,[64],[0,256]).flatten()) #r
	ft += list(cv2.calcHist([img],[1],None,[64],[0,256]).flatten()) #g
	ft += list(cv2.calcHist([img],[2],None,[64],[0,256]).flatten()) #b
	m, s = cv2.meanStdDev(img) #mean and standard deviation
	ft += list(m.ravel())
	ft += list(s.ravel())
	ft += [cv2.Laplacian(bw, cv2.CV_64F).var()]
	ft += [cv2.Laplacian(img, cv2.CV_64F).var()]
	ft += [cv2.Sobel(bw,cv2.CV_64F,1,0,ksize=5).var()]
	ft += [cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=5).var()]
	ft += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]
	ft += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]
	return ft

def preprocess_gbt(folder_name, names_and_labels):
	imf_d = {}
	for i, img_name in enumerate(names_and_labels['name'].iloc[:]):
		img_src = folder_name + str(img_name) + '.jpg'
		imf_d[img_name] = get_features(img_src)
	imf_df = pd.DataFrame.from_dict(imf_d, orient="index")
	return imf_df

if __name__ == "__main__":
	main(sys.argv[1])
