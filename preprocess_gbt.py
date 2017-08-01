#!/usr/bin/env python3

from PIL import Image
from PIL import ImageStat
import scipy
import scipy.stats
import yaml
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main(config_file):
	with open(config_file, 'r') as f:
		config = yaml.load(f)

	# Load file of training image names and correct labels.
	train_set = pd.read_csv(config['train_set'])
	# Load file of test image names and dummy labels.
	test_set = pd.read_csv(config['test_set'])
	# Preprocess GBT features.

	print("Preprocessing GBT training set features...")
	train_features_gbt_unscaled = preprocess_gbt(config['train_imgs'], train_set)
	print("Centering and scaling...")
	scaler = StandardScaler().fit(train_features_gbt_unscaled)
	train_features_gbt = scaler.transform(train_features_gbt_unscaled)
	n_components = 3
	print("Adding projections onto first " + str(n_components) + " principle components...")
	pca = PCA(n_components)
	# Use fit_transform here.
	train_features_gbt = np.hstack(train_features_gbt,
							pca.fit_transform(train_features_gbt))
	np.savetxt(config['train_features_gbt'], train_features_gbt, delimiter=',')
	print("Saved training features to " + config['train_features_gbt'])
	print("Preprocessing GBT test set features...")
	test_features_gbt_unscaled = preprocess_gbt(config['test_imgs'], test_set)
	print("Centering and scaling (same transformation as for train features)...")
	test_features_gbt = scaler.transform(test_features_gbt_unscaled)
	print("Adding projections onto principle components...")
	# Use transform here, to use the same pinciple components from the train data.
	test_features_gbt = np.hstack(test_features_gbt,
							pca.transform(test_features_gbt))
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
	# bw = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
	# ft += list(cv2.HuMoments(cv2.moments(bw)).flatten())

	# Histograms:
	ft += m.histogram() # All bands, concatenated.
	# Get the bands:
	#im.split() ⇒ sequence
	# BW:
	# img.convert(mode='L')
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
	# I don't think this is the same thing.
	# ft += [cv2.Laplacian(bw, cv2.CV_64F).var()]
	# ft += [cv2.Laplacian(img, cv2.CV_64F).var()]

	# Sobel:
	#image = Image.open('your_image.png')
	#image = image.filter(ImageFilter.FIND_EDGES)

	# ft += [cv2.Sobel(bw,cv2.CV_64F,1,0,ksize=5).var()]
	# ft += [cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=5).var()]
	# ft += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]
	# ft += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]
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
