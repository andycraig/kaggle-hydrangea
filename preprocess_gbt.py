#!/usr/bin/env python3

from PIL import Image
import yaml
import sys

def main(config_file):
	config = yaml.load(config_file)

	# Preprocess GBT features.
	print("Preprocessing GBT training set features...")
	train_features_gbt = preprocess_gbt('train', train_set)
	train_features_gbt.to_csv(config['train_features_gbt'])
	print("Done.")
	print("Preprocessing GBT test set features...")
	test_features_gbt = preprocess_gbt('test', test_set)
	test_features_gbt.to_csv(config['test_features_gbt'])
	print("Done.")

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

def preprocess_gbt(folder_name, names_and_labels):
	imf_d = {}
    for i, img_name in enumerate(names_and_labels['name'].iloc[:]):
		img_src = '../data/' + folder_name + '/img/' + str(img_name) + '.jpg'
	    imf_d[img_name] = get_features(img_src)
	return imf_d

if __name__ == "__main__":
	main(sys.argv[1])
