#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import time

import numpy as np

from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

import parameters
import utils.datasetLoader as datasetLoader
import utils.imageLoader as imageLoader
import utils.imageSegmentor as imageSegmentor

if __name__ == '__main__':
	
	dataset = datasetLoader.load(parameters.patchHeight, parameters.patchWidth, parameters.patchStepY, parameters.patchStepX)
	
# 	x = dataset["water"]["train"]["x"]
# 	y = dataset["water"]["train"]["y"]
# 
# 	mean = np.mean(x, axis=0)
# 	std = np.std(x, axis=0)
# 	
# 	x = (x - mean) / std
# 	
# 	print("Training SVM classifier with radial basis function kernel on raw features...")
# 	t0 = time.time()
# 	
# 	classifier = SVC(kernel="rbf")
# 	classifier.fit(x, y)
# 	
# 	print("done in %.2fs." % (time.time() - t0))
# 	print("")
# 	
# 	print("Evaluating classifier on the training data set...")
# 	t0 = time.time()
# 	
# 	predictions = classifier.predict(x)
# 	
# 	print("Number of training samples = %d" % (x.shape[0]))
# 	print("Number of errors = %d" % (y != predictions).sum())
# 	print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
# 	print("done in %.2fs." % (time.time() - t0))
# 	print("")
# 
# 	x = dataset["water"]["test"]["x"]
# 	y = dataset["water"]["test"]["y"]
# 	
# 	print("Evaluating classifier on the testing data set...")
# 	t0 = time.time()
# 	
# 	x = (x - mean) / std
# 	
# 	predictions = classifier.predict(x)
# 	
# 	print("Number of testing samples = %d" % (x.shape[0]))
# 	print("Number of errors = %d" % (y != predictions).sum())
# 	print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
# 	print("done in %.2fs." % (time.time() - t0))
# 	print("")
# 	
# 	if parameters.segmentImages:
# 		
# 		print("Segmenting water images in the testing data set...")
# 		t0 = time.time()
# 		
# 		images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
# 		
# 		imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, mean, std, pca=None, tag="svm/svm")
# 		
# 		print("done in %.2fs." % (time.time() - t0))
# 		print("")
		
	x = dataset["water"]["train"]["x"]
	y = dataset["water"]["train"]["y"]

	k = parameters.k
	
	print("Extracting %i principal components from raw features..." % (k))
	t0 = time.time()
	
	pca = RandomizedPCA(n_components=k)
	x = pca.fit_transform(x)
	
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
	mean = np.mean(x, axis=0)
	std = np.std(x, axis=0)
	
	x = (x - mean) / std
	
	print("Training SVM classifier with radial basis function kernel on PCA features...")
	t0 = time.time()
	
	classifier = SVC(kernel="rbf")
	classifier.fit(x, y)
	
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
	print("Evaluating classifier on the training data set...")
	t0 = time.time()
	
	predictions = classifier.predict(x)
	
	print("Number of training samples = %d" % (x.shape[0]))
	print("Number of errors = %d" % (y != predictions).sum())
	print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
	x = dataset["water"]["test"]["x"]
	y = dataset["water"]["test"]["y"]
	
	x = (pca.transform(x) - mean) / std
	
	print("Evaluating classifier on the testing data set...")
	t0 = time.time()
	
	predictions = classifier.predict(x)
	
	print("Number of testing samples = %d" % (x.shape[0]))
	print("Number of errors = %d" % (y != predictions).sum())
	print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
	if parameters.segmentImages:
		
		print("Segmenting water images in the testing data set...")
		t0 = time.time()
		
		images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
		
		imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, mean, std, pca=pca, tag="svm/svm_pca_" + "k:" + str(k))
		
		print("done in %.2fs." % (time.time() - t0))
		print("")
	