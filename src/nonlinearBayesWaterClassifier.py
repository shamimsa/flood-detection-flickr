#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import time

import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.decomposition import RandomizedPCA

import parameters
import utils.datasetLoader as datasetLoader
import utils.statistics as statistics
import utils.imageLoader as imageLoader
import utils.imageSegmentor as imageSegmentor

if __name__ == '__main__':

	dataset = datasetLoader.load(parameters.patchHeight, parameters.patchWidth, parameters.patchStepY, parameters.patchStepX)
	
	x = dataset["water"]["train"]["x"]
	y = dataset["water"]["train"]["y"]
		
	print("Training quadratic Bayesian classifier on raw features...")
	t0 = time.time()
		
	classifier = QuadraticDiscriminantAnalysis(reg_param=0.0, store_covariances=True)
	classifier.fit(x, y)
		
	print("done in %.2fs." % (time.time() - t0))
	print("")
		
	print("Calculating separability measures on raw features:")
	t0 = time.time()
		
	separabilityMeasures = statistics.calcSeparabilityMeasures(classifier.means_, classifier.covariances_, classifier.priors_)
	
	print("J1 / l = %f" % (separabilityMeasures["J1"] / separabilityMeasures["l"]))
	print("J3 / l = %f" % (separabilityMeasures["J3"] / separabilityMeasures["l"]))
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
		
	print("Evaluating classifier on the testing data set...")
	t0 = time.time()
		
	predictions = classifier.predict(x)
		
	print("Number of testing samples = %d" % (x.shape[0]))
	print("Number of errors = %d" % (y != predictions).sum())
	print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
# 	if parameters.segmentImages:
# 			
# 		print("Segmenting water images in the testing data set...")
# 		t0 = time.time()
# 				
# 		images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
# 				
# 		imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, 0., 1., pca=None, tag="bayes/nonlinear_bayes")
# 				
# 		print("done in %.2fs." % (time.time() - t0))
# 		print("")
		
	x = dataset["water"]["train"]["x"]
	y = dataset["water"]["train"]["y"]
	
	k = parameters.k
	
	print("Extracting %i principal components from the raw features..." % (k))
	t0 = time.time()
	
	pca = RandomizedPCA(n_components=k)
	x = pca.fit_transform(x)
	
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
	print("Training quadratic Bayesian classifier on PCA features...")
	t0 = time.time()
	
	classifier = QuadraticDiscriminantAnalysis(reg_param=0.0, store_covariances=True)
	classifier.fit(x, y)
	
	print("done in %.2fs." % (time.time() - t0))
	print("")
	
	print("Calculating separability measures for PCA features:")
	t0 = time.time()
	
	separabilityMeasures = statistics.calcSeparabilityMeasures(classifier.means_, classifier.covariances_, classifier.priors_)
	
	print("J1 / l = %f" % (separabilityMeasures["J1"] / separabilityMeasures["l"]))
	print("J3 / l = %f" % (separabilityMeasures["J3"] / separabilityMeasures["l"]))
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
	
	x = pca.transform(x)
	
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
				
		imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, 0., 1., pca=pca, tag="bayes/nonlinear_bayes_pca_" + "k:" + str(k))
				
		print("done in %.2fs." % (time.time() - t0))
		print("")
	