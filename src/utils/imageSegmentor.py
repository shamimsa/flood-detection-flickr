#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import os

import numpy as np

from matplotlib import colors

from scipy import misc

import utils.featureExtractor as featureExtractor

"""
 Segment the image by calculating the likelihood ratio as the number of times the pixel is 
 classified as water to the number of image patches containing the pixel.
"""
def segment(images, patchHeight, patchWidth, patchStepY, patchStepX, classifier, mean, std, pca=None, classMappings=[0,1], tag="img"):

	# iterate over all images
	for idx in range(images.shape[0]):
	
		print("segmenting image # %i of %i" % (idx + 1, images.shape[0]))
		
		segmentedImage = colors.rgb_to_hsv(images[idx].copy() / 255.)
		
		segmentedClassification = np.zeros((segmentedImage.shape[0], segmentedImage.shape[1]))
		segmentedNormalization = np.zeros(segmentedClassification.shape)
		
		# saturation - set constant saturation level
		segmentedImage[:, :, 1] = 0.3 
		
		# scan the whole image, extracting image patches for classification
		for j in range(0, images[idx].shape[0] - patchHeight + 1, patchStepY):
		
			for i in range(0, images[idx].shape[1] - patchWidth + 1, patchStepX):
				
				# extract image patch from whole image
				imagePatch = images[idx][j:j+patchHeight,i:i+patchWidth,:]
				
				# extract raw features
				x = featureExtractor.extract(np.array([imagePatch]), patchHeight, patchWidth, patchHeight, patchWidth)
				
				# if using pricipal component analysis, project onto pca axes before normalizing
				if pca != None:
					x = pca.transform(x)
				
				# normalize and classify feature
				classPrediction = classifier.predict((x - mean) / std)

				if np.choose(classPrediction, classMappings) == 1:
					segmentedClassification[j:j+patchHeight,i:i+patchWidth] += 1
				
				segmentedNormalization[j:j+patchHeight,i:i+patchWidth] += 1

		segmentedClassification /= (segmentedNormalization + 1.0e-7)
		
		# hue - tint target pixels blue
		segmentedImage[...,0][segmentedClassification - 0.5 >= 0] = 0.6
		
		# hue - tint non-target pixels red
		segmentedImage[...,0][segmentedClassification - 0.5 < 0] = 1.0 
		
		misc.imsave(os.getcwd() + "/../output/segmentation/" + tag + "_{0:03d}.jpg".format(idx), colors.hsv_to_rgb(segmentedImage))
	