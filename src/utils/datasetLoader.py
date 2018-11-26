#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import os
import time

import numpy as np

from sklearn.externals import joblib

import imageLoader
import featureExtractor

waterClass = ["no_water", "water"]

def loadWaterDataset(basePath="/../data/patches", height=28, width=28, stepY=14, stepX=14):
    
    # Extract water training features
    trainImagesPath = basePath + "/train"
    trainImages = imageLoader.loadAllPatches(trainImagesPath, "", False)
    
    waterFeaturesTrain = np.random.permutation(featureExtractor.extract(np.array(trainImages["water"]), height, width, stepY, stepX))
    noWaterFeaturesTrain = np.random.permutation(featureExtractor.extract(np.array(trainImages["no_water"]), height, width, stepY, stepX))

    featuresTrain = np.concatenate((waterFeaturesTrain, noWaterFeaturesTrain), axis=0)
    targetsTrain = np.concatenate((waterClass.index("water") * np.ones(waterFeaturesTrain.shape[0]), waterClass.index("no_water") * np.ones(noWaterFeaturesTrain.shape[0])), axis=0).astype(int)
    
    idx = np.random.permutation(featuresTrain.shape[0])
    featuresTrain = featuresTrain[idx]
    targetsTrain = targetsTrain[idx]
    
    priorsTrain = 1. * np.bincount(targetsTrain) / targetsTrain.shape[0]
    
    # Extract water testing features
    testImagesPath = basePath + "/test"
    testImages = imageLoader.loadAllPatches(testImagesPath, "", False)
    
    waterFeaturesTest = np.random.permutation(featureExtractor.extract(np.array(testImages["water"]), height, width, stepY, stepX))
    noWaterFeaturesTest = np.random.permutation(featureExtractor.extract(np.array(testImages["no_water"]), height, width, stepY, stepX))
    
    featuresTest = np.concatenate((waterFeaturesTest, noWaterFeaturesTest), axis=0)
    targetsTest = np.concatenate((waterClass.index("water") * np.ones(waterFeaturesTest.shape[0]), waterClass.index("no_water") * np.ones(noWaterFeaturesTest.shape[0])), axis=0).astype(int)
    
    idx = np.random.permutation(featuresTest.shape[0])
    featuresTest = featuresTest[idx]
    targetsTest = targetsTest[idx]
    
    priorsTest = 1. * np.bincount(targetsTest) / targetsTest.shape[0]
    
    return { "train" : {"x" : featuresTrain, "y" : targetsTrain, "priors" : priorsTrain}, "test" : {"x" : featuresTest, "y" : targetsTest, "priors" : priorsTest}}


def load(height=28, width=28, stepY=14, stepX=14):
    
    configTag = "water" + ":" + str(height) + ":" + str(width) + ":" + str(stepY) + ":" + str(stepX)
    
    featuresFile = os.getcwd() + "/../output/data/raw_features_" + configTag + ".pkl"
    
    if os.path.isfile(featuresFile):
        
        print("Loading water dataset from file: '" + featuresFile + "'")
        print("")
        
        dataset = joblib.load(featuresFile)
        
    else:
        
        print("Extracting water dataset from image patches...")
        t0 = time.time()
        
        dataset = loadWaterDataset(basePath="/../data/patches", height=height, width=width, stepY=stepY, stepX=stepX)
        
        joblib.dump(dataset, featuresFile)
        
        print("Saved water dataset to file: '" + featuresFile + "'")
        print("done in %.2fs." % (time.time() - t0))
        print("")
    
    print("Dimensionality of training features = %i" % (dataset["train"]["x"].shape[1]))
    
    print("Total number of training features = %i" % (dataset["train"]["y"].shape[0]))
    print("Prior probablity of water training features = %f" % (dataset["train"]["priors"][waterClass.index("water")]))
    print("Prior probablity of non-water training features = %f" % (dataset["train"]["priors"][waterClass.index("no_water")]))
    
    print("Total number of testing features = %i" % (dataset["test"]["y"].shape[0]))
    print("Prior probablity of water testing features = %f" % (dataset["test"]["priors"][waterClass.index("water")]))
    print("Prior probablity of non-water testing features = %f" % (dataset["test"]["priors"][waterClass.index("no_water")]))
    print("")
    
    return { "water" : dataset }
    