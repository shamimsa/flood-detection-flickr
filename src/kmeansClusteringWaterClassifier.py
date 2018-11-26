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
from sklearn.cluster import MiniBatchKMeans

import parameters
import utils.datasetLoader as datasetLoader
import utils.imageLoader as imageLoader
import utils.imageSegmentor as imageSegmentor

def mapTargetsToCluster(y, numberOfClusters, predictions):
    
    classMappings = np.zeros(numberOfClusters).astype("int")
    for idx in range(numberOfClusters):
        
        mappedPredictions = np.array([np.choose(prediction, classMappings) for prediction in predictions])
        errors = (y != mappedPredictions).sum()
        
        testClassMappings = np.copy(classMappings)
        testClassMappings[idx] = 1

        testErrors = (y != np.array([np.choose(prediction, testClassMappings) for prediction in predictions])).sum()
        if  testErrors < errors:
            classMappings[idx] = 1
        
    return classMappings

def calcClusterPopulations(numberOfClusters, predictions):
    
    clusterPopulations = np.zeros(numberOfClusters).astype("int")
    for idx in range(numberOfClusters):
        
        classMappings = np.zeros(numberOfClusters).astype("int")
        classMappings[idx] = 1

        clusterPopulations[idx] = np.array([np.choose(prediction, classMappings) for prediction in predictions]).sum()
        
    return clusterPopulations

if __name__ == '__main__':
    
    numberOfClusters = 2
    
    dataset = datasetLoader.load(parameters.patchHeight, parameters.patchWidth, parameters.patchStepY, parameters.patchStepX)
    
#     x = dataset["water"]["train"]["x"]
#     y = dataset["water"]["train"]["y"]
#      
#     print("Partitioning raw features into " + str(numberOfClusters) + " clusters with k-means clustering...")
#     t0 = time.time()
#  
#     classifier = MiniBatchKMeans(n_clusters=numberOfClusters, init='k-means++', max_iter=100, batch_size=100, verbose=False, compute_labels=True)
#     classifier.fit(x)
#      
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#      
#     print("Evaluating classifier on the training data set...")
#     t0 = time.time()
#      
#     predictions = classifier.predict(x)
#      
#     classMappings = mapTargetsToCluster(y, numberOfClusters, predictions)
#     mappedPredictions = np.array([np.choose(prediction, classMappings) for prediction in predictions])
#      
#     clusterPopulations = calcClusterPopulations(numberOfClusters, predictions)
#      
#     print("Number of clusters = %d" % (numberOfClusters))
#     print("Cluster assignments for water class: %s" % (classMappings))
#     print("Total number of training samples = %d" % (x.shape[0]))
#     for idx in range(numberOfClusters):
#         print("Number of training samples in cluster %d = %d" % (idx, clusterPopulations[idx]))
#     print("Number of errors = %d" % (y != mappedPredictions).sum())
#     print("Error rate = %5.2f %%" % (100. * (y != mappedPredictions).sum() / x.shape[0]))
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#      
#     x = dataset["water"]["test"]["x"]
#     y = dataset["water"]["test"]["y"]
#      
#     print("Evaluating classifier on the testing data set...")
#     t0 = time.time()
#      
#     predictions = classifier.predict(x)
#     mappedPredictions = np.array([np.choose(prediction, classMappings) for prediction in predictions])
#      
#     clusterPopulations = calcClusterPopulations(numberOfClusters, predictions)
#      
#     print("Total number of testing samples = %d" % (x.shape[0]))
#     for idx in range(numberOfClusters):
#         print("Number of testing samples in cluster %d = %d" % (idx, clusterPopulations[idx]))
#     print("Number of errors = %d" % (y != mappedPredictions).sum())
#     print("Error rate = %5.2f %%" % (100. * (y != mappedPredictions).sum() / x.shape[0]))
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#      
#     if parameters.segmentImages:
#           
#         print("Segmenting water images in the testing data set...")
#         t0 = time.time()
#           
#         images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
#           
#         imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, 0., 1., pca=None, classMappings=classMappings, tag="kmeans/kmeans_" + "c:" + str(numberOfClusters))
#           
#         print("done in %.2fs." % (time.time() - t0))
#         print("")
     
    x = dataset["water"]["train"]["x"]
    y = dataset["water"]["train"]["y"]
    
    k = parameters.k
    
    print("Extracting %i principal components from raw features..." % (k))
    t0 = time.time()
    
    pca = RandomizedPCA(n_components=k)
    x = pca.fit_transform(x)
    
    print("done in %.2fs." % (time.time() - t0))
    print("")

    print("Partitioning PCA features into " + str(numberOfClusters) + " clusters with k-means clustering...")
    t0 = time.time()
    
    classifier = MiniBatchKMeans(n_clusters=numberOfClusters, init='k-means++', max_iter=100, batch_size=100, verbose=False, compute_labels=True)
    classifier.fit(x)
    
    print("done in %.2fs." % (time.time() - t0))
    print("")
    
    print("Evaluating classifier on the training data set...")
    t0 = time.time()
    
    predictions = classifier.predict(x)
    
    classMappings = mapTargetsToCluster(y, numberOfClusters, predictions)
    mappedPredictions = np.array([np.choose(prediction, classMappings) for prediction in predictions])
    
    clusterPopulations = calcClusterPopulations(numberOfClusters, predictions)

    print("Number of clusters = %d" % (numberOfClusters))
    print("Cluster assignments for water class: %s" % (classMappings))
    print("Total number of training samples = %d" % (x.shape[0]))
    for idx in range(numberOfClusters):
        print("Number of training samples in cluster %d = %d" % (idx, clusterPopulations[idx]))
    print("Number of errors = %d" % (y != mappedPredictions).sum())
    print("Error rate = %5.2f %%" % (100. * (y != mappedPredictions).sum() / x.shape[0]))
    print("done in %.2fs." % (time.time() - t0))
    print("")
    
    x = dataset["water"]["test"]["x"]
    y = dataset["water"]["test"]["y"]
  
    x = pca.transform(x)

    print("Evaluating classifier on the testing data set...")
    t0 = time.time()
    
    predictions = classifier.predict(x)
    mappedPredictions = np.array([np.choose(prediction, classMappings) for prediction in predictions])
    
    clusterPopulations = calcClusterPopulations(numberOfClusters, predictions)
    
    print("Total number of testing samples = %d" % (x.shape[0]))
    for idx in range(numberOfClusters):
        print("Number of testing samples in cluster %d = %d" % (idx, clusterPopulations[idx]))
    print("Number of errors = %d" % (y != mappedPredictions).sum())
    print("Error rate = %5.2f %%" % (100. * (y != mappedPredictions).sum() / x.shape[0]))
    print("done in %.2fs." % (time.time() - t0))
    print("")
    
    if parameters.segmentImages:
          
        print("Segmenting water images in the testing data set...")
        t0 = time.time()
          
        images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
          
        imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, 0., 1., pca=pca, classMappings=classMappings, tag="kmeans/kmeans_pca_" + "k:" + str(k) + "_c:" + str(numberOfClusters))
          
        print("done in %.2fs." % (time.time() - t0))
        print("")
    