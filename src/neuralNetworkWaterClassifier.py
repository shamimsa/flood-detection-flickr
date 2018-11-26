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

from matplotlib import colors
from matplotlib import pyplot as plt

from sklearn.decomposition import RandomizedPCA

import parameters
import utils.datasetLoader as datasetLoader
import utils.imageLoader as imageLoader
import utils.imageSegmentor as imageSegmentor

from utils.neuralNetworks import ClassificationNeuralNetwork


"""
 Plot the learned dictionary atoms used to initialize the first hidden layer of the neural network. 
"""
def plotDictionaryAtoms(patchHeight, patchWidth, atoms, mean, std, pca, title="", tag="atoms", figNum=0):
    
    plt.figure(num=figNum, figsize=(7.5, 5.5))
    gridSize =  min(np.floor(np.sqrt(atoms.shape[0])).astype(int), 9)
    
    for i, atom in enumerate(atoms[:gridSize * gridSize]):
        plt.subplot(gridSize, gridSize, i + 1)
        
        if pca != None:
            patch = pca.inverse_transform(std * atom + mean).reshape((patchHeight, patchWidth, 3))
        else:
            patch = (std * atom + mean).reshape((patchHeight, patchWidth, 3))

        plt.imshow(colors.hsv_to_rgb(patch), cmap="jet", interpolation='lanczos')
        plt.xticks(())
        plt.yticks(())
        
    plt.suptitle(title, fontsize=11)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.savefig(os.getcwd() + "/../output/models/nn/" + tag + ".jpg")
    plt.show()

if __name__ == '__main__':
    
    hiddenLayerDimensions = np.array([100, 100])
    
    numberOfEpochs = 25
    momentum = 0.005
    weightdecay = 0.001
    
    displayFigures = True
    
    
    dataset = datasetLoader.load(parameters.patchHeight, parameters.patchWidth, parameters.patchStepY, parameters.patchStepX)
    
    x = dataset["water"]["train"]["x"]
    y = dataset["water"]["train"]["y"]
     
    arch_tag = str(x.shape[1]) + "x"
    for idx in range(hiddenLayerDimensions.shape[0]):
        arch_tag = arch_tag + str(hiddenLayerDimensions[idx]) + "x"
    arch_tag = arch_tag + "2"
             
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
            
    x = (x - mean) / std
            
    print("Training neural network classifier on raw features...")
    t0 = time.time()
            
    classifier = ClassificationNeuralNetwork(x.shape[1], hiddenLayerDimensions, 2)
    errors = classifier.fit(x, y, numberOfEpochs, momentum, weightdecay, True)
         
    print("done in %.2fs." % (time.time() - t0))
    print("")
  
    if displayFigures:
           
        plt.figure(1)
        plt.plot(np.arange(1,len(errors) + 1), errors)
           
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.xlim(xmin=1, xmax=numberOfEpochs)
 
        tag = "nn_raw_train_error_" + "net:" + arch_tag + ".jpg"
        plt.savefig(os.getcwd() + "/../output/models/nn/" + tag)
       
        plt.show()
       
        title = "First hidden layer weights learned on a total of {0:d} image patches".format(x.shape[0])
        plotDictionaryAtoms(parameters.patchHeight, parameters.patchWidth, classifier.getFirstHiddenLayerWeights(), mean, std, pca=None, title=title, tag="nn_raw_atoms_net:" + arch_tag, figNum=3)
                 
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
              
    print("Evaluating classifier on the test data set...")
    t0 = time.time()
              
    x = (x - mean) / std
              
    predictions = classifier.predict(x)
              
    print("Number of training samples = %d" % (x.shape[0]))
    print("Number of errors = %d" % (y != predictions).sum())
    print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
    print("done in %.2fs." % (time.time() - t0))
    print("")
         
    if parameters.segmentImages:
               
        print("Segmenting water images in testing data set...")
        t0 = time.time()
               
        images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
               
        imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, mean, std, pca=None,tag="nn/nn_raw_net:" + arch_tag)
               
        print("done in %.2fs." % (time.time() - t0))
        print("")
    
#     x = dataset["water"]["train"]["x"]
#     y = dataset["water"]["train"]["y"]
#        
#     k = parameters.k
#     
#     arch_tag = str(k) + "x"
#     for idx in range(hiddenLayerDimensions.shape[0]):
#         arch_tag = arch_tag + str(hiddenLayerDimensions[idx]) + "x"
#     arch_tag = arch_tag + "2"
#        
#     print("Extracting %i principal components from raw features..." % (k))
#     t0 = time.time()
#        
#     pca = RandomizedPCA(n_components=k)
#     x = pca.fit_transform(x)
#        
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#        
#     mean = np.mean(x, axis=0)
#     std = np.std(x, axis=0)
#     
#     x = (x - mean) / std
#        
#     print("Training neural network classifier on PCA features...")
#     t0 = time.time()
#     
#     classifier = ClassificationNeuralNetwork(x.shape[1], hiddenLayerDimensions, 2)
#     errors = classifier.fit(x, y, numberOfEpochs, momentum, weightdecay, True)
# 
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#     
#     if displayFigures:
#          
#         plt.figure(2)
#         plt.plot(np.arange(1,len(errors) + 1), errors)
#          
#         plt.xlabel("Epoch")
#         plt.ylabel("Mean Squared Error")
#         plt.xlim(xmin=1, xmax=numberOfEpochs)
#          
#         tag = "nn_pca_train_error_" + "k:" + str(k) + "_net:" + arch_tag + ".jpg"
#         plt.savefig(os.getcwd() + "/../output/models/nn/" + tag)
#      
#         plt.show()
#      
#         title = "First hidden layer weights learned on a total of {0:d} image patches".format(x.shape[0])
#         plotDictionaryAtoms(parameters.patchHeight, parameters.patchWidth, classifier.getFirstHiddenLayerWeights(), mean, std, pca=pca, title=title, tag="nn_pca_atoms_" + "k:" + str(k) + "_net:" + arch_tag, figNum=4)
#     
#     print("Evaluating classifier on the training data set...")
#     t0 = time.time()
#        
#     predictions = classifier.predict(x)
#        
#     print("Number of training samples = %d" % (x.shape[0]))
#     print("Number of errors = %d" % (y != predictions).sum())
#     print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#     
#     x = dataset["water"]["test"]["x"]
#     y = dataset["water"]["test"]["y"]
#        
#     x = (pca.transform(x) - mean) / std
#        
#     print("Evaluating classifier on the testing data set...")
#     t0 = time.time()
#        
#     predictions = classifier.predict(x)
#        
#     print("Number of testing samples = %d" % (x.shape[0]))
#     print("Number of errors = %d" % (y != predictions).sum())
#     print("Error rate = %5.2f %%" % (100. * (y != predictions).sum() / x.shape[0]))
#     print("done in %.2fs." % (time.time() - t0))
#     print("")
#        
#     if parameters.segmentImages:
#            
#         print("Segmenting water images in testing data set...")
#         t0 = time.time()
#            
#         images = np.array(imageLoader.loadImages("/../data/raw/test/water", "/", False))
#            
#         imageSegmentor.segment(images, parameters.patchHeight, parameters.patchWidth, parameters.segmentationStepY, parameters.segmentationStepX, classifier, mean, std, pca=pca,tag="nn/nn_pca_" + "k:" + str(k) + "_net:" + arch_tag)
#            
#         print("done in %.2fs." % (time.time() - t0))
#         print("")
    