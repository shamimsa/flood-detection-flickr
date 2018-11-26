#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import glob
import os
from scipy import misc

def loadImages(path, tag, isGrayscale):
    
    basePath = os.getcwd() + path
    
    images = []
    if isGrayscale:
        for filename in glob.glob(basePath + tag + "*.jpg"):
            image = misc.imread(filename, flatten = 1)
            images.append(image)
            
    else:
        for filename in glob.glob(basePath + tag + "*.jpg"):
            image = misc.imread(filename)
            images.append(image)
            
    return images
    
def loadAllPatches(path, tag, isGrayscale):
    
    waterImages = loadImages(path + "/water/", tag, isGrayscale)
    noWaterImages = loadImages(path + "/no_water/", tag, isGrayscale)
    
    skyImages = loadImages(path + "/sky/", tag, isGrayscale)
    vegetationImages = loadImages(path + "/vegetation/", tag, isGrayscale)
    
    return { "water" : waterImages, "no_water" : noWaterImages, "sky" : skyImages, "vegetation" : vegetationImages }
    
if __name__ == '__main__':
    
    path = "/../data/patches/test"
    tag = ""
    isGrayscale = False
    
    data = loadAllPatches(path, tag, isGrayscale)
    
    numOfWaterImages = len(data["water"])
    numOfNoWaterImages = len(data["no_water"])
    numOfSkyImages = len(data["sky"])
    numOfVegetationImages = len(data["vegetation"])
    
    print("Found %i water images" % (numOfWaterImages))
    print("Found %i non-water images" % (numOfNoWaterImages))
    print("Found %i sky images" % (numOfSkyImages))
    print("Found %i vegetation images" % (numOfVegetationImages))
    