#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import numpy as np

from matplotlib import colors
from matplotlib import pyplot as plt

import imageLoader

"""
 Extract features from a list of images by extracting sub-image patches across individual 
 images and scanning the patch across the image with a step size of: [stepX by stepY]. 
"""
def extract(images, height, width, stepY, stepX):
    
    features = None
    for idx in range(images.shape[0]):
        image = images[idx]
        
        image = colors.rgb_to_hsv(image / 255.0)
        
        for j in range(0, image.shape[0] - height + 1, stepY):
            for i in range(0, image.shape[1] - width + 1, stepX):
                
                imagePatch = image[j:j+height,i:i+width,:]
                
                if features == None:
                    features = imagePatch.flatten().reshape(1,-1)
                else:
                    features = np.concatenate((features, imagePatch.flatten().reshape(1,-1)), axis=0)
                
    return features
    

if __name__ == '__main__':
   
    path = "/../data/patches/train"
    tag = ""
    isGrayscale = False
    
    images = imageLoader.load(path, tag, isGrayscale)
    
    numOfNoWaterImages = len(images["no_water"])
    
    features = extract(np.array(images["no_water"]), 28, 28, 14, 14)
     
    print features.shape
    
    plt.imshow(colors.hsv_to_rgb(features[756].reshape((28, 28, 3))))
    plt.show()
    
    plt.imshow(colors.hsv_to_rgb(features[757].reshape((28, 28, 3))), cmap="jet", interpolation='lanczos')
    plt.show()
    
