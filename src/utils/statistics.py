#===========================================================================
# Project Part II - Water Region Classification & Segmentation
# NCSU ECE 759
# April 25th, 2016
# 
# Chris Jenkins, Shamim Samadi, & Erik Skau 
#===========================================================================

import numpy as np
from numpy import linalg


"""
 Calculates class separability measures.  
"""
def calcSeparabilityMeasures(means, covariances, priors):
    
    mean_1 = means[0]
    mean_2 = means[1]
    
    sigma_1 = covariances[0]
    sigma_2 = covariances[1]
    
    prior_1 = priors[0]
    prior_2 = priors[1]
    
    mean_0 = prior_1 * mean_1 + prior_2 * mean_2
    
    Sw = prior_1 * sigma_1 + prior_2 * sigma_2
    Sb = prior_1 * np.outer((mean_1 - mean_0), (mean_1 - mean_0)) + prior_2 * np.outer((mean_2 - mean_0), (mean_2 - mean_0))
    Sm = Sw + Sb
    
    J1 = np.trace(Sm) / np.trace(Sw)
    J3 = np.trace(np.dot(linalg.inv(Sw), Sm))
    
    l = mean_1.shape[0]
    
    return { "l" : l, "J1" : J1, "J3" : J3 }
