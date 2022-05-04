from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode
from tqdm import tqdm

#TA: K = 5
# ref:
# https://notebook.community/joefutrelle/scipy-talk/talk3/2%20knn%20cdist%20impl
# https://blog.csdn.net/MacwinWin/article/details/80002584
# https://www.cnblogs.com/cgmcoding/p/13590505.html
# https://python-course.eu/machine-learning/k-nearest-neighbor-classifier-in-python.php

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    test_predicts = []
    train_labels_arr = np.array(train_labels)
    k = 5
    dist = distance.cdist(test_image_feats, train_image_feats, metric = 'cityblock')
    for row in tqdm(dist):#for every test_image_feats
        k_near = np.argsort(row)[:k]
        test_predicts.append(mode(train_labels_arr[k_near])[0][0])

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
