from pathlib import Path
from PIL import Image
import numpy as np
from numpy import linalg as l_a
import cv2

def normalize(array_):
    norm = np.linalg.norm(array_)
    if norm == 0: 
       return array_
    return array_ / norm

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    
    
    tiny_images = []
    new_size = (10, 10)
    
    for path in image_paths:
        im = Image.open(path)
        # print(np.shape(im))
        im = im.resize(new_size)
        # crop:((l, t, r, b))
        # im = normalize(im)
        im = np.array(im).flatten()
        tiny_images.append(im)
    
    

    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
