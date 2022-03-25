import numpy as np
import cv2

from collections import defaultdict

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        
        gaussian_images1 = [image]
        #store the 10 pic
        for i in range(self.num_guassian_images_per_octave-1):
            gaussian_images1.append(cv2.GaussianBlur (image, (0, 0), self.sigma**(i+1)))
        #down-sample
        image_resize = cv2.resize(gaussian_images1[4],None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_NEAREST)
        gaussian_images2 = [image_resize]
        for i in range(self.num_guassian_images_per_octave-1):
            gaussian_images2.append(cv2.GaussianBlur (image_resize, (0, 0), self.sigma**(i+1)))
        
        

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images1 = []
        dog_images2 = []
        for i in range(self.num_DoG_images_per_octave):
            dog_img1, dog_img2 = cv2.subtract(gaussian_images1[i+1], gaussian_images1[i]), cv2.subtract(gaussian_images2[i+1], gaussian_images2[i])
            dog_images1.append(dog_img1)
            dog_img1 = 255*(dog_img1-np.min(dog_img1))/(np.max(dog_img1) - np.min(dog_img1))
            dog_images2.append(dog_img2)
            dog_img2 = 255*(dog_img2-np.min(dog_img2))/(np.max(dog_img2) - np.min(dog_img2))
            # cv2.imwrite(f'/Users/mingtzu/Desktop/dog_img1_{i}.png', dog_img1)
            # cv2.imwrite(f'/Users/mingtzu/Desktop/dog_img2_{i}.png', dog_img2)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        #3-1: combine the pictures in octaves respectively
        octave1 = np.stack((dog_images1[0], dog_images1[1], dog_images1[2], dog_images1[3]), axis = 0)

        octave2 = np.stack((dog_images2[0], dog_images2[1], dog_images2[2], dog_images2[3]), axis = 0)
        
        #3-2: 3*3*3 kernel and store
        keypoints = []
        #octave1
        one1, two1, three1 = octave1.shape[:]
        for i in range(1, one1-1):
            for j in range(1, two1-1):
                for k in range(1, three1-1):
                    if ((octave1[i, j, k] <= octave1[i-1:i+2, j-1:j+2, k-1:k+2]).all() or (octave1[i, j, k] >= octave1[i-1:i+2, j-1:j+2, k-1:k+2]).all()) and np.abs(octave1[i, j, k]) > self.threshold:
                        keypoints.append([j, k]) 
        #octave2
        one2, two2, three2 = octave2.shape[:]
        for i in range(1, one2-1):
            for j in range(1, two2-1):
                for k in range(1, three2-1):
                    if ((octave2[i, j, k] <= octave2[i-1:i+2, j-1:j+2, k-1:k+2]).all() or (octave2[i, j, k] >= octave2[i-1:i+2, j-1:j+2, k-1:k+2]).all()) and np.abs(octave2[i, j, k]) > self.threshold:
                        # print(i, j, k)
                        keypoints.append([2*j, 2*k])
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis = 0)
        
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
