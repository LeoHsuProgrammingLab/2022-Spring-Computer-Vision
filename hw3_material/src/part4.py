import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(500)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])
    
    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
    # for all images to be stitched:
    w_pos = imgs[0].shape[1]
    
    for idx in tqdm(range(len(imgs) - 1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # ref:
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
        # https://www.twblogs.net/a/5c5497e8bd9eee06ee21ab9a
        # https://www.csdn.net/tags/MtjaMg3sNDAwNS1ibG9n.html Fabulous one, with explicit explanation!
        # https://github.com/hughesj919/HomographyEstimation/blob/1a29d5f673852e5ac21fc4ab5c0b12164b1a2423/Homography.py#L131

        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = bf.knnMatch(des1, des2, k = 2)

        src_pt = []
        dst_pt = []
        for first, second in matches:
            if first.distance < 0.7*second.distance:
                src_pt.append( kp1[first.queryIdx].pt )#img1's feature position
                dst_pt.append( kp2[first.trainIdx].pt )#img2's feature position
        
        src_pt = np.array(src_pt)
        dst_pt = np.array(dst_pt)
        # TODO: 2. apply RANSAC to choose best H
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, ransac(src_pt, dst_pt, threshold = 4, itr_times = 1000))
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction = 'b')
        w_pos+=im2.shape[1]

    return out

def ransac(src_pt, dst_pt, threshold, itr_times):
    final_H = None
    max_inlier = 0
    all_features_num = src_pt.shape[0]
    for i in range(itr_times):
        u = np.zeros((10, 2))
        v = np.zeros((10, 2))
        for i in range(10):
            rdmn = random.randint(0, all_features_num-1)
            u[i] = src_pt[rdmn]
            v[i] = dst_pt[rdmn]
        
        # from img2 to img1 because H3 = H1*H2 instead of H2*H1
        H = solve_homography(v, u)

        #sort the src ux, uy, 1
        all_u = np.zeros((3, all_features_num))
        all_u[0:2] = np.transpose(src_pt)
        all_u[2] = np.ones((1, all_features_num))
        all_v = np.zeros((3, all_features_num))
        all_v[0:2] = np.transpose(dst_pt)
        all_v[2] = np.ones((1, all_features_num))
        #calculate the distance
        u_H = np.dot(H, all_v)
        u_H = u_H/u_H[2]
        error = np.linalg.norm((u_H - all_u)[:-1, :], ord = 1, axis = 0)
        inlier_num = np.sum(error<threshold)
        if inlier_num > max_inlier:
            max_inlier = inlier_num
            final_H = H
    return final_H


if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)




