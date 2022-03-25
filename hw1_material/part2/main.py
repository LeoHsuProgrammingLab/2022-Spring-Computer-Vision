import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--sigma_s', default=1, type=int, help='sigma of spatial kernel')
    parser.add_argument('--sigma_r', default=0.05, type=float, help='sigma of range kernel')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray1  = (img_rgb[:, :, 0]*0.1 + img_rgb[:, :, 1]*0 + img_rgb[:, :, 2]*0.9).astype(np.uint8)#1's highest #2's lowest 
    img_gray2  = (img_rgb[:, :, 0]*0.2 + img_rgb[:, :, 1]*0 + img_rgb[:, :, 2]*0.8).astype(np.uint8)  
    img_gray3  = (img_rgb[:, :, 0]*0.2 + img_rgb[:, :, 1]*0.8 + img_rgb[:, :, 2]*0).astype(np.uint8)#2's highest
    img_gray4  = (img_rgb[:, :, 0]*0.4 + img_rgb[:, :, 1]*0 + img_rgb[:, :, 2]*0.6).astype(np.uint8)
    img_gray5  = (img_rgb[:, :, 0]*1 + img_rgb[:, :, 1]*0 + img_rgb[:, :, 2]*0).astype(np.uint8)#1's lowest

    ### TODO ###
    # JBF & BF
    JBF = Joint_bilateral_filter(args.sigma_s, args.sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)
    jbf_out1 = JBF.joint_bilateral_filter(img_rgb, img_gray1).astype(np.uint8)
    jbf_out2 = JBF.joint_bilateral_filter(img_rgb, img_gray2).astype(np.uint8)
    jbf_out3 = JBF.joint_bilateral_filter(img_rgb, img_gray3).astype(np.uint8)
    jbf_out4 = JBF.joint_bilateral_filter(img_rgb, img_gray4).astype(np.uint8)
    jbf_out5 = JBF.joint_bilateral_filter(img_rgb, img_gray5).astype(np.uint8)
    # cost computation
    cost = np.sum(np.abs(jbf_out.astype(np.int32) - bf_out.astype(np.int32)))
    print(cost)
    cost1 = np.sum(np.abs(jbf_out1.astype(np.int32) - bf_out.astype(np.int32)))
    print(cost1)
    cost2 = np.sum(np.abs(jbf_out2.astype(np.int32) - bf_out.astype(np.int32)))
    print(cost2)
    cost3 = np.sum(np.abs(jbf_out3.astype(np.int32) - bf_out.astype(np.int32)))
    print(cost3)
    cost4 = np.sum(np.abs(jbf_out4.astype(np.int32) - bf_out.astype(np.int32)))
    print(cost4)
    cost5 = np.sum(np.abs(jbf_out5.astype(np.int32) - bf_out.astype(np.int32)))
    print(cost5)


    cv2.imwrite('/Users/mingtzu/Desktop/2_highest_RGB.png', cv2.cvtColor(jbf_out3, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/Users/mingtzu/Desktop/2_highest_gray.png', img_gray3)
    cv2.imwrite('/Users/mingtzu/Desktop/2_lowest_RGB.png', cv2.cvtColor(jbf_out1, cv2.COLOR_RGB2BGR))
    cv2.imwrite('/Users/mingtzu/Desktop/2_lowest_gray.png', img_gray1)

    # cv2.imwrite('/Users/mingtzu/Desktop/1_highest_RGB.png', cv2.cvtColor(jbf_out1, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('/Users/mingtzu/Desktop/1_highest_gray.png', img_gray1)
    # cv2.imwrite('/Users/mingtzu/Desktop/1_lowest_RGB.png', cv2.cvtColor(jbf_out5, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('/Users/mingtzu/Desktop/1_lowest_gray.png', img_gray5)
if __name__ == '__main__':
    main()