from math import inf
from textwrap import fill
from turtle import left
import numpy as np
import cv2
import cv2.ximgproc as xip
from tqdm import tqdm


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    padded_Il = cv2.copyMakeBorder(Il, 1, 1, 1, 1, 0)
    padded_Ir = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, 0)

    cost_map_l2r = np.zeros((max_disp+1, h, w), dtype = np.float32)
    cost_map_r2l = np.zeros((max_disp+1, h, w), dtype = np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    pixel_value_8pos_l = np.zeros((h, w, 8), dtype = np.float32)
    pixel_value_8pos_r = np.zeros((h, w, 8), dtype = np.float32)

    for i in range(8):
        j = i//3
        k = i%3
        pixel_value_8pos_l[ :, :, i] = np.sum(padded_Il[j:j+h, k:k+w, :] - Il, axis = 2) > 0
        pixel_value_8pos_r[ :, :, i] = np.sum(padded_Ir[j:j+h, k:k+w, :] - Ir, axis = 2) > 0
    # combine RGB channel and compare to Il, Ir
    
    
    
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    for d in tqdm(range(max_disp+1)):
        total_move_cost_l = np.sum(np.logical_xor(pixel_value_8pos_l[:, d:, :], pixel_value_8pos_r[:, :w-d, :]).astype(np.int8), axis = 2)
        cost_map_l2r[d, :, :] = np.concatenate([np.tile(total_move_cost_l[:, :1], d), total_move_cost_l ], axis = 1)

        total_move_cost_r = np.sum(np.logical_xor(pixel_value_8pos_r[:, :w-d, :], pixel_value_8pos_l[:, d:, :]).astype(np.int8), axis = 2)
        cost_map_r2l[d, :, :] = np.concatenate([total_move_cost_r, np.tile(total_move_cost_r[:, -1:], d)], axis = 1)
    
    
    

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparity)
        cost_map_l2r[d, :, :] = xip.jointBilateralFilter(Il, cost_map_l2r[d, :, :], 30, 30, 30)
        cost_map_r2l[d, :, :] = xip.jointBilateralFilter(Ir, cost_map_r2l[d, :, :], 30, 30, 30)
    


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    
    disparity_map_l2r = np.argmin(cost_map_l2r, axis = 0)
    disparity_map_r2l = np.argmin(cost_map_r2l, axis = 0)

    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h), sparse=False)
    disparity_map_l2r[ disparity_map_l2r[yy, xx] != disparity_map_r2l[yy, xx-disparity_map_l2r[yy, xx]] ] = -1

    padding_max = np.max(disparity_map_l2r)
    padded_disparity = np.pad(disparity_map_l2r, ((0, 0), (1, 1)), 'constant', constant_values = padding_max)

    isHole = np.array( np.where(padded_disparity == -1) )
    notHole = np.array( np.where(padded_disparity != -1) ) 
    
    y_axis_of_hole = np.unique(isHole[0])

    for y in tqdm(y_axis_of_hole):
        x_need_change = isHole[1][np.where(isHole[0] == y)]
        x_choice = notHole[1][np.where(notHole[0] == y)]
        for x in x_need_change:
            Fl = np.max(x_choice[x_choice < x])
            Fr = np.min(x_choice[x_choice > x])
            fill_spot = min(padded_disparity[y, Fl], padded_disparity[y, Fr])
            padded_disparity[y, x] = fill_spot

            # padded_disparity[y, x] = padded_disparity[y, np.argmin(np.abs(x_choice - x))]

    disparity_map = padded_disparity[:, 1:1+w]  
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), disparity_map.astype(np.uint8), 10, 30 )

    return labels.astype(np.uint8)
    