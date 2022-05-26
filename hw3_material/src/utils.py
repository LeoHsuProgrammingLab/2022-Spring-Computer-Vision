from random import sample
import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]

    H = None 

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    ux = u[:, 0]
    uy = u[:, 1]
    vx = v[:, 0]
    vy = v[:, 1]
    
    # TODO: 1.forming A
    A = np.zeros((2*N, 9))
    for i in range(N):
        A[2*i] = [ ux[i], uy[i], 1, 0, 0, 0, (-1)*ux[i]*vx[i], (-1)*uy[i]*vx[i], (-1)*vx[i] ] 
        A[2*i + 1] = [ 0, 0, 0, ux[i], uy[i], 1, (-1)*ux[i]*vy[i], (-1)*uy[i]*vy[i], (-1)*vy[i] ]
    
    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    H = VT[-1, :]/VT[-1, -1]
    H = H.reshape((3, 3))
#     https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='f'):

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # H*src = dst in forward warping
    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax), sparse = False)
    
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    # I use 3*N
    grid_x = xx.reshape((1, -1))
    grid_y = yy.reshape((1, -1))
    ones = np.ones((1, grid_x.shape[1]))
    grid = np.zeros((3, grid_x.shape[1]))
    grid[0] = grid_x
    grid[1] = grid_y
    grid[2] = ones

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        wanted_src_info = np.dot(H_inv, grid)
        wanted_src_info = wanted_src_info/wanted_src_info[2]
        wanted_src_info = np.round(wanted_src_info).astype(np.int16)
        want_posX_src = wanted_src_info[0].reshape((ymax-ymin, xmax-xmin))
        want_posY_src = wanted_src_info[1].reshape((ymax-ymin, xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        want_sampleX_src = (want_posX_src<w_src)&(want_posX_src>=0)
        want_sampleY_src = (want_posY_src<h_src)&(want_posY_src>=0)
        # this is what I wanna sample in dst, not in src
        want_sample_src = want_sampleY_src&want_sampleX_src

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax, xmin:xmax][want_sample_src] = src[ want_posY_src[want_sample_src], want_posX_src[want_sample_src] ]  

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_v = np.dot(H, grid)
        new_v = new_v/new_v[2]
        new_v = np.round(new_v).astype(np.int16)
        posX_dst = new_v[0].reshape((ymax-ymin, xmax-xmin))
        posY_dst = new_v[1].reshape((ymax-ymin, xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        limitX_dst = np.clip(posX_dst, 0, w_dst - 1)#position equals dimension - 1 
        limitY_dst = np.clip(posY_dst, 0, h_dst - 1)
    
        # TODO: 5.filter the valid coordinates using previous obtained mask
        # TODO: 6. assign to destination image using advanced array indicing
        dst[limitY_dst, limitX_dst] = src
        
    return dst

"""
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """
