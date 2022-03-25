import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
    
        ### TODO ###
        #height & width of the guidance and img
        h, w = guidance.shape[0:2]
        output = np.zeros(img.shape)
        #Notes:
        # a. Normalize the RGB pic to [0, 1] by divide by 255 only for Tp-Tq
        # b. time:1.28s
        # c. Use the value in txt. file
        # d. turn unit8 img to unit32 

        # a.build gr table & normalize
        # e. only one for loop
        # 1.find the grayscale to generate the pic(parameters selection)
        # 2.Do Joint Bilateral Filter by grayscale pic and the original pic(with many different guidance gray pic)
        # 3.Find the lowest cost between different JBF & BF
        # 4.Cost L1 Normalization
        padded_guidance = padded_guidance.astype(np.float64)
        padded_img = padded_img.astype(np.float64)
        # Build Gs Table:
        Gs_table = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                Gs_table[i, j] = np.exp( ((i-self.pad_w)**2 + (j-self.pad_w)**2) / ((-2)*(self.sigma_s**2)) )
        Gs_table = Gs_table.reshape((-1, 1, 1))
        
        # Build Gr Table
        Gr_table = np.zeros(256,)
        for i in range(256):
            Gr_table[i] = np.exp( ((i/255)**2) / ((-2)*(self.sigma_r**2)) )
        
        #preprocessing the padded_img Iq
        cube_img_r = np.zeros((self.wndw_size**2, h, w))
        cube_img_g = np.zeros((self.wndw_size**2, h, w))
        cube_img_b = np.zeros((self.wndw_size**2, h, w))
        for id in range(self.wndw_size**2):
            #w
            i = id%self.wndw_size
            #h
            j = id//self.wndw_size
            cube_img_r[id, :, :] = padded_img[j:j+h, i:i+w, 0]
            cube_img_g[id, :, :] = padded_img[j:j+h, i:i+w, 1]
            cube_img_b[id, :, :] = padded_img[j:j+h, i:i+w, 2]
        #JBF
        if (len(guidance.shape) == 3):
            cube_r = np.zeros((self.wndw_size**2, h, w)).astype(int)
            cube_g = np.zeros((self.wndw_size**2, h, w)).astype(int)
            cube_b = np.zeros((self.wndw_size**2, h, w)).astype(int)
            for id in range(self.wndw_size**2):
                #w
                i = id%self.wndw_size
                #h
                j = id//self.wndw_size
                #(h, w, c) for padded_guidance
                cube_r[id, :, :] = guidance[:, :, 0] - padded_guidance[j:j+h, i:i+w, 0]
                cube_g[id, :, :] = guidance[:, :, 1] - padded_guidance[j:j+h, i:i+w, 1]
                cube_b[id, :, :] = guidance[:, :, 2] - padded_guidance[j:j+h, i:i+w, 2]
               
            Gr = Gr_table[np.abs(cube_r)] * Gr_table[np.abs(cube_g)] * Gr_table[np.abs(cube_b)]
        #BF
        else:
            cube = np.zeros((self.wndw_size**2, h, w)).astype(int)
            for id in range(self.wndw_size**2):
                #w
                i = id%self.wndw_size
                #h
                j = id//self.wndw_size
                cube[id] = guidance - padded_guidance[j:j+h, i:i+w]
            Gr = Gr_table[np.abs(cube)]
        
        # implement output
        parameter = Gs_table*Gr
        denominator = parameter.sum(axis = 0)
        
        output[:, :, 0] = (parameter*cube_img_r).sum(axis = 0)/denominator
        output[:, :, 1] = (parameter*cube_img_g).sum(axis = 0)/denominator
        output[:, :, 2] = (parameter*cube_img_b).sum(axis = 0)/denominator
        
        return np.clip(output, 0, 255).astype(np.uint8)