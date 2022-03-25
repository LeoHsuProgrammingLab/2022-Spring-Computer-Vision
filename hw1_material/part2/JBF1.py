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

        output = np.empty((img.shape))
        i = 3*self.sigma_s
        j = 3*self.sigma_s

        Gs_pixelx = np.full((self.wndw_size, 1), i)
        Gs_pixelx = np.expand_dims(np.tile(Gs_pixelx, (1, self.wndw_size)), axis=0)
        Gs_pixely = np.full((1, self.wndw_size), j)
        Gs_pixely = np.expand_dims(np.tile(Gs_pixely, (self.wndw_size, 1)), axis=0)
        Gs_pixel = np.concatenate((Gs_pixelx, Gs_pixely ), axis=0)

        Gs_nearx = np.arange((i-3*self.sigma_s), (i+3*self.sigma_s+1)).reshape(-1,1)
        Gs_nearx = np.expand_dims(np.tile(Gs_nearx, (1, self.wndw_size)), axis=0)
        Gs_neary = np.arange((j-3*self.sigma_s), (j+3*self.sigma_s+1)).reshape(1,-1)
        Gs_neary = np.expand_dims(np.tile(Gs_neary, (self.wndw_size, 1)), axis=0)
        Gs_near = np.concatenate((Gs_nearx, Gs_neary ), axis=0)
        Gs_table = Joint_bilateral_filter.spatial_kernel(self, Gs_pixel, Gs_near).reshape(-1, 1, 1)
        
        # Gs_table = np.zeros((self.wndw_size, 1, 1))
        # for i in range(self.wndw_size) :
        #     for j in range(self.wndw_size) :
        #         Gs_table[i+j, 1, 1] = Joint_bilateral_filter.spatial_kernel(self, )

        
        Gr = np.empty((self.wndw_size**2, img.shape[0], img.shape[1]))
        Iq = np.empty((self.wndw_size**2, img.shape[0], img.shape[1], img.shape[2]))
        for i in range(self.wndw_size) :
            for j in range(self.wndw_size) :
                if guidance.ndim == 2:
                    Gr_padded = padded_guidance[i:i+img.shape[0], j:j+img.shape[1]]
                else :
                    Gr_padded = padded_guidance[i:i+img.shape[0], j:j+img.shape[1], :]
                Gr[i*self.wndw_size+j, :, :] = Joint_bilateral_filter.range_kernel(self, guidance, Gr_padded)
                Iq[:, i:i+1, j:j+1,:] = padded_img[i:i+self.wndw_size, j:j+self.wndw_size, :].reshape(-1, 1, 1, padded_img.shape[2])
        print('Gr', Gr.shape)
        print('Gs', Gs_table.shape)
        print('Gr*Gs', (Gs_table*Gr).shape)
        

        mul = Gs_table*Gr
        
        frac = np.expand_dims(mul, axis=-1)*Iq
        for i in range(img.shape[0]) :
            for j in range(img.shape[1]) :
                output[i, j, :] = np.sum(frac[:,i, j,:], axis=0) / np.sum((np.expand_dims(mul, axis = -1))[:,i, j,:], axis=0)
        print('output', output.shape)

        ########################################################################################################
        # mole = np.sum(mul)
        # print('mul*Iq', (mul*Iq).shape)
        # frac = np.sum(mul*Iq, axis =-1)
        # print('frac', frac.shape)
        # frac = np.sum(frac, axis =-1)
        

        #output[i-self.pad_w, j-self.pad_w] = frac / mole

        # while i < (padded_img.shape[0]-3*self.sigma_s) :
        #     j = 3*self.sigma_s
        #     while j < (padded_img.shape[1]-3*self.sigma_s) :
        #         Gr = np.empty((self.wndw_size, self.wndw_size))
        #         if img.ndim == 2 :
        #             img_kernel = np.expand_dims(padded_img[ (i-3*self.sigma_s):(i+3*self.sigma_s+1) , (j-3*self.sigma_s):(j+3*self.sigma_s+1)], axis=-1)
        #         else :
        #             img_kernel = padded_img[ (i-3*self.sigma_s):(i+3*self.sigma_s+1) , (j-3*self.sigma_s):(j+3*self.sigma_s+1),: ]
        #         if guidance.ndim == 2:
        #             gui_kernel = np.expand_dims(padded_guidance[ (i-3*self.sigma_s):(i+3*self.sigma_s+1) , (j-3*self.sigma_s):(j+3*self.sigma_s+1)], axis=-1)
        #         else :
        #             gui_kernel = padded_guidance[ (i-3*self.sigma_s):(i+3*self.sigma_s+1) , (j-3*self.sigma_s):(j+3*self.sigma_s+1),: ]

                
        #         # Gr_pixel = padded_guidance[i:i+1, j:j+1]
        #         # Gr_pixel = np.tile(Gr_pixel, (self.wndw_size, self.wndw_size,1))
        #         # Gr = Joint_bilateral_filter.range_kernel(self, Gr_pixel, gui_kernel)
            
                

        #         mole = np.sum(Gs_table*Gr)
        #         # if img_kernel.shape[2] == 3:
        #         #     Gs = np.expand_dims(Gs_table, axis=-1)
        #         #     Gs = np.tile(Gs, (1,1,3))
        #         #     Gr = np.expand_dims(Gr, axis=-1)
        #         #     Gr = np.tile(Gr, (1,1,3))
        #         # else :
        #         #     Gs = np.expand_dims(Gs, axis=-1)
        #         #     Gr = np.expand_dims(Gr, axis=-1)
        #         frac = np.sum(Gs_table*Gr*img_kernel, axis =-1)
        #         frac = np.sum(frac, axis =-1)

        #         output[i-self.pad_w, j-self.pad_w] = frac / mole
        
        #         j += 1
        #     i += 1

        return np.clip(output, 0, 255).astype(np.uint8)

    def spatial_kernel(self, img, img_near) :
        Gs = np.exp( -((img[0,:,:] - img_near[0,:,:])**2 + (img[1,:,:] - img_near[1,:,:])**2)   /  (2*(self.sigma_s**2)) )
        return Gs

    def range_kernel (self, gui, gui_near):
        gui_normalize = gui / 255
        gui_near_normalize = gui_near / 255
        if gui.ndim == 2 :
            Gr = np.exp( -(gui_normalize[:,:] - gui_near_normalize[:,:])**2  /  (2*(self.sigma_r**2)) )
        else :
            Gr = np.exp( -( (gui_normalize[:,:,0] - gui_near_normalize[:,:,0])**2+(gui_normalize[:,:,1] - gui_near_normalize[:,:,1])**2+(gui_normalize[:,:,2] - gui_near_normalize[:,:,2])**2 ) /  (2*(self.sigma_r**2)) )

        return Gr




# import numpy as np
# import cv2


# class Joint_bilateral_filter(object):
#     def __init__(self, sigma_s, sigma_r):
#         self.sigma_r = sigma_r
#         self.sigma_s = sigma_s
#         self.wndw_size = 6*sigma_s+1
#         self.pad_w = 3*sigma_s
    
#     def joint_bilateral_filter(self, img, guidance):
#         BORDER_TYPE = cv2.BORDER_REFLECT
#         padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
#         padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
    
#         ### TODO ###
#         #height & width of the guidance and img
#         h, w = guidance.shape[0:2]
#         output = np.zeros(img.shape)
#         #Notes:
#         # a. Normalize the RGB pic to [0, 1] by divide by 255 only for Tp-Tq
#         # b. time:1.28s
#         # c. Use the value in txt. file
#         # d. turn unit8 img to unit32 

#         # a.build gr table & normalize
#         # e. only one for loop
#         # 1.find the grayscale to generate the pic(parameters selection)
#         # 2.Do Joint Bilateral Filter by grayscale pic and the original pic(with many different guidance gray pic)
#         # 3.Find the lowest cost between different JBF & BF
#         # 4.Cost L1 Normalization
#         padded_guidance = padded_guidance.astype(np.float64)
#         padded_img = padded_img.astype(np.float64)
#         # Build Gs Table:
#         Gs_table = np.zeros((self.wndw_size, self.wndw_size))
#         for i in range(self.wndw_size):
#             for j in range(self.wndw_size):
#                 Gs_table[i, j] = np.exp( ((i-self.pad_w)**2 + (j-self.pad_w)**2) / ((-2)*(self.sigma_s**2)) )
#         Gs_table = Gs_table.reshape((-1, 1, 1))
        
#         # Build Gr Table
#         Gr_table = np.zeros(256,)
#         for i in range(256):
#             Gr_table[i] = np.exp( ((i/255)**2) / ((-2)*(self.sigma_r**2)) )
        
#         #preprocessing the padded_img Iq
#         cube_img_r = np.zeros((self.wndw_size**2, h, w))
#         cube_img_g = np.zeros((self.wndw_size**2, h, w))
#         cube_img_b = np.zeros((self.wndw_size**2, h, w))
#         for id in range(self.wndw_size**2):
#             #w
#             i = id%self.wndw_size
#             #h
#             j = id//self.wndw_size
#             cube_img_r[id, :, :] = padded_img[j:j+h, i:i+w, 0]
#             cube_img_g[id, :, :] = padded_img[j:j+h, i:i+w, 1]
#             cube_img_b[id, :, :] = padded_img[j:j+h, i:i+w, 2]
#         #JBF
#         if (len(guidance.shape) == 3):
#             cube_r = np.zeros((self.wndw_size**2, h, w)).astype(int)
#             cube_g = np.zeros((self.wndw_size**2, h, w)).astype(int)
#             cube_b = np.zeros((self.wndw_size**2, h, w)).astype(int)
#             for id in range(self.wndw_size**2):
#                 #w
#                 i = id%self.wndw_size
#                 #h
#                 j = id//self.wndw_size
#                 #(h, w, c) for padded_guidance
#                 cube_r[id, :, :] = guidance[:, :, 0] - padded_guidance[j:j+h, i:i+w, 0]
#                 cube_g[id, :, :] = guidance[:, :, 1] - padded_guidance[j:j+h, i:i+w, 1]
#                 cube_b[id, :, :] = guidance[:, :, 2] - padded_guidance[j:j+h, i:i+w, 2]
               
#             Gr = Gr_table[np.abs(cube_r)] * Gr_table[np.abs(cube_g)] * Gr_table[np.abs(cube_b)]
#         #BF
#         else:
#             cube = np.zeros((self.wndw_size**2, h, w)).astype(int)
#             for id in range(self.wndw_size**2):
#                 #w
#                 i = id%self.wndw_size
#                 #h
#                 j = id//self.wndw_size
#                 cube[id] = guidance - padded_guidance[j:j+h, i:i+w]
#             Gr = Gr_table[np.abs(cube)]
        
#         # implement output
#         parameter = Gs_table*Gr
#         denominator = parameter.sum(axis = 0)
        
#         output[:, :, 0] = (parameter*cube_img_r).sum(axis = 0)/denominator
#         output[:, :, 1] = (parameter*cube_img_g).sum(axis = 0)/denominator
#         output[:, :, 2] = (parameter*cube_img_b).sum(axis = 0)/denominator
        
#         return np.clip(output, 0, 255).astype(np.uint8)