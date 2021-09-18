import numpy as np
import cv2
import torch
import random

class extreme_transform:
    def __init__(self):
        pass
    def __call__(self, img): # torch.tensor: (n_channels, n, n), range: [0, 1]
        img= img.permute(1,2,0) # torch.tensor: (n, n, n_channels)
        img_np= (img.numpy()*255.0).astype('uint8')
        extaug_covered = torch.tensor(self.extremeaug(img_np)).permute(2,0,1)
        return extaug_covered.float()/255.0

    def extremeaug(self, img_IR):
        """
        Creates extreme condition data using image processing techniques.
        Input : 
            path_img_IR = path to image
            num_dots  = number of dark kernels
        """
        num_dots = 8
        edge_map = cv2.Canny(img_IR,90, 150)

        l_start = 160
        r_end = 0 
        for i in range(img_IR.shape[0]):
            row =  edge_map[i,:]
            index = np.array(np.where(row == 255))
            if index.shape[1]!= 0:
                index_1 = index[0,0]
                index_2 = index[0,-1]
                if index_1 < l_start:
                    l_start = index_1
                if index_2 > r_end:
                    r_end = index_2
        if l_start == 160:
            l_start = 40
        if l_start - 20 < 0:
            l_start = 16
        if r_end == 0:
            r_end = 100  
        if r_end + 20 > 119:
            r_end = 104
        if l_start+20 >= r_end-20:
            l_start =40
            r_end = 100 
        start_point = random.randint(20,50)
        img_IR[start_point:,:,:] = img_IR[start_point:,:,:]//2
        rand_x = []
        rand_y = []
        c = 0
        while num_dots > 0:
            y = random.randint(start_point+20,130)
            x = random.randint(l_start+20,r_end-20)
            c+=1

            if c ==300:
                # print(c)
                break
            if x in rand_x:
                continue
            if y in rand_x:
                continue
            rand_x.append(x)
            rand_y.append(y)
            
            num_dots-=1
            img_IR[y-10:y+10,x-10:x+10,:] = img_IR[159,119,:] #0
            img_IR[y-20:y+20,x-20:x+20,:] = cv2.GaussianBlur(img_IR[y-20:y+20,x-20:x+20,:],(5,5),20,20)

        
        if img_IR.min() <=20:
            diff = 20-img_IR.min()
            img_IR[start_point:,:,:] = img_IR[start_point:,:,:] + diff

        kernel = np.ones((3,3),np.uint8)
        img_IR = cv2.erode(img_IR,kernel,iterations = 1)

        img_IR = cv2.GaussianBlur(img_IR,(3,3),10,10)
        return img_IR

