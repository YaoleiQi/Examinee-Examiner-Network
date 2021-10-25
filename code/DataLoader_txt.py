# -*- coding: utf-8 -*-

from os.path import exists, join
from os import listdir
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

def read_file_from_txt(txt_path):
    files=[]
    for line in open(txt_path, 'r'):
        files.append(line.strip())
    return files

class DataLoad(data.Dataset):
    def __init__(self, i_path, l_path, m_path, shape):
        super(DataLoad, self).__init__()
        self.image_file=read_file_from_txt(i_path)
        self.label_file=read_file_from_txt(l_path)
        self.mask_file=read_file_from_txt(m_path)        
        self.shape = shape
        
        '''
        If the original image is preprocessed, it is not needed here, if not, it needs to be added
        '''
        # meanstd = np.load('meanstd.npy')
        # self.mean = meanstd[0]
        # self.std = meanstd[1]
    def __getitem__(self, index):
        image_path=self.image_file[index]
        label_path=self.label_file[index]
        mask_path=self.mask_file[index]

        '''
        Since we added preprocessing, the image is saved as a float, depending on the type of data to be processed
        '''
        image = np.fromfile(file=image_path, dtype=np.float32)
        target1 = np.fromfile(file=label_path, dtype=np.float32)
        target2 = np.fromfile(file=mask_path, dtype=np.float32)
        
        image_name = str(image_path.split('/')[-1])
        print(image_name)

        '''
        Read in the cropped dimensions here. This address is the address where the dimensions are stored.
        '''
        shape1 = np.load('data/ASOCA_afterprocess/Npy/'+s.rstrip('.raw')+'.npy')
        x = shape1[2]
        y = shape1[1]
        z = shape1[0]
        
        image = image.reshape(z, y, x)
        image = image.astype(np.float32)
        target1 = target1.reshape(z, y, x)
        target2 = target2.reshape(z, y, x)

        
        if self.shape[0] > z:
            z = self.shape[0]   
            image = reshape_img(image, z, y, x)
            target1 = reshape_img(target1, z, y, x) 
            target2 = reshape_img(target2, z, y, x) 
        if self.shape[1] > y:
            y = self.shape[1]
            image = reshape_img(image, z, y, x)
            target1 = reshape_img(target1, z, y, x) 
            target2 = reshape_img(target2, z, y, x)  
        if self.shape[2] > x:
            x = self.shape[2]
            image = reshape_img(image, z, y, x)
            target1 = reshape_img(target1, z, y, x) 
            target2 = reshape_img(target2, z, y, x) 
               
            
        '''
        Filter the random upper left corner points for random cropping.
        '''
        center_z = np.random.randint(0, z - self.shape[0] + 1, 1, dtype=np.int)[0]
        center_y = np.random.randint(0, y - self.shape[1] + 1, 1, dtype=np.int)[0]
        center_x = np.random.randint(0, x - self.shape[2] + 1, 1, dtype=np.int)[0]


        image = image[center_z:self.shape[0] +
                               center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]
        target1 = target1[center_z:self.shape[0] +
                               center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]
        target2 = target2[center_z:self.shape[0] +
                               center_z, center_y:self.shape[1] + center_y, center_x:self.shape[2] + center_x]

        #image = (image - self.mean) / self.std

        image = image.astype(np.float32)  
        target1 = target1.astype(np.float32)
        target2 = target2.astype(np.float32)

        
        image = image[np.newaxis, :, :, :]
        target1 = target1[np.newaxis, :, :, :]
        target2 = target2[np.newaxis, :, :, :]

        return image, target1, target2

    def __len__(self):
        return len(self.image_file)


'''
If the cut size is larger than the original size.
'''
def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
    return out

    




