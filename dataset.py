import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from sklearn.cross_validation import train_test_split
try:
    import ipdb
except:
    import pdb as ipdb

class ImageDataset(Dataset):
    def __init__(self, npyDataPath, transforms_=None, mode='train'):#mode也不要了 直接npyDataPath就是npy数据路径
        self.transform = transforms.Compose(transforms_)
       # ipdb.set_trace()


        self.imgs = np.load(npyDataPath)
        #print(self.imgs.shape)
       # train,test = train_test_split(self.files, test_size=0.2, random_state=0)
        
    def __getitem__(self, index):
        img_A_a = self.imgs[index][:,:256,:]
        img_B_b = self.imgs[index][:,256:,:]
        
        img_A = self.transform(img_A_a.astype(np.uint8))#jinghei
        img_B = self.transform(img_B_b.astype(np.uint8))#heiti


        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.imgs)