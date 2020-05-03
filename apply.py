import argparse
import os
import numpy as np
import math
import itertools  
import time
import datetime
import sys


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import *
#from dataset2 import *
from dataset import *
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch

try:
    import ipdb
except:
    import pdb as ipdb


parser = argparse.ArgumentParser()     #创建解析器对象 可以添加参数

parser.add_argument('--dataset_name', type=str, default="zi2zi_canshu_pix_2", help='name of the dataset')
opt = parser.parse_args()
print(opt)



os.makedirs('test_images/%s'%(opt.dataset_name), exist_ok=True)  #过程图片


cuda = True if torch.cuda.is_available() else False

generator = GeneratorUNet()



if cuda:
    generator = generator.cuda()
    
    
transforms_ = [ transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]


val_dataloader = DataLoader(ImageDataset("/mnt/pix/b/train_500_npy" , transforms_=transforms_, mode='test'),
                            batch_size=20, shuffle=False, num_workers=1)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    
filename = os.listdir(r"saved_models/%s/"%(opt.dataset_name))
#ipdb.set_trace()

for j in range(len(filename)):
    if os.path.splitext(filename[j])[1] == '.pth':
        generator.load_state_dict(torch.load('saved_models/%s/%s'%(opt.dataset_name ,filename[j]) ))
        generator.eval()
        for i, batch in enumerate(val_dataloader):
            print(i)      
            real_A = Variable(batch['B'].type(Tensor))
            real_B = Variable(batch['A'].type(Tensor))
            fake_B = generator(real_A)        
            save_image(fake_B, 'test_images/%s/%s.png'%(opt.dataset_name, (filename[j] + str(i) + '_' + str(i))) , nrow=10, normalize=True)
            save_image(real_B, 'test_images/%s/%s.png'%(opt.dataset_name, str(i)) , nrow=10, normalize=True)
        

        
   

'''
for j in range(len(filename)):
    generator.load_state_dict(torch.load('saved_models/%s/%s'%(opt.dataset_name ,filename[j]) ))
    generator.eval()
    for i, batch in enumerate(val_dataloader):
        print(i)      
        real_A = Variable(batch['B'].type(Tensor))
        real_B = Variable(batch['A'].type(Tensor))
        fake_B = generator(real_A)        
        save_image(fake_B, 'test_images/%s/%s.png'%(opt.dataset_name,(filename[j])) , nrow=10, normalize=True)
'''


'''
generator.load_state_dict(torch.load('saved_models/%s/generator_99.pth'%(opt.dataset_name) ))
generator.eval()
for i, batch in enumerate(val_dataloader):
    12
    real_A = Variable(batch['B'].type(Tensor))
    real_B = Variable(batch['A'].type(Tensor))
    fake_B = generator(real_A)        
    save_image(fake_B, 'test_images/haojieguo_train/%s.png'%(i) , nrow=10, normalize=True)
'''
            


    
    
    
'''    
generator.load_state_dict(torch.load('saved_models/newtest_27/generator_299.pth'))

generator.eval()






for i, batch in enumerate(val_dataloader):
    print(i)
    if i == 0:
        
        real_A = Variable(batch['B'].type(Tensor))
        real_B = Variable(batch['A'].type(Tensor))
        fake_B = generator(real_A)        
        save_image(fake_B, 'test_images/newtest_27/minmodel_299.png' , nrow=10, normalize=True)
        
    else:
        break
      
'''    



    
    

    
    
    
    
 