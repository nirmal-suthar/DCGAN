#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import sys
import cv2


# In[10]:


# for resizing the dataset to desired size
def celebA_data_prepocess(root):

    os.chdir(r'/data/namanb/celeb')
    save_root = 'celebA_cropped/'
    resize_size = 64

    


    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root + 'celebA'):
        os.mkdir(save_root + 'celebA')
    
    try:
        img_list = os.listdir(root)
    except:
        print('root directory is invalid')
        return

    for i in range(len(img_list)):
        img = plt.imread(root + img_list[i])
        img = cv2.resize(img, (resize_size,resize_size))
        plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

        if (i % 1000) == 0:
            print('%d images complete' % i)
    print('data processing done!')

    return save_root


# In[15]:


root = 'celeba/images/'        #this path depends on your computer
save_root = celebA_data_prepocess(root)    #save_root contains the path to the new directory containing the resized dataset


# In[8]:


os.chdir(r'/data/namanb/celeba')
data_dir = 'celebA_cropped'         # this path depends on your computer

  
# data_loader
img_size = 64

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dset = datasets.ImageFolder(data_dir, transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=True)

temp = plt.imread(train_loader.dataset.imgs[0][0])


if (temp.shape[0] != img_size) or (temp.shape[0] != img_size):
    sys.stderr.write('Error! image size is not 64 x 64! run \"celebA_data_prepocess\" !!!')
    sys.exit(1)

else:
    plt.imshow(temp)
    print('done!')

