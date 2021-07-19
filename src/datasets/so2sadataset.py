import os
import sys
import h5py
import torch
import torchvision
import random
import numpy as np
from skimage import io
from random import seed, sample
from skimage.transform import resize
from torch.utils.data import Dataset

from config import centers, means, stds

class So2SaDataset(Dataset):
    """So2Sa LCZ42  dataset."""

    def __init__(self, root_dir, is_train=True, numdata=5994, nchannels=3, is_pretraining=False, mean_needed=True, num_views=0, is_linear=False, array=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Train set if True, Validation set otherwise.
            numdata (int): Number of samples used to train.
            nchannels (int): 3 for RGB, 5 and 15 for multi-channel arrays
            
            *Sentinel-1 total training samples: (352366,32,32,8)
            *Sentinel-2 total training samples: (352366,32,32,10)
            *Sentinel-1 total validation samples: (24119,32,32,8)
            *Sentinel-2 total validation samples: (24119,32,32,10)
        """
        numdata = max(100, numdata)
        
        if is_train:
            if numdata <= 10000:
                filename = 'training_small.h5'
            else:
                filename = 'training.h5'
        else:
            if numdata <= 10000:
                filename = 'validation_small.h5'
            else:
                filename = 'validation.h5'

        if numdata <= 10000: #and is_train:
            fid = h5py.File(os.path.join(root_dir, filename),'r')
            files = np.array(fid['data'])
            labels = np.array(fid['label'])
        else:
            fid = h5py.File(os.path.join(root_dir, filename),'r')
            files = np.concatenate((np.array(fid['sen1']), np.array(fid['sen2'])), axis=3)
            labels = np.array(fid['label'])
            labels = np.argmax(labels, axis=1) + 1

        seed(0)
        rndidx = sample(range(len(files)), len(files))
        self.files = []
        self.labels = []
        for i in rndidx:
            self.files.append(files[i])
            self.labels.append(labels[i])

        if is_train:
            self.files = self.files[0:int(numdata)]
            self.labels = self.labels[0:int(numdata)]

        # For multi-view:
        views = []
        for i in range(num_views):
            random.seed(i+array)
            views.append(random.sample(range(nchannels), 3))
        print('views: %s'%(str(views)))

        self.is_linear = is_linear
        self.views = views
        self.meank = {}
        self.stdk = {}
        self.meank[18] = means['so2sa']
        self.stdk[18] = stds['so2sa']
        self.root_dir = root_dir
        self.mean = np.array([0.485, 0.456, 0.406]) # imagenet
        self.std = np.array([0.229, 0.224, 0.225]) # imagenet
        self.is_train = is_train
        self.nchannels = nchannels # number of channels of sample
        self.is_pretraining = is_pretraining # True when pre-training adaptors


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.labels[idx]]
        image = self.files[idx] 
        
        image = np.zeros((18,224,224))
        for i in range(18):
            image[i,:,:] = resize(nimage[i,:,:], (224,224), anti_aliasing=False, mode='reflect')

        if self.is_train:
            image = self.RandomFlip(image)

        if self.nchannels==3:
            image = np.zeros((3,224,224))
            for i in range(3):
                image[i,:,:] = resize(nimage0[i,:,:], (224,224), anti_aliasing=False, mode='reflect')
            image = (image - self.mean.reshape(-1,1,1))/self.std.reshape(-1,1,1)
        elif not self.views:
            image = (image - self.meank[self.nchannels].reshape(-1,1,1))/self.stdk[self.nchannels].reshape(-1,1,1)
        elif self.views and self.is_linear:
            image = (image - self.meank[self.nchannels].reshape(-1,1,1))/self.stdk[self.nchannels].reshape(-1,1,1)

        if self.is_pretraining:
            image = image.transpose(2,0,1)
            image = (image - self.meank[self.nchannels].reshape(-1,1,1))/self.stdk[self.nchannels].reshape(-1,1,1)
            image = image.transpose(1,2,0)
            image = tResize(toTensor(image))#.numpy()
            sample = {'image': image, 'label': torch.from_numpy(image)}
            return sample

        if self.views and not self.is_linear:
            imagemv = np.zeros((len(self.views),3,224,224))
            for i in range(len(self.views)):
                imagemv[i,:,:,:] = image[self.views[i],:,:]
                imagemv[i,:,:,:] = (imagemv[i,:,:,:] - self.mean.reshape(-1,1,1))/self.std.reshape(-1,1,1)
            sample = {'image': torch.from_numpy(imagemv), 'label': torch.LongTensor(label)}
            return sample

        sample = {'image': tResize(toTensor(image)), 'label': torch.LongTensor(label)}
        return sample

    def RandomFlip(self, image):
        """Flip randomly the image in a sample."""
        sr = np.random.randint(0, 2)
        if sr:
            return np.flip(image, axis=2).copy()
        else:
            return image


