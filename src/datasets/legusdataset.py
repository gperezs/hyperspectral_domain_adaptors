import os
import sys
import torch
import random
import numpy as np
from skimage import io
from random import seed, sample
from scipy.ndimage import rotate
from skimage.transform import resize
from torch.utils.data import Dataset

from config import centers, means, stds

sys.path.insert(0, './src/utils')


class LEGUSDataset(Dataset):
    """LEGUS dataset."""

    def __init__(self, root_dir, is_train=True, numdata=12376, nchannels=5, is_pretraining=False, num_views=0, is_linear=False, array=0, is_starcnet=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Train set if True, Validation set otherwise.
            numdata (int): Number of samples used to train.
            nchannels (int): 3 for RGB, 5 and 15 for multi-channel arrays
        """
        numdata  = max(100, numdata)

        # get data and labels
        if is_train:
            # to create npy file with dataset check https://github.com/gperezs/StarcNet repo:
            files = np.load(os.path.join(root_dir, 'trainval_raw_32x32_data.npy'))
            labels = np.load(os.path.join(root_dir, 'trainval_raw_32x32_label.npy'))
            # balancing dataset
            # Indices of objects of each class
            c1 = np.where(labels == 0)[0]
            c2 = np.where(labels == 1)[0]
            c3 = np.where(labels == 2)[0]
            c4 = np.where(labels == 3)[0]
            # Augmentation factor for each class, so is approx. balanced
            m1 = np.round(len(c4)/len(c1))
            m2 = np.round(len(c4)/len(c2))
            m3 = np.round(len(c4)/len(c3))
            # Augmentation of dataset to balance it
            data1, label1 = self.reflection_augment(files[c1], labels[c1], m1)
            data2, label2 = self.reflection_augment(files[c2], labels[c2], m2)
            data3, label3 = self.reflection_augment(files[c3], labels[c3], m3)
            data4, label4 = files[c4], labels[c4] # Class 4 has more objects that the other 3
            data = np.concatenate((data1, data2, data3, data4), axis=0)
            label = np.concatenate((label1, label2, label3, label4), axis=0)
        else: 
            files = np.load(os.path.join(root_dir, 'test_raw_32x32_data.npy'))
            labels = np.load(os.path.join(root_dir, 'test_raw_32x32_label.npy'))

        if is_train:
            # randomizing indices for when numdata < max (useful when calculating performance with
            # less number of training samples)
            seed(0)
            rndidx = sample(range(len(data)), len(data))
            self.files = []
            self.labels = []
            for i in rndidx:
                self.files.append(data[i])
                self.labels.append(label[i])
        else:
            self.files = files
            self.labels = labels

        
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
        self.meank[5] = means['legus']
        self.stdk[5] = stds['legus']
        self.root_dir = root_dir
        self.mean = np.array([0.485, 0.456, 0.406]) # imagenet
        self.std = np.array([0.229, 0.224, 0.225]) # imagenet
        self.is_train = is_train
        self.is_pretraining = is_pretraining # True when pre-training adaptors
        self.nchannels = nchannels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.labels[idx]]
        nimage = self.files[idx]

        image = np.zeros((5,224,224))
        for i in range(5):
            image[i,:,:] = resize(nimage[i,:,:], (224,224), anti_aliasing=False, mode='reflect')

        if self.nchannels==3:
            datac = np.zeros((3,image.shape[1],image.shape[2]))
            datac[2,:,:] = (21.63*image[0,:,:] + 8.63*image[1,:,:]) / 2.
            datac[1,:,:] = (4.52*image[2,:,:])
            datac[0,:,:] = (1.23*image[3,:,:] + image[4,:,:]) / 2.
            image = (datac - self.mean.reshape(-1,1,1))/self.std.reshape(-1,1,1)
            
        if self.is_train:
            image = self.RandomRotation(image)
        
        image = (image - image.min())/(image.max() - image.min())

        if self.is_pretraining:
            sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(image)}
            return sample

        if self.views and not self.is_linear:
            imagemv = np.zeros((len(self.views),3,224,224))
            for i in range(len(self.views)):
                imagemv[i,:,:,:] = image[self.views[i],:,:]
                imagemv[i,:,:,:] = (imagemv[i,:,:,:] - self.mean.reshape(-1,1,1))/self.std.reshape(-1,1,1)
            sample = {'image': torch.from_numpy(imagemv), 'label': torch.LongTensor(label)}
            return sample
        
        sample = {'image': torch.from_numpy(image), 'label': torch.LongTensor(label)}
        return sample

    def RandomRotation(self, image):
        """ Randomly rotate image """
        sc = np.random.randint(0, 3)
        if sc == 0:
            image = rotate(image, 90, axes=(1,2))
        elif sc == 1:
            image = rotate(image, 270, axes=(1,2))
        elif sc == 2:
            return image
        return image


