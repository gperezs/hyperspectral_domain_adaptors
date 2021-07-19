import os
import sys
import torch
import random
import numpy as np
from skimage import io
from scipy.io import loadmat
from random import seed, sample
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms

from config import centers, means, stds

class CarsDataset(Dataset):
    """Cars dataset."""

    def __init__(self, root_dir, is_train=True, numdata=8144, nchannels=3, is_pretraining=False, mean_needed=True, num_views=0, is_linear=False, array=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Train set if True, Validation set otherwise.
            numdata (int): Number of samples used to train.
            nchannels (int): 3 for RGB, 5 and 15 for multi-channel arrays
        """
        numdata = max(196, numdata)

        #get labels
        filenames = []
        labels = []
        if is_train:
            mat = loadmat(os.path.join(root_dir, 'devkit', 'cars_train_annos.mat'))
            for i in range(len(mat['annotations'][0])):
                labels.append(mat['annotations'][0,i][4][0][0])
                filenames.append(mat['annotations'][0,i][5][0])
        else:
            mat = loadmat(os.path.join(root_dir, 'devkit', 'cars_test_annos_withlabels.mat'))
            for i in range(len(mat['annotations'][0])):
                labels.append(mat['annotations'][0,i][4][0][0])
                filenames.append(mat['annotations'][0,i][5][0])

        self.files = []
        self.labels = []
        if is_train:
            # randomizing indices for when numdata < max (useful when calculating performance with
            # less number of training samples)
            seed(0)
            rndidx = sample(range(len(filenames)), len(filenames))
            for i in rndidx:
                self.files.append(filenames[i])
                self.labels.append(labels[i]-1)
        else:
            for i in range(len(labels)):
                self.files.append(filenames[i])
                self.labels.append(labels[i]-1)
        
        labels = self.labels.copy()
        files = self.files.copy()
        if is_train:
            ftemp = []
            ltemp = []
            while labels:
                idxs = []
                for i in range(len(set(labels))):
                    if len(set(labels)) == 196:
                        if i in set(labels):
                            ltemp.append(labels[labels.index(i)])
                            ftemp.append(files[labels.index(i)])
                            idxs.append(labels.index(i))
                        else: continue
                    else:
                        ltemp += labels
                        ftemp += files
                        labels = []
                        break
                for idx in idxs:
                    files[idx], labels[idx] = 'rm', 'rm'
                files = list(filter(('rm').__ne__, files))
                labels = list(filter(('rm').__ne__, labels))
            self.files = ftemp[0:int(numdata)]
            self.labels = ltemp[0:int(numdata)]

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
        self.meank[5] = means['cars5']
        self.stdk[5] = stds['cars5']
        self.meank[15] = means['cars15']
        self.stdk[15] = stds['cars15']
        self.root_dir = root_dir
        self.mean = np.array([0.485, 0.456, 0.406]) # imagenet
        self.std = np.array([0.229, 0.224, 0.225]) # imagenet
        self.is_train = is_train
        self.centers = {}
        self.centers[5] = centers['5'] # kmeans centers for multi-channel arrays
        self.centers[15] = centers['15'] # kmeans centers for multi-channel arrays
        self.nchannels = nchannels # number of channels of sample
        self.is_pretraining = is_pretraining # True when pre-training adaptors

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.is_train:
            img_name = os.path.join(self.root_dir, 'cars_train', self.files[idx])
        else:
            img_name = os.path.join(self.root_dir, 'cars_test', self.files[idx])

        label = [self.labels[idx]]
        image = io.imread(img_name)

        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        image = resize(image, (224,224,3), anti_aliasing=False, mode='reflect')
        
        image = image.transpose((2, 0, 1))

        if self.is_train:
            image = self.RandomFlip(image)
        
        if self.nchannels==3:
            image = (image - self.mean.reshape(-1,1,1))/self.std.reshape(-1,1,1)
        elif self.views:
            image = self.toKchannels(image, self.nchannels) # convert RGB image to multi-channel
        else:
            image = self.toKchannels(image, self.nchannels) # convert RGB image to multi-channel

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

    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    
    def toKchannels(self, im, k):
        """
        Convert RGB image to multi-channel array using kmeans codebook
        Args:
            k (int): number of channels of output array (5 or 15)
        """
        d, w, h = tuple(im.shape)
        assert d == 3
        image = np.zeros((k, w, h))
        for l in range(k):
            image[l,:,:] = np.exp(-(np.sum((im-self.centers[k][l].reshape(-1,1,1)/255.)**2, axis=0)))
        return image


    def RandomFlip(self, image):
        """Flip randomly the image in a sample."""
        sr = np.random.randint(0, 2)
        if sr:
            return np.flip(image, axis=2)
        else:
            return image

