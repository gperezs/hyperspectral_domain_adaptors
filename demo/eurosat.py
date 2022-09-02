import os
import sys
import torch
import random
import numpy as np
from skimage import io
from random import seed, sample
from skimage.transform import resize
from torch.utils.data import Dataset

class EuroSATDataset(Dataset):
    """EuroSAT dataset."""

    def __init__(self, 
                 root_dir, 
                 nchannels=13, # EuroSAT number of channels 
                 num_views=5,  # Number of views
                 imsize=224,   # cause we are using ImageNet pretrained models
                 is_train=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Train set if True, Validation set otherwise.
            numdata (int): Number of samples used to train.
            nchannels (int): 3 for RGB, 5 and 15 for multi-channel arrays
        """
        
        root_dir = os.path.join(root_dir, 'images', 'remote_sensing', 'otherDatasets', 'sentinel_2', 'tif')
        classes_dir = sorted(os.listdir(root_dir))

        # get files and labels
        img_files = []
        labelsi = []
        classes = []
        for class_i in classes_dir:
            if class_i not in classes: classes.append(class_i)
            imgfiles = sorted(os.listdir(os.path.join(root_dir,class_i)))
            for filename in imgfiles:
                img_files.append(os.path.join(root_dir,class_i,filename))
                labelsi.append(class_i)
                
        classes = list(set(labelsi))

        # there is no split on their werbsite so randomize everything and train/test split
        trainsize = 10000 # number of samples to train (use the remaining for evaluation)
        rndidx = sample(range(len(img_files)), len(img_files))
        files = []
        labels = []
        for i in rndidx:
            files.append(img_files[i])
            labels.append(labelsi[i])
        
        if is_train:
            files = files[0:trainsize]
            labels = labels[0:trainsize]
        else:
            files = files[trainsize:]
            labels = labels[trainsize:]

        self.root_dir = root_dir
        self.files = files
        self.labels = labels
        self.classes = classes
        self.imsize = imsize
        self.mean = np.array([0.485, 0.456, 0.406]) # imagenet
        self.std = np.array([0.229, 0.224, 0.225]) # imagenet
        self.is_train = is_train
        self.nchannels = nchannels # number of channels of sample


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = [self.classes.index(self.labels[idx])]
        nimage = io.imread(self.files[idx])/255.
        nimage = nimage.transpose((2, 1, 0))

        image = np.zeros((self.nchannels,self.imsize,self.imsize))
        for i in range(self.nchannels):
            image[i,:,:] = resize(nimage[i,:,:], (self.imsize,self.imsize), anti_aliasing=False, mode='reflect')

        if self.is_train:
            image = self.RandomFlip(image)
        
        sample = {'image': torch.from_numpy(image), 'label': torch.LongTensor(label)}
        return sample

    def RandomFlip(self, image):
        """Flip randomly the image in a sample."""
        sr = np.random.randint(0, 2)
        if sr:
            return np.flip(image, axis=2).copy()
        else:
            return image
