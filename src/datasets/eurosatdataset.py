import os
import sys
import torch
import random
import numpy as np
from skimage import io
from random import seed, sample
from skimage.transform import resize
from torch.utils.data import Dataset

from config import centers, means, stds

class EuroSATDataset(Dataset):
    """EuroSAT dataset."""

    def __init__(self, root_dir, is_train=True, numdata=5994, nchannels=3, is_pretraining=False, mean_needed=True, num_views=0, is_linear=False, array=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Train set if True, Validation set otherwise.
            numdata (int): Number of samples used to train.
            nchannels (int): 3 for RGB, 5 and 15 for multi-channel arrays
        """
        numdata = max(100, numdata)
        
        root_dir0 = os.path.join(root_dir, 'RGB')
        root_dir = os.path.join(root_dir, 'images', 'remote_sensing', 'otherDatasets', 'sentinel_2', 'tif')
        classes_dir = sorted(os.listdir(root_dir))

        # get files and labels
        img_files = []
        img_files_rgb = []
        labelsi = []
        classes = []
        for class_i in classes_dir:
            if class_i not in classes: classes.append(class_i)
            imgfiles = sorted(os.listdir(os.path.join(root_dir,class_i)))
            for filename in imgfiles:
                img_files.append(os.path.join(root_dir,class_i,filename))
                img_files_rgb.append(os.path.join(root_dir0,class_i,filename[:-4]+'.jpg'))
                labelsi.append(class_i)
                
        classes = list(set(labelsi))

        # there is no split so randomize everything and train/test split
        seed(0)
        rndidx = sample(range(len(img_files)), len(img_files))
        files = []
        files_rgb = []
        labels = []
        for i in rndidx:
            files.append(img_files[i])
            files_rgb.append(img_files_rgb[i])
            labels.append(labelsi[i])
        
        if is_train:
            files = files[0:int(numdata)]
            files_rgb = files_rgb[0:int(numdata)]
            labels = labels[0:int(numdata)]
        else:
            files = files[10000:]
            files_rgb = files_rgb[10000:]
            labels = labels[10000:]

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
        self.meank[13] = means['eurosat']
        self.stdk[13] = stds['eurosat']
        self.root_dir = root_dir
        self.files = files
        self.files_rgb = files_rgb
        self.labels = labels
        self.classes = classes
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

        label = [self.classes.index(self.labels[idx])]
        nimage = io.imread(self.files[idx])/255.
        nimage0 = io.imread(self.files_rgb[idx])/255.
        nimage = nimage.transpose((2, 1, 0))

        image = np.zeros((13,224,224))
        for i in range(13):
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

    def RandomFlip(self, image):
        """Flip randomly the image in a sample."""
        sr = np.random.randint(0, 2)
        if sr:
            return np.flip(image, axis=2).copy()
        else:
            return image


