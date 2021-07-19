import os
import sys
import torch
import random
import numpy as np
from skimage import io
from random import seed, sample
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms

from config import centers, means, stds

class CUBDataset(Dataset):
    """CUB dataset."""

    def __init__(self, root_dir, is_train=True, numdata=5994, nchannels=3, is_pretraining=False, mean_needed=True, num_views=0, is_linear=False, array=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            is_train (bool): Train set if True, Validation set otherwise.
            numdata (int): Number of samples used to train.
            nchannels (int): 3 for RGB, 5 and 15 for multi-channel arrays
        """
        numdata = max(200, numdata)
        
        # get ids
        text_file = open(os.path.join(root_dir, 'CUB_200_2011', 'images.txt'), "r")
        text_ids = text_file.readlines() # file id used for tran/test split
        ids = []
        for text_id in text_ids:
            ids.append(text_id.split()[1])

        # get labels
        text_file = open(os.path.join(root_dir, 'CUB_200_2011', 'image_class_labels.txt'), "r")
        text_labels = text_file.readlines() # image labels (1 to 200)
        labels = []
        for text_label in text_labels:
            labels.append(int(text_label.split()[1])-1)

        # get train/test split (1 for train, 0 for test)
        text_file = open(os.path.join(root_dir, 'CUB_200_2011', 'train_test_split.txt'), "r")
        text_split = text_file.readlines() # train/test split
        train_test_split = []
        for split in text_split:
            train_test_split.append(int(split.split()[1]))

        # get file names
        folders = sorted(os.listdir(os.path.join(root_dir, 'CUB_200_2011', 'images')))
        files_train = []
        files_test = []
        for folder in folders:
            filenames = sorted(os.listdir(os.path.join(root_dir, 'CUB_200_2011', 'images', folder)))
            for filename in filenames:
                file_name = os.path.join(folder, filename)
                if train_test_split[ids.index(file_name)]: 
                    files_train.append(file_name)
                else:
                    files_test.append(file_name)

        if is_train:
            # randomizing indices for when numdata < max (useful when calculating performance with
            # less number of training samples)
            seed(0)
            rndidx = sample(range(len(files_train)), len(files_train))
            files = []
            for i in rndidx:
                files.append(files_train[i])
        else:
            files = files_test
            

        if is_train:
            lbls = []
            for i in range(len(files)):
                lbls.append(labels[ids.index(files[i])])
            ftemp = [] 
            ltemp = [] 
            while lbls:
                idxs = []
                for i in range(len(set(lbls))):
                    if len(set(lbls)) == 200:
                        if i in set(lbls):
                            ltemp.append(lbls[lbls.index(i)])
                            ftemp.append(files[lbls.index(i)])
                            idxs.append(lbls.index(i))
                        else: continue
                    else:
                        ltemp += lbls
                        ftemp += files
                        lbls = []
                        break
                for idx in idxs: 
                    files[idx], lbls[idx] = 'rm', 'rm'
                files = list(filter(('rm').__ne__, files))
                lbls = list(filter(('rm').__ne__, lbls))
            files = ftemp[0:int(numdata)]

        # For multi-view with subsets:
        views = []
        for i in range(num_views):
            random.seed(i+array)
            views.append(random.sample(range(nchannels), 3))
        print('views: %s'%(str(views)))
       
        self.is_linear = is_linear 
        self.views = views
        self.meank = {}
        self.stdk = {}
        self.meank[5] = means['cub5']
        self.stdk[5] = stds['cub5']
        self.meank[15] = means['cub15']
        self.stdk[15] = stds['cub15']
        self.root_dir = root_dir
        self.files = files
        self.ids = ids
        self.labels = labels
        self.mean = np.array([0.485, 0.456, 0.406]) # imagenet
        self.std = np.array([0.229, 0.224, 0.225]) # imagenet
        self.is_train = is_train
        self.centers = {}
        self.centers[5] = centers['cub5'] # kmeans centers for multi-channel arrays
        self.centers[15] = centers['cub15'] # kmeans centers for multi-channel arrays 	
        self.nchannels = nchannels # number of channels of sample
        self.is_pretraining = is_pretraining # True when pre-training adaptors

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'CUB_200_2011', 'images', self.files[idx])
        
        label = [self.labels[self.ids.index(self.files[idx])]]
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

    
    def rgb2gray(self, rgb):
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

