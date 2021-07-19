import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearAdaptor(nn.Module):
    def __init__(self, k=3, is_pretrain=False):
        super(LinearAdaptor, self).__init__()
        self.is_pretrain = is_pretrain
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda() # imagenet
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda() # imagenet
        self.conv1 = nn.Conv2d(k, 3, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        if not self.is_pretrain:
            out = (out - self.mean.reshape(1,-1,1,1))/self.std.reshape(1,-1,1,1)
        return out

class MediumAdaptor(nn.Module):
    def __init__(self, k=3, is_pretrain=False):
        super(MediumAdaptor, self).__init__()
        self.is_pretrain = is_pretrain
        self.mean = np.array([0.485, 0.456, 0.406]) # imagenet
        self.std = np.array([0.229, 0.224, 0.225]) # imagenet
        self.conv1 = nn.Conv2d(k, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn7   = nn.BatchNorm2d(16)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.bn8   = nn.BatchNorm2d(3)
        self.relu8 = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu7(self.bn7(self.conv7(out)))
        out = self.relu8(self.bn8(self.conv8(out)))
        if not self.is_pretrain:
            out = (out - torch.from_numpy(self.mean.reshape(-1,1,1)).float().cuda())/torch.from_numpy(self.std.reshape(-1,1,1)).float().cuda()
        return out

class MediumDecoder(nn.Module):
    def __init__(self, k=3):
        super(MediumDecoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn7   = nn.BatchNorm2d(16)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(16, k, kernel_size=3, stride=1, padding=1)
        self.bn8   = nn.BatchNorm2d(k)
        self.relu8 = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu7(self.bn7(self.conv7(out)))
        out = self.relu8(self.bn8(self.conv8(out)))
        return out
