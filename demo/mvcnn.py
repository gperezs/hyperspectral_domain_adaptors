import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class LinearAdaptor(nn.Module):
    def __init__(self, in_channels=5):
        super(LinearAdaptor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        return out

class LMVCNN(nn.Module):
    '''Multi-View CNN with Linear Adaptors'''
    def __init__(self, model='vgg16', nclasses=200, num_views=5, in_channels=5):
        super(LMVCNN, self).__init__()
        self.num_views = num_views
    
        # initialize one adaptor per view
        self.adaptor = []
        for i in range(num_views):
            self.adaptor.append(LinearAdaptor(in_channels).cuda())
       
        # divide model where the aggregation is done and modify las layer to 
        # personalized number of classes.
        if model == 'vgg16':
            self.features = models.vgg16(pretrained=True).features
            self.classifier = models.vgg16(pretrained=True).classifier
            self.classifier._modules['6'] = nn.Linear(4096, nclasses)
        elif model == 'resnet18':
            model = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = model.fc
            self.classifier = nn.Linear(512, self.nclasses)
        elif model == 'resnet50':
            model = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = model.fc
            self.classifier = nn.Linear(2048, self.nclasses)

    def forward(self, x):
        '''
        Shapes on each step when using VGG16 and BS 64 (before scaling rule:
        - input: [12, 13, 224, 224])
        - after adaptors and rearranging: [60, 3, 224, 224]
        - after features(x): [60, 512, 7, 7]
        - after rearranging y.view(): [12, 5, 512, 7, 7]
        - after aggregation (max): [12, 512, 7, 7]
        - after rearranging before classifier(): [12, 25088]
        '''
        N,C,H,W = x.size() 
        x_v = []
        for i in range(self.num_views):
            x_v.append(self.adaptor[i](x))

        # rearrange samples
        x = torch.empty(N*self.num_views,3,H,W).cuda() 
        cont = 0
        for i in range(N):
            for j in range(self.num_views):
                x[cont,:,:,:] = x_v[j][i,:,:,:]
                cont += 1

        y = self.features(x) 
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 
        out = self.classifier(torch.max(y,1)[0].view(y.shape[0],-1))
        return out, x_v
