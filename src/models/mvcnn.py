import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

sys.path.insert(0, './src/models')

from adaptors import LinearAdaptor, MediumAdaptor
import random

def get_views(num_views, channels):
    lviews = []
    for s in range(num_views):
            random.seed(s)
            sumc = 0
            linear_weights = np.zeros((3, channels))
            for i in range(3):
                sumc = 0
                indices = []
                cont = 1
                for j in range(channels):
                    while True:
                        idx = random.randint(0, channels-1)
                        if idx not in indices:
                            indices.append(idx)
                            break
                    if cont == channels:
                        linear_weights[i,idx] = 1-sumc
                    else:
                        linear_weights[i,idx] = random.uniform(0, 1-sumc)
                        sumc += linear_weights[i,idx]
                    cont += 1
            lviews.append(linear_weights)
    return lviews

class MVCNN_vgg16(nn.Module):
    def __init__(self, nclasses=200, num_views=10):
        super(MVCNN_vgg16, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        self.features = models.vgg16(pretrained=True).features
        self.classifier = models.vgg16(pretrained=True).classifier
        self.classifier._modules['6'] = nn.Linear(512*8, self.nclasses)

    def forward(self, x):
        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1))

class LMVCNN2_vgg16(nn.Module):
    def __init__(self, nclasses=200, num_views=10, channels=5):
        super(LMVCNN2_vgg16, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        torch.manual_seed(0)
        self.adaptor1 = LinearAdaptor(channels)
        torch.manual_seed(1)
        self.adaptor2 = LinearAdaptor(channels)
        self.features = models.vgg16(pretrained=True).features
        self.classifier = models.vgg16(pretrained=True).classifier
        self.classifier._modules['6'] = nn.Linear(512*8, self.nclasses)

    def forward(self, x):
        N,C,H,W = x.size()
        x1 = self.adaptor1(x)
        x2 = self.adaptor2(x)

        x = torch.empty(N*2,3,H,W).cuda()
        cont = 0
        for i in range(N):
            x[cont,:,:,:] = x1[i,:,:,:]
            x[cont+1,:,:,:] = x2[i,:,:,:]
            cont += 2

        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1)), x1, x2

class LMVCNN5_vgg16(nn.Module):
    def __init__(self, nclasses=200, num_views=10, channels=5):
        super(LMVCNN5_vgg16, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        torch.manual_seed(0)
        self.adaptor1 = LinearAdaptor(channels)
        torch.manual_seed(1)
        self.adaptor2 = LinearAdaptor(channels)
        torch.manual_seed(2)
        self.adaptor3 = LinearAdaptor(channels)
        torch.manual_seed(3)
        self.adaptor4 = LinearAdaptor(channels)
        torch.manual_seed(4)
        self.adaptor5 = LinearAdaptor(channels)
        self.features = models.vgg16(pretrained=True).features
        self.classifier = models.vgg16(pretrained=True).classifier
        self.classifier._modules['6'] = nn.Linear(512*8, self.nclasses)

    def forward(self, x):
        N,C,H,W = x.size()
        x1 = self.adaptor1(x)
        x2 = self.adaptor2(x)
        x3 = self.adaptor3(x)
        x4 = self.adaptor4(x)
        x5 = self.adaptor5(x)

        x = torch.empty(N*5,3,H,W).cuda()
        cont = 0
        for i in range(N):
            x[cont,:,:,:] = x1[i,:,:,:]
            x[cont+1,:,:,:] = x2[i,:,:,:]
            x[cont+2,:,:,:] = x3[i,:,:,:]
            x[cont+3,:,:,:] = x4[i,:,:,:]
            x[cont+4,:,:,:] = x5[i,:,:,:]
            cont += 5

        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1)), x1, x2, x3, x4, x5

class MVCNN_resnet18(nn.Module):
    def __init__(self, nclasses=200, num_views=10):
        super(MVCNN_resnet18, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512*1, self.nclasses)

    def forward(self, x):
        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 


class LMVCNN2_resnet18(nn.Module):
    def __init__(self, nclasses=200, num_views=10, channels=5):
        super(LMVCNN2_resnet18, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        torch.manual_seed(0)
        self.adaptor1 = LinearAdaptor(channels)
        torch.manual_seed(1)
        self.adaptor2 = LinearAdaptor(channels)

        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512*1, self.nclasses)

    def forward(self, x):
        N,C,H,W = x.size()
        x1 = self.adaptor1(x)
        x2 = self.adaptor2(x)

        x = torch.empty(N*2,3,H,W).cuda()
        cont = 0
        for i in range(N):
            x[cont,:,:,:] = x1[i,:,:,:]
            x[cont+1,:,:,:] = x2[i,:,:,:]
            cont += 2

        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1)), x1, x2

class LMVCNN5_resnet18(nn.Module):
    def __init__(self, nclasses=200, num_views=10, channels=5):
        super(LMVCNN5_resnet18, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        torch.manual_seed(0)
        self.adaptor1 = LinearAdaptor(channels)
        torch.manual_seed(1)
        self.adaptor2 = LinearAdaptor(channels)
        torch.manual_seed(2)
        self.adaptor3 = LinearAdaptor(channels)
        torch.manual_seed(3)
        self.adaptor4 = LinearAdaptor(channels)
        torch.manual_seed(4)
        self.adaptor5 = LinearAdaptor(channels)

        model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512*1, self.nclasses)

    def forward(self, x):
        N,C,H,W = x.size()
        x1 = self.adaptor1(x)
        x2 = self.adaptor2(x)
        x3 = self.adaptor3(x)
        x4 = self.adaptor4(x)
        x5 = self.adaptor5(x)

        x = torch.empty(N*5,3,H,W).cuda()
        cont = 0
        for i in range(N):
            x[cont,:,:,:] = x1[i,:,:,:]
            x[cont+1,:,:,:] = x2[i,:,:,:]
            x[cont+2,:,:,:] = x3[i,:,:,:]
            x[cont+3,:,:,:] = x4[i,:,:,:]
            x[cont+4,:,:,:] = x5[i,:,:,:]
            cont += 5

        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1)), x1, x2, x3, x4, x5

class MVCNN_resnet50(nn.Module):
    def __init__(self, nclasses=200, num_views=10):
        super(MVCNN_resnet50, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512*4, self.nclasses)

    def forward(self, x):
        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1])) 
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1))

class LMVCNN2_resnet50(nn.Module):
    def __init__(self, nclasses=200, num_views=10, channels=5):
        super(LMVCNN2_resnet50, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        torch.manual_seed(0)
        self.adaptor1 = LinearAdaptor(channels)
        torch.manual_seed(1)
        self.adaptor2 = LinearAdaptor(channels)

        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512*4, self.nclasses)

    def forward(self, x):
        N,C,H,W = x.size()
        x1 = self.adaptor1(x)
        x2 = self.adaptor2(x)

        x = torch.empty(N*2,3,H,W).cuda()
        cont = 0
        for i in range(N):
            x[cont,:,:,:] = x1[i,:,:,:]
            x[cont+1,:,:,:] = x2[i,:,:,:]
            cont += 2

        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1)), x1, x2

class LMVCNN5_resnet50(nn.Module):
    def __init__(self, nclasses=200, num_views=10, channels=5):
        super(LMVCNN5_resnet50, self).__init__()
        self.nclasses = nclasses
        self.num_views = num_views
        torch.manual_seed(0)
        self.adaptor1 = LinearAdaptor(channels)
        torch.manual_seed(1)
        self.adaptor2 = LinearAdaptor(channels)
        torch.manual_seed(2)
        self.adaptor3 = LinearAdaptor(channels)
        torch.manual_seed(3)
        self.adaptor4 = LinearAdaptor(channels)
        torch.manual_seed(4)
        self.adaptor5 = LinearAdaptor(channels)

        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = model.fc
        self.classifier = nn.Linear(512*4, self.nclasses)

    def forward(self, x):
        N,C,H,W = x.size()
        x1 = self.adaptor1(x)
        x2 = self.adaptor2(x)
        x3 = self.adaptor3(x)
        x4 = self.adaptor4(x)
        x5 = self.adaptor5(x)

        x = torch.empty(N*5,3,H,W).cuda()
        cont = 0
        for i in range(N):
            x[cont,:,:,:] = x1[i,:,:,:]
            x[cont+1,:,:,:] = x2[i,:,:,:]
            x[cont+2,:,:,:] = x3[i,:,:,:]
            x[cont+3,:,:,:] = x4[i,:,:,:]
            x[cont+4,:,:,:] = x5[i,:,:,:]
            cont += 5

        y = self.features(x)
        y = y.view((int(x.shape[0]/self.num_views),self.num_views,y.shape[-3],y.shape[-2],y.shape[-1]))
        return self.classifier(torch.max(y,1)[0].view(y.shape[0],-1)), x1, x2, x3, x4, x5

