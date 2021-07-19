#! /usr/bin/env python

import os
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './src/models')
sys.path.insert(0, './src/datasets')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torch.autograd import Variable
import torchvision.models as models

from adaptors import LinearAdaptor, MediumAdaptor
from mvcnn import (MVCNN_vgg16, LMVCNN2_vgg16, LMVCNN5_vgg16,
                   MVCNN_resnet18, LMVCNN2_resnet18, LMVCNN5_resnet18,
                   MVCNN_resnet50, LMVCNN2_resnet50, LMVCNN5_resnet50)

from config import dset_root


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for CUB')
    parser.add_argument('--array', type=int, default=0,
                        help='array id for sbatch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--model',  help='Adaptor model: linear, snall, medium, none(no adaptor from scratch)',
                        default='none', type=str)
    parser.add_argument('--backbone',  help='pretrained backbone architecture (pytorch model zoo)',
                        default='vgg16', type=str)
    parser.add_argument('--dataset',  help='Datasets: cub, cars, aircrafts, legus, sunrgbd',
                        default='cub', type=str)
    parser.add_argument('--pretrained_adaptor', type=int, default=0,
                        help='use pre-trained adaptor')
    parser.add_argument('--inflate', type=int, default=0,
                        help='use inflated filters in the first layer')
    parser.add_argument('--num_views', type=int, default=0,
                        help='number of views for multi-view cnn (0 for no mvcnn)')
    parser.add_argument('--numdata', type=int, default=5994,
                        help='number of training samples (default is equal to all train samples)')
    parser.add_argument('--channels', type=int, default=5,
                        help='number of channels of the input arrays')
    parser.add_argument('--gpu', dest='gpu', help='CUDA visible device',
                        default='', type=str)
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir', dest='save_dir', help='save dir for checkpoints',
                        default='checkpoints/V001/', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network', default='', type=str)
    args = parser.parse_args()
    return args


def get_diversity_reg(input, p=2):
    a, b, c, d = input.size() # NxCxHxW
    diversity_reg = 0
    for i in range(a):
        diversity_reg += torch.linalg.norm(gram_matrix(input[i,:,:,:]), ord=p)
    return diversity_reg

def gram_matrix(input): # GPS: from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    b, c, d = input.size()
    features = input.view(b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(b * c * d)


def train(epoch, train_loader, log_name):
    model.train()
    train_loss = 0
    correct = 0
    num_data = 0

    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['image'].float()
        target = np.squeeze(sample_batched['label'].long())
        num_data += len(data)
        if args.num_views and args.model == 'none':
            data = data.reshape(-1,3,224,224)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if not (target.detach().cpu().numpy().shape): continue
        optimizer.zero_grad()
        if args.num_views == 2:
            output, adaptor1, adaptor2 = model(data)
        elif args.num_views == 5:
            output, adaptor1, adaptor2, adaptor3, adaptor4, adaptor5 = model(data)
        else: 
            output = model(data)
        diversity_reg = 0
        if args.num_views: # to add diversity reg
            adaptors = torch.cat((adaptor1, adaptor2, adaptor3, adaptor4, adaptor5), 1)
            diversity_reg = 1e-2*get_diversity_reg(adaptors, p=2) 
        loss = criterion(output, target) + diversity_reg
        train_loss += loss.data
        pred = output.max(1)[1]
        correct += pred.eq(target).cpu().sum().numpy()
        accuracy_batch = 100. * correct / num_data 
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print_line = ('Train Epoch: {:2d} [{:4d}/{} ({:3.0f}%)] Loss: {:.4f} ({:.3f}) Acc: {:.3f}% lr: {:.0e} '.format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader), loss.data, train_loss/(batch_idx+1.0), 
                accuracy_batch,optimizer.param_groups[-1]['lr']))
            with open(os.path.join(args.save_dir,log_name), "a") as f:
                f.write(print_line+"\n")
            print(print_line)
    return accuracy_batch, train_loss/(batch_idx+1.0)


def test(epoch, test_loader, log_name):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = np.array([], dtype=np.int64).reshape(0)
    targets = np.array([], dtype=np.int64).reshape(0)
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            data = sample_batched['image'].float()
            target = np.squeeze(sample_batched['label'].long())
            if args.num_views and args.model == 'none':
                data = data.reshape(-1,3,224,224)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if not (target.detach().cpu().numpy().shape): continue
            if args.num_views == 2:
                output, _, _ = model(data)
            elif args.num_views == 5:
                output, _, _, _, _, _ = model(data)
            else:
                output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1)[1]
            correct += pred.eq(target).cpu().sum().numpy()
            predictions = np.concatenate((predictions, pred.cpu().numpy()))
            targets = np.concatenate((targets, target.data.cpu().numpy()))
            if batch_idx % args.log_interval == 0:
                print_line = ('Test epoch: {} [{}/{} ({:3.0f}%)] Loss: {:2.4f} ({:2.3f}) lr: {:.0e} '.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss.data, test_loss/(batch_idx+1.0), 
                    optimizer.param_groups[-1]['lr']))
                with open(os.path.join(args.save_dir,log_name), "a") as f:
                    f.write(print_line+"\n")
    test_loss /= len(test_loader)
    print_line = ('Test set: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.2f}%)\n '.format(test_loss, correct, len(test_loader.dataset), 
              correct/len(test_loader.dataset)*100.))
    with open(os.path.join(args.save_dir,log_name), "a") as f:
        f.write(print_line+"\n")
    print(print_line)
    return targets, predictions, correct/len(test_loader.dataset)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()


if __name__ == '__main__':

    args = parse_args()
    if args.model == 'none':
        args.pretrained_adaptor = 0
    log_name = 'log_%s_adaptor_%s-%d_%d_pa%d_mv%d_%d.txt'%(args.model, args.dataset, args.channels, 
                                                        args.numdata, args.pretrained_adaptor, 
                                                        args.num_views, args.array)
    if args.inflate:
        args.save_dir = os.path.join('checkpoints', args.backbone, 'inflated', '%s-%d'%(args.dataset, args.channels))
    elif args.num_views:
        args.save_dir = os.path.join('checkpoints', args.backbone, 'mvcnn', '%s-%d'%(args.dataset, args.channels))
    else:
        args.save_dir = os.path.join('checkpoints', args.backbone, '%s-%d'%(args.dataset, args.channels))

    if args.dataset == 'cub':
        classes, lre = 200, 7 # lre: lr decrease epoch (default lre: 7, mv: 12)
        from cubdataset import CUBDataset as KDataset
    elif args.dataset == 'cars':
        classes, lre = 196, 15
        from carsdataset import CarsDataset as KDataset
    elif args.dataset == 'aircrafts':
        classes, lre = 100, 10
        from aircraftsdataset import AircraftsDataset as KDataset
    elif args.dataset == 'legus':
        classes, lre = 4, 5 # default lre: 5, mv: 15
        from legusdataset import LEGUSDataset as KDataset
    elif args.dataset == 'eurosat':
        classes, lre = 10, 4 # tune hyperparams
        from eurosatdataset import EuroSATDataset as KDataset
    elif args.dataset == 'so2sat':
        classes, lre = 18, 4 # tune hyperparams
        from so2sadataset import So2SaDataset as KDataset

    # to train from scratch
    if args.model == 'none' and not args.num_views and not args.inflate and args.channels != 3 :
        lre *= 2
        args.epochs *= 17
    
    if args.num_views:	
        args.lr /= args.num_views # linear scaling rule (Goyal et al., 2018)
        if args.backbone != 'googlenet':
            lre *= args.num_views
            args.epochs *= args.num_views

    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('Called with args:')
    print(args)
    with open(os.path.join(args.save_dir,log_name), "w") as f:
        f.write('Called with args:\n')
        f.write(str(args)+'\n')
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
   
    if args.num_views:
        args.batch_size = int(args.batch_size/args.num_views)
        args.log_interval = int(args.log_interval*args.num_views)
    
    # load dataset
    train_dset = KDataset(dset_root[args.dataset],
                            is_train=True,
                            numdata=args.numdata,
                            nchannels=args.channels,
                            is_pretraining=False,
                            num_views=args.num_views,
                            is_linear=(args.model == 'linear' or args.model == 'medium'),
                            array=args.array)
    train_loader = torch_du.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    test_dset  = KDataset(dset_root[args.dataset],
                            is_train=False,
                            nchannels=args.channels,
                            is_pretraining=False,
                            num_views=args.num_views,
                            is_linear=(args.model == 'linear' or args.model == 'medium'),
                            array=args.array)
    test_loader = torch_du.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    # load model and adaptor
    if args.model == 'none':
        if args.channels == 3:
            if args.backbone == 'vgg16':
                model = models.vgg16(pretrained=True)
                model.classifier[6] = nn.Linear(512*8, classes) # change last layer
            elif args.backbone == 'resnet18':
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(512*1, classes) # change last layer
            elif args.backbone == 'resnet50':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(512*4*1, classes) # change last layer
        elif args.num_views:
            if args.backbone == 'vgg16':
                model = MVCNN_vgg16(classes, args.num_views)
            elif args.backbone == 'resnet18':
                model = MVCNN_resnet18(classes, args.num_views)
            elif args.backbone == 'resnet50':
                model = MVCNN_resnet50(classes, args.num_views)
        else:
            if args.inflate:
                if args.backbone == 'vgg16':
                    model = models.vgg16(pretrained=True)
                    convmean = model.features[0].weight.mean(axis=1) # mean of weights ([64, 3, 3])
                    prevparams = model.features[0].weight
                    model.features[0] = nn.Conv2d(args.channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    with torch.no_grad(): # so the layer stays as leaf
                        for i in range(args.channels):
                            model.features[0].weight[:,i,:,:] = torch.nn.Parameter(convmean)
                    model.features[0].weight.requires_grad = True
                    model.classifier[6] = nn.Linear(512*8, classes) # change last layer
                elif args.backbone == 'resnet18':
                    model = models.resnet18(pretrained=True)
                    convmean = model.conv1.weight.mean(axis=1) # mean of weights ([64, 3, 3])
                    prevparams = model.conv1.weight
                    model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
                    with torch.no_grad(): # so the layer stays as leaf
                        for i in range(args.channels):
                            model.conv1.weight[:,i,:,:] = torch.nn.Parameter(convmean)
                    model.conv1.weight.requires_grad = True
                    model.fc = nn.Linear(512*1, classes) # change last layer
                elif args.backbone == 'resnet50':
                    model = models.resnet50(pretrained=True)
                    convmean = model.conv1.weight.mean(axis=1) # mean of weights ([64, 3, 3])
                    prevparams = model.conv1.weight
                    model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
                    with torch.no_grad(): # so the layer stays as leaf
                        for i in range(args.channels):
                            model.conv1.weight[:,i,:,:] = torch.nn.Parameter(convmean)
                    model.conv1.weight.requires_grad = True
                    model.fc = nn.Linear(512*4, classes) # change last layer
            else:
                if args.backbone == 'vgg16':
                    model = models.vgg16()
                    model.features[0] = nn.Conv2d(args.channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    model.classifier[6] = nn.Linear(512*8, classes) # change last layer
                    model.apply(weights_init) # weights initialization
                elif args.backbone == 'resnet18':
                    model = models.resnet18()
                    model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
                    model.fc = nn.Linear(512*1, classes) # change last layer
                elif args.backbone == 'resnet50':
                    model = models.resnet50()
                    model.conv1 = nn.Conv2d(args.channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
                    model.fc = nn.Linear(512*4, classes) # change last layer
    else:
        if args.num_views == 2:
            if args.model == 'linear':
                if args.backbone == 'vgg16':
                    model = LMVCNN2_vgg16(classes, args.num_views, args.channels)
                elif args.backbone == 'resnet18':
                    model = LMVCNN2_resnet18(classes, args.num_views, args.channels)
                elif args.backbone == 'resnet50':
                    model = LMVCNN2_resnet50(classes, args.num_views, args.channels)
        elif args.num_views == 5:
            if args.model == 'linear':
                if args.backbone == 'vgg16':
                    model = LMVCNN5_vgg16(classes, args.num_views, args.channels)
                elif args.backbone == 'resnet18':
                    model = LMVCNN5_resnet18(classes, args.num_views, args.channels)
                elif args.backbone == 'resnet50':
                    model = LMVCNN5_resnet50(classes, args.num_views, args.channels)
        else:
            if args.backbone == 'vgg16':
                model = models.vgg16(pretrained=True)
                model.classifier[6] = nn.Linear(512*8, classes) # change last layer
            elif args.backbone == 'resnet18':
                model = models.resnet18(pretrained=True)
                model.fc = nn.Linear(512*1, classes) # change last layer
            elif args.backbone == 'resnet50':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(512*4, classes) # change last layer
            if args.model == 'linear':
                adaptor = LinearAdaptor(args.channels)
            elif args.model == 'medium':
                adaptor = MediumAdaptor(args.channels)

            if args.pretrained_adaptor:
                model_dict_r = adaptor.state_dict()
                pretrained_dict_r = torch.load(os.path.join('checkpoints', 'adaptors', 
                    '%s_adaptor_%s-%d'%(args.model,args.dataset,args.channels), 
                    '%s_adaptor_%s-%d.pth'%(args.model,args.dataset,args.channels)))
                pretrained_dict_r = {k: v for k, v in pretrained_dict_r.items() if k in model_dict_r and v.size() == model_dict_r[k].size()}
                model_dict_r.update(pretrained_dict_r)
                adaptor.load_state_dict(model_dict_r)
            model = nn.Sequential(adaptor, model) # adaptor + VGG-D

    print(model)
    if args.cuda:
        model.cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: %d\n'%(params))

    if args.channels == 3 or args.inflate:
        if args.backbone == 'vgg16':
            optimizer = optim.Adam([{'params': model.features.parameters()},
                                {'params': model.classifier[0:6].parameters()},
                                {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                               ], lr=args.lr)
        elif 'resnet' in args.backbone:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        if args.model == 'none':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.model != 'none' and args.num_views: # GPS: LINEAR MULTI VIEW
            lra = 100
            if args.num_views == 2:
                if args.backbone == 'vgg16':
                    optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier[0:6].parameters()},
                                            {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                elif 'resnet' in args.backbone:
                    optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier.parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
            if args.num_views == 5:
                if args.backbone == 'vgg16':
                    optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor3.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor4.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor5.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier[0:6].parameters()},
                                            {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                elif 'resnet' in args.backbone:
                    optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor3.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor4.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor5.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier.parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
        elif args.model != 'none' and not args.num_views:
            lra = 10
            if args.backbone == 'vgg16':
                optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*lra},
                                    {'params': model[1].features.parameters()},
                                    {'params': model[1].classifier[0:6].parameters()},
                                    {'params': model[1].classifier[6].parameters(), 'lr': args.lr*10}
                                   ], lr=args.lr)
            elif 'resnet' in args.backbone:
                optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*lra},
                                    {'params': model[1].parameters()}
                                   ], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
 
    try:
        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_accuracy, train_loss = train(epoch, train_loader, log_name)
            if epoch%lre == 0 and args.lr > 1e-6:: 
                args.lr *= 0.1
                if args.channels == 3 or args.inflate:
                    if args.backbone == 'vgg16':
                        optimizer = optim.Adam([{'params': model.features.parameters()},
                                            {'params': model.classifier[0:6].parameters()},
                                            {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                    elif 'resnet' in args.backbone:
                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                else:
                    if args.model == 'none':
                        optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    elif args.model != 'none' and args.num_views: # GPS: LINEAR MULTI VIEW
                        if args.num_views == 2:
                            if args.backbone == 'vgg16':
                                optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier[0:6].parameters()},
                                            {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                            elif 'resnet' in args.backbone:
                                optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier.parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                        if args.num_views == 5:
                            if args.backbone == 'vgg16':
                                optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor3.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor4.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor5.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier[0:6].parameters()},
                                            {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                            elif 'resnet' in args.backbone:
                                optimizer = optim.Adam([{'params': model.adaptor1.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor2.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor3.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor4.parameters(), 'lr': args.lr*lra},
                                            {'params': model.adaptor5.parameters(), 'lr': args.lr*lra},
                                            {'params': model.features.parameters()},
                                            {'params': model.classifier.parameters(), 'lr': args.lr*10}
                                           ], lr=args.lr)
                    elif args.model != 'none' and not args.num_views:
                        if args.backbone == 'vgg16':
                            optimizer = optim.Adam([{'params': model[0].parameters() ,'lr': args.lr*lra},
                                                {'params': model[1].features.parameters()},
                                                {'params': model[1].classifier[0:6].parameters()},
                                                {'params': model[1].classifier[6].parameters(), 'lr': args.lr*10}
                                               ], lr=args.lr)
                        elif 'resnet' in args.backbone:
                            optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*lra},
                                                {'params': model[1].parameters()}
                                               ], lr=args.lr)
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))
            print('-' * 80)
        print('\nTesting...')
        labels, predictions, test_accuracy = test(epoch, test_loader, log_name)
        with open(os.path.join(args.save_dir,log_name), "a") as f:
            f.write(('-'*80)+'\n')
            f.write('Test accuracy: %.4f \n'%(test_accuracy))
            f.write(('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
            epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))+'\n')
            f.write(('-'*80)+'\n')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


