#! /usr/bin/env python

import os
import sys
import time
import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torch.autograd import Variable
import torchvision.models as models

from mvcnn import LMVCNN

sys.path.insert(0, '..')
from config import dset_root
from eurosat import EuroSATDataset 

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for CUB')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lre', type=int, default=4, help='use forlearning rate schedule')
    parser.add_argument('--backbone', type=str, default='vgg16', help='pytorch model zoo')
    parser.add_argument('--num_views', type=int, default=5, help='number of views for multi-view cnn')
    parser.add_argument('--channels', type=int, default=13, help='number of channels of the input arrays')
    parser.add_argument('--classes', type=int, default=10, help='number of classes of dataset')
    parser.add_argument('--gpu', type=str, default='', help='CUDA visible device')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--log-interval', type=int, default=30, help='#batches before logging training status')
    parser.add_argument('--save_dir', type=str, default='.', help='save dir for log and checkpoints')
    args = parser.parse_args()
    return args

def get_diversity_reg(input, p=2):
    a, b, c, d = input.size() # NxCxHxW
    diversity_reg = 0
    for i in range(a):
        diversity_reg += torch.linalg.norm(gram_matrix(input[i,:,:,:]), ord=p)
    return diversity_reg

def gram_matrix(input): # from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
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
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        if not (target.detach().cpu().numpy().shape): continue
        optimizer.zero_grad()
        output, adaptors_out = model(data)
        diversity_reg = 0
        if args.num_views: # to add diversity reg
            adaptors = torch.cat(adaptors_out, 1)
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
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if not (target.detach().cpu().numpy().shape): continue
            output, _ = model(data)
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


if __name__ == '__main__':

    args = parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    log_name = 'demo.log'
    os.makedirs(args.save_dir, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # we need to reduce the batch size proportional to the number of views
    # to fit it into the GPU (EuroSAT uses ~10GB GPU RAM with a BS=64) so we 
    # compensate using the linear scaling rule (Goyal et al., 2018):
    if args.num_views:
        args.lr /= args.num_views 
        args.lre *= args.num_views
        args.epochs *= args.num_views
        args.batch_size = int(args.batch_size/args.num_views)
        args.log_interval = int(args.log_interval*args.num_views)

    print('Called with args:')
    print(args)
    with open(os.path.join(args.save_dir,log_name), "w") as f:
        f.write('Called with args:\n')
        f.write(str(args)+'\n')

    # load dataset
    train_dset = EuroSATDataset(dset_root['eurosat'],
                                nchannels=args.channels,
                                num_views=args.num_views,
                                is_train=True)
    train_loader = torch_du.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    test_dset  = EuroSATDataset(dset_root['eurosat'],
                                nchannels=args.channels,
                                num_views=args.num_views)
    test_loader = torch_du.DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    # load model
    model = LMVCNN(model=args.backbone,
                   nclasses=args.classes,
                   num_views=args.num_views,
                   in_channels=args.channels)
    print(model.adaptor)
    print(model)
     
    if args.cuda:
        model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: %e\n'%(params))

    # increase LR of adaptors and last layer for training
    # and define optimizer
    if args.backbone == 'vgg16':
        params = []
        for i in range(args.num_views):
            params.append({'params': model.adaptor[i].parameters(), 'lr': args.lr*100},)
        params.append({'params': model.features.parameters()})
        params.append({'params': model.classifier[0:6].parameters()})
        params.append({'params': model.classifier[6].parameters(), 'lr': args.lr*10})
        optimizer = optim.Adam(params, lr=args.lr)
    elif 'resnet' in args.backbone:
        params = []
        for i in range(args.num_views):
            params.append({'params': model.adaptor[i].parameters(), 'lr': args.lr*100},)
        params.append({'params': model.features.parameters()})
        params.append({'params': model.classifier.parameters(), 'lr': args.lr*10})
        optimizer = optim.Adam(params, lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    try:
        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_accuracy, train_loss = train(epoch, train_loader, log_name)
            
            # LR schedule:
            if epoch%lre == 0 and args.lr > 1e-6:
                args.lr *= 0.1
                if args.backbone == 'vgg16':
                    params = []
                    for i in range(args.num_views):
                        params.append({'params': model.adaptor[i].parameters(), 'lr': args.lr*100},)
                    params.append({'params': model.features.parameters()})
                    params.append({'params': model.classifier[0:6].parameters()})
                    params.append({'params': model.classifier[6].parameters(), 'lr': args.lr*10})
                    optimizer = optim.Adam(params, lr=args.lr)
                elif 'resnet' in args.backbone:
                    params = []
                    for i in range(args.num_views):
                        params.append({'params': model.adaptor[i].parameters(), 'lr': args.lr*100},)
                    params.append({'params': model.features.parameters()})
                    params.append({'params': model.classifier.parameters(), 'lr': args.lr*10})
                    optimizer = optim.Adam(params, lr=args.lr)
            
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

