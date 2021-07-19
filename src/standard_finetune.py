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

from config import dset_root

from adaptors import LinearAdaptor2, MediumAdaptor2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a model for CUB')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--model',  help='Model, as named in PyTorch (torchvision.models)',
                        default='vgg16', type=str)
    parser.add_argument('--dataset',  help='Datasets: cub, cars, aircrafts',
                        default='cub', type=str)
    parser.add_argument('--numdata', type=int, default=5994,
                        help='number of training samples (default is equal to all train samples)')
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


def train(epoch, train_loader, log_name):
    model.train()
    train_loss = 0
    correct = 0
    num_data = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['image'].float()
        target = np.squeeze(sample_batched['label'].long())
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        num_data += len(data)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) 
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
                accuracy_batch,optimizer.param_groups[0]['lr']))
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
    print('Test set: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.2f}%)\n '.format(test_loss, correct, len(test_loader.dataset), 
              correct/len(test_loader.dataset)*100.))
    return targets, predictions, correct/len(test_loader.dataset)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform_(m.weight)
        #nn.init.xavier_normal_(m.weight)
        #nn.init.kaiming_uniform_(m.weight)
        #nn.init.kaiming_normal_(m.weight)
        #m.weight.data.normal_(0.00, 0.01)
        nn.init.dirac_(m.weight)
        m.bias.data.fill_(0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

if __name__ == '__main__':

    args = parse_args()
    log_name = 'log_%d_%s.txt'%(args.numdata, args.model)
    args.save_dir = os.path.join('checkpoints', 'standard_finetune_%s'%(args.dataset))

    if args.dataset == 'cub':
        classes, lre = 200, 10 # lre: lr decrease epoch (default: 7)
        from cubdataset import CUBDataset as KDataset
    elif args.dataset == 'cars':
        classes, lre = 196, 15
        from carsdataset import CarsDataset as KDataset
    elif args.dataset == 'aircrafts':
        classes, lre = 100, 10
        from aircraftsdataset import AircraftsDataset as KDataset
    elif args.dataset == 'flowers':
        classes, lre = 102, 10 # tune hyperparams
        from flowersdataset import FlowersDataset as KDataset

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
   
    # load dataset
    train_dset = KDataset(dset_root[args.dataset], is_train=True, numdata=args.numdata, nchannels=3)
    train_loader = torch_du.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    test_dset = KDataset(dset_root[args.dataset], is_train=False, nchannels=3)
    test_loader = torch_du.DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=False)

    # load model (VGG-D)
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(512*8, classes) # change last layer for 200 classes of CUB
    elif args.model == 'resnet18': 
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512*1, classes) # change last layer
    elif args.model == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(512*4, classes) # change last layer
    elif args.backbone == 'googlenet':
        model = models.googlenet(pretrained=True)
        model.fc = nn.Linear(512*2, classes) # change last layer
    
    # GPS: True to fix weights of backbone
    if False:
         for name, param in model.named_parameters():
             param.requires_grad = False
         if args.model == 'vgg16':
            model.classifier[6].weight.requires_grad = True
            model.classifier[6].bias.requires_grad = True
         else:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True

    # GPS: this is for the permutation experiments
    linearAdaptor = False
    if linearAdaptor:
        adaptor = LinearAdaptor2()
        adaptor.apply(weights_init)
        #for param in adaptor.parameters():
        #    param.requires_grad = False
        model = nn.Sequential(adaptor, model)

    MediumAdaptor = False
    if MediumAdaptor:
        adaptor = MediumAdaptor2()
        adaptor.apply(weights_init)
        #for param in adaptor.parameters():
        #    param.requires_grad = False
        model = nn.Sequential(adaptor, model)

    print(model)
    if args.cuda:
        model.cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: %d\n'%(params))

    if args.model == 'vgg16' and (linearAdaptor or MediumAdaptor):
        optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*10},
                            {'params': model[1].features.parameters()},
                            {'params': model[1].classifier[0:6].parameters()},
                            {'params': model[1].classifier[6].parameters(), 'lr': args.lr*10}
                           ], lr=args.lr)
    elif args.model == 'vgg16' and not (linearAdaptor or MediumAdaptor):
        optimizer = optim.Adam([{'params': model.features.parameters()},
                            {'params': model.classifier[0:6].parameters()},
                            {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                            ], lr=args.lr)
    elif 'resnet' in args.model and linearAdaptor:
        optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*10},
                                    {'params': model[1].parameters()}
                                   ], lr=args.lr)
    elif args.model == 'googlenet' or 'resnet' in args.model:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
  
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_accuracy, train_loss = train(epoch, train_loader, log_name)
            labels, predictions, test_accuracy = test(epoch, test_loader, log_name)
            if epoch == 10 and False: # GPS: to start with fixed weights ans linear adaptor 
                for name, param in model.named_parameters():
                    param.requires_grad = True
            if epoch%lre == 0 and args.lr >= 1e-6:
                args.lr *= 0.1
                if args.model == 'vgg16' and (linearAdaptor or MediumAdaptor):
                    optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*10},
                                        {'params': model[1].features.parameters()},
                                        {'params': model[1].classifier[0:6].parameters()},
                                        {'params': model[1].classifier[6].parameters(), 'lr': args.lr*10}
                                        ], lr=args.lr)
                elif args.model == 'vgg16' and not (linearAdaptor or MediumAdaptor):
                    optimizer = optim.Adam([{'params': model.features.parameters()},
                                        {'params': model.classifier[0:6].parameters()},
                                        {'params': model.classifier[6].parameters(), 'lr': args.lr*10}
                                       ], lr=args.lr)    
                elif 'resnet' in args.model and linearAdaptor:
                    optimizer = optim.Adam([{'params': model[0].parameters(), 'lr': args.lr*10},
                                    {'params': model[1].parameters()}
                                   ], lr=args.lr)
                elif args.model == 'googlenet' or 'resnet' in args.model:
                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))
            print('-' * 80)
        print('\nTesting...')
        labels, predictions, test_accuracy = test(epoch, test_loader, log_name)
        #torch.save(model.state_dict(), '%s/%s_vgg-d.pth' % (args.save_dir,args.dataset))
        with open(os.path.join(args.save_dir,log_name), "a") as f:
            f.write(('-'*80)+'\n')
            f.write('Test accuracy: %.4f \n'%(test_accuracy))
            f.write(('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
            epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))+'\n')
            f.write(('-'*80)+'\n')
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


