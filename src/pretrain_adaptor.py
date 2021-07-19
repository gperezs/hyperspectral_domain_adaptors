#! /usr/bin/env python

import os
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, './src/utils')
sys.path.insert(0, './src/models')
sys.path.insert(0, './src/datasets')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_du
from torch.autograd import Variable
import torchvision.models as models

from visualization_utils import  show_decoded_im
from adaptors import MediumAdaptor, MediumDecoder

from config import dset_root


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pre-train adaptor')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--model',  help='Adaptor model: medium refers to multi-layer adaptor',
                        default='medium', type=str)
    parser.add_argument('--dataset',  help='Datasets: cub, cars, aircrafts',
                        default='cub', type=str)
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
                        default='checkpoints/small_adaptor/', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network', default='', type=str)
    args = parser.parse_args()
    return args


def train(epoch, train_loader):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['image'].float()
        target = sample_batched['label'].float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        pred = output.detach().cpu().numpy()
        loss = mse(output, target)
        train_loss += loss.data#[0]
        accuracy_batch = np.sum((pred - target.detach().cpu().numpy())**2)/(np.prod(data.size()))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print_line = ('Train epoch: {} [{}/{} ({:3.0f}%)] Loss: {:2.4f} ({:2.3f}) Dist: {:.4f} lr: {:.0e} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.data, train_loss/(batch_idx+1.0),
                accuracy_batch, optimizer.param_groups[-1]['lr']))
            with open(os.path.join(args.save_dir,'log.txt'), "a") as f:
                f.write(print_line+"\n")
            print(print_line)
    if True and epoch%10 == 0:
        encmodel = model[0]
        imout = encmodel(data).detach().cpu().numpy()
        show_decoded_im(imout[-1,:,:,:], args.save_dir, epoch)
    return accuracy_batch, train_loss/((batch_idx+1.0)*len(data))


def test(epoch, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, sample_batched in enumerate(test_loader):
        data = sample_batched['image'].float()
        target = sample_batched['label'].float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        pred = output.detach().cpu().numpy()
        loss = mse(output, target)
        test_loss += loss.data
        accuracy_batch = np.sum((pred - target.detach().cpu().numpy())**2)/(np.prod(data.size()))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print_line = ('Test epoch: {} [{}/{} ({:3.0f}%)] Loss: {:2.4f} ({:2.3f}) Dist: {:.4f} lr: {:.0e} '.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader), test_loss.data, test_loss/(batch_idx+1.0),
                accuracy_batch, optimizer.param_groups[-1]['lr']))
            with open(os.path.join(args.save_dir,'log.txt'), "a") as f:
                f.write(print_line+"\n")
            print(print_line)
    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, '
          'Dist: {:.4f}\n '.format(test_loss, accuracy_batch))
    if True:
        encmodel = model[0]
        imout = encmodel(data).detach().cpu().numpy()
        show_decoded_im(imout[-1,:,:,:], args.save_dir, epoch)
    return accuracy_batch, test_loss/((batch_idx+1.0)*len(data))


if __name__ == '__main__':
    
    args = parse_args()
    args.save_dir = os.path.join('checkpoints', 'adaptors', '%s_adaptor_%s-%d'%(args.model,
                                                                args.dataset, args.channels))
    if args.dataset == 'cub':
        from cubdataset import CUBDataset as KDataset
    elif args.dataset == 'cars':
        from carsdataset import CarsDataset as KDataset
    elif args.dataset == 'aircrafts':
        from aircraftsdataset import AircraftsDataset as KDataset
    elif args.dataset == 'legus':
        from legusdataset import LEGUSDataset as KDataset
    elif args.dataset == 'so2sat':
        from so2sadataset import So2SaDataset as KDataset
    elif args.dataset == 'eurosat':
        from eurosatdataset import EuroSATDataset as KDataset

    args.cuda = args.cuda and torch.cuda.is_available()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    print('Called with args:')
    print(args)
    with open(os.path.join(args.save_dir,'log.txt'), "w") as f:
        f.write('Called with args:\n')
        f.write(str(args)+'\n')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load dataset
    train_dset = KDataset(dset_root[args.dataset], 
                            is_train=True, 
                            numdata=args.numdata,
                            nchannels=args.channels,
                            is_pretraining=True)
    train_loader = torch_du.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    test_dset  = KDataset(dset_root[args.dataset], 
                            is_train=False,
                            nchannels=args.channels,
                            is_pretraining=True)
    test_loader = torch_du.DataLoader(test_dset, batch_size=args.test_batch_size, shuffle=False)

    # load model
    model = MediumAdaptor(args.channels, True)
    decoder = MediumDecoder(args.channels)
    model = nn.Sequential(model, decoder) # autoencoder

    print(model)
    if args.cuda:
        model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters: %d'%(params))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_accuracy, train_loss = train(epoch, train_loader)
            if epoch % 10 == 0:
                _, _= test(epoch, test_loader)
            if epoch % 10 == 0:
                torch.save(model.state_dict(), '%s/%s_adaptor_%s-%d.pth' % (args.save_dir, 
                                                    args.model, args.dataset, args.channels)) 
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time)/3600.0))
            print('-' * 89)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

