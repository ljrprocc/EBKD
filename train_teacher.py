from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models import model_dict
import json

from datasets.cifar100 import get_cifar100_dataloaders
from datasets.imagenet import get_imagenet_dataloader, get_dataloader_sample
from train_student import load_teacher
from distiller_zoo import EBLoss

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    # optimization
    parser.add_argument('--learning_rate_ebm', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs_ebm', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_ebm', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay_ebm', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50' ])
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10','cifar100', 'imagenet'], help='dataset')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')
    parser.add_argument('--resume', action='store_true', help='whether to resume from path_t')
    parser.add_argument('--datafree', action='store_true', help='whether datafree distilling.')
    parser.add_argument('--mode', type=str, default='D', choices=['D', 'G'])
    parser.add_argument('--energy', default='mcmc', type=str, help='Sampling method to update EBM.')
    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate_ebm = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/model'
        opt.tb_path = '/path/to/my/tensorboard'
    else:
        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

    iterations = opt.lr_decay_epochs_ebm.split(',')
    opt.lr_decay_epochs_ebm = list([])
    for it in iterations:
        opt.lr_decay_epochs_ebm.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate_ebm,
                                                            opt.weight_decay_ebm, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100' or opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar100_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100 if opt.dataset == 'cifar100' else 10
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
        n_cls = 1000
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls, norm='batch')
    if opt.resume:
        model = load_teacher(opt.path_t, n_cls)

    criterion = nn.CrossEntropyLoss()
    
    # criterion = EBLoss(n_cls=n_cls, teacher_path=opt.path_t)

    # optimizer
    if opt.mode == 'D':
        
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate_ebm, momentum=opt.momentum,  weight_decay=opt.weight_decay_ebm)
    else:
        G = model_dict['Score'](image_size=32, num_classes=n_cls)
        optimizer = optim.Adam(G.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
        for p in model.parameters():
            p.requires_grad = False
    
    if torch.cuda.is_available():
        # G = G.cuda()
        # model = model.cuda()
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # buffer = SampleBuffer(net_T=opt.path_t)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if opt.mode == 'G':
            train_loss = train_generator(epoch, train_loader, [G], criterion, optimizer, opt, buffer)
        else:
            train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        if opt.mode != 'G':

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)
        
            test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

            logger.log_value('test_acc', test_acc, epoch)
            logger.log_value('test_acc_top5', test_acc_top5, epoch)
            logger.log_value('test_loss', test_loss, epoch)

            # save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    # 'embed': criterion.embed.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
                print('saving the best model!')
                torch.save(state, save_file)

            # regular saving
            if epoch % opt.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    # 'embed': criterion.embed.state_dict(),
                    'accuracy': test_acc,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

            # This best accuracy is only for printing purpose.
            # The results reported in the paper/README is from the last epoch.
            print('best accuracy:', best_acc)

        else:
            print('Sample Buffer:')
            print('length: %d' % (len(buffer)))
            print('Id Statistics:')
            ids = [k[1] for k in buffer.buffer]
            # print(ids[0].dtype)
            # res_dict = {
            #     'sample': [],
            #     'class_id': [],
            # }
            ids = torch.stack(ids)
            # print(i)
            sample = np.asarray([k[0].detach().numpy() for k in buffer.buffer])
            class_id = np.asarray([k[1].item() for k in buffer.buffer])
            # res_dict['class_id'] = [k[1].item() for k in buffer.buffer]

            for i in range(n_cls):
                # print(i)
                matched = torch.sum(ids == i).item()
                print('Class {}: {} samples.'.format(i, matched))
            print('Writing valid samples')
            

            if epoch % opt.save_freq == 0:
                np.savez(opt.save_folder, 'res_epoch_{epoch}.npz'.format(epoch=epoch), sample=sample, class_id=class_id)
                state = {
                    'opt': opt,
                    'model': model.state_dict(),
                    # 'G': G.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        # 'G': G.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
