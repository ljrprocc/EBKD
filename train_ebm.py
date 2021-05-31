from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnns
import numpy as np

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from datasets.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from datasets.imagenet import get_imagenet_dataloader, get_dataloader_sample

from helper.util import adjust_learning_rate, TVLoss
from helper.util_gen import get_replay_buffer

from helper.loops import train_generator
from helper.pretrain import init
# from helper.util import SampleBuffer

def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print_frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=0, help='init training for two-stage methods and resume')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210,250', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # Generator Details
    parser.add_argument('--g_steps', type=int, default=100, help='Updating steps for x')
    parser.add_argument('--g_lr', type=float, default=0.025, help='Start learning rate of update_stpes')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')

    # # I/O
    # parser.add_argument('--save_dir', type=str, default='../save/', help='The directory for saving the generated samples.')

    # model
    parser.add_argument('--model', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50' ])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--energy', default='mcmc', type=str, help='Sampling method to update EBM.')
    parser.add_argument('--lmda_ebm', default=0.7, type=float, help='Hyperparameter for update EBM.')
    parser.add_argument('--lmda_l2', default=1.2e-5, type=float, help='Hyperparameter for l2-norm for generated loss')
    parser.add_argument('--lmda_tv', default=2.5e-3, type=float, help='Hyperparameter for total variation loss.')
    parser.add_argument('--lmda_p_x', default=1., type=float, help='Hyperparameter for building p(x)')
    parser.add_argument('--lmda_p_x_y', default=0., type=float, help='Hyperparameter for building p(x,y)')
    parser.add_argument('--steps', default=20, type=int, help='Total MCMC steps for generating images.')
    parser.add_argument('--step_size', default=1, type=float, help='learning rate of MCMC updation.')
    parser.add_argument('--capcitiy', default=10000, type=int, help='Capcity of sample buffer.')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--reinit_freq', type=str, default=0.05, help='reinitialization frequency.')
    parser.add_argument('--plot_uncond', action="store_true", help="Flag for saving class-conditional samples.")
    parser.add_argument('--plot_cond', action="store_true", help="Flag for saving class-conditional samples")
    parser.add_argument('--load_buffer_path', type=str, default=None, help='If not none, the loading path of replay buffer.')
    parser.add_argument('--n_valid', type=int, default=5000, help='Set validation data.')
    parser.add_argument('--labels_per_class', type=int, default=-1, help='Number of labeled examples per class.')


    # DDP options
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = '{}_{}_lr_{}_decay_{}_buffer_size{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.capcitiy, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

def load_teacher(model_path, opt):
    print('=====> loading teacher model.')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=opt.n_cls)
    if opt.dataset == 'imagenet':
        model = model_dict[model_t](num_classes=opt.n_cls, pretrained=True)
    else:
        model.load_state_dict(torch.load(model_path)['model'])
    print('===> done.')
    return model

def setup_ranks():
    os.environ['OPM_NUM_THREADS'] = '1'
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device

def main():
    opt = parse_option()
    opt.save_dir = os.path.join(opt.save_folder, 'img_samples/')
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers, use_subdataset=True)
        opt.n_cls = 100
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True, use_subdataset=True)
        opt.n_cls = 1000
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=opt.n_cls)
    model = load_teacher(opt.path_t, opt)
    model = model_dict['Score'](model=model, n_cls=opt.n_cls)
   
    print(model)
        
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum,  weight_decay=opt.weight_decay)
    # optimizer = nn.DataParallel(optimizer)
    criterion = TVLoss()
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnns.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # buffer = SampleBuffer(net_T=opt.path_t, max_samples=opt.capcitiy)
    buffer = get_replay_buffer(opt)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_loss = train_generator(epoch, train_loader, model, criterion, optimizer, opt, buffer)
        time2 = time.time()
        logger.log_value('train_loss', train_loss, epoch)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # save the best model
        # print('Saving Sampling Buffer')
        print('Sample Buffer:')
        print('length: %d' % (len(buffer)))
        print('Id Statistics:')
            

        if epoch % opt.save_freq == 0:
            print('Writing valid samples')
            ckpt_dict = {
                "model_state_dict": model.state_dict(),
                "replay_buffer": buffer
            }
            torch.save(ckpt_dict, os.path.join(opt.save_folder, 'res_epoch_{epoch}.pts'.format(epoch=epoch)))

if __name__ == '__main__':
    main()