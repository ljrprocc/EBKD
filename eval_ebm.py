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

from helper.sampling import get_replay_buffer

from helper.gen_loops import validate_G, generation_stage
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
    parser.add_argument('--warmup_iters', type=int, default=200, help="number of iters to linearly increase learning rate, -1 set no warmup.")

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210,250', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.3, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # Generator Details

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'cifar10', 'svhn'], help='dataset')

    # # I/O
    # parser.add_argument('--save_dir', type=str, default='../save/', help='The directory for saving the generated samples.')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8x4',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50','resnet28x10'])
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'batch', 'instance'])
    parser.add_argument('--model_stu', type=str, default='resnet8x4')
    parser.add_argument('--model_t', type=str, default='resnet32x4')
    
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--path_s', type=str, default=None, help="student model snapshot")

    parser.add_argument('--energy', default='mcmc', type=str, help='Sampling method to update EBM.')
    parser.add_argument('--lmda_lc', default=100, type=float, help='Hyperparameter for update EBM.')
    parser.add_argument('--lmda_l2', default=1.2e-5, type=float, help='Hyperparameter for l2-norm for generated loss')
    parser.add_argument('--lmda_tv', default=2.5e-4, type=float, help='Hyperparameter for total variation loss.')
    parser.add_argument('--lmda_adi', default=1., type=float, help='Hyperparameter for l2-norm for generated loss')
    parser.add_argument('--lmda_norm', default=1e-5, type=float, help='Hyperparameter for building p(x)')
    parser.add_argument('--lmda_p_x_y', default=0., type=float, help='Hyperparameter for building p(x,y)')
    parser.add_argument('--g_steps', default=40, type=int, help='Total MCMC steps for generating images.')
    parser.add_argument('--step_size', default=1, type=float, help='learning rate of MCMC updation.')
    parser.add_argument('--capcitiy', default=10000, type=int, help='Capcity of sample buffer.')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--reinit_freq', type=str, default=0.05, help='reinitialization frequency.')
    parser.add_argument('--print_every', type=int, default=20, help='reinitialization frequency.')
    parser.add_argument('--plot_uncond', action="store_true", help="Flag for saving class-conditional samples.")
    parser.add_argument('--plot_cond', action="store_true", help="Flag for saving class-conditional samples")
    parser.add_argument('--load_buffer_path', type=str, default=None, help='If not none, the loading path of replay buffer.')
    parser.add_argument('--n_valid', type=int, default=5000, help='Set validation data.')
    parser.add_argument('--labels_per_class', type=int, default=-1, help='Number of labeled examples per class.')
    parser.add_argument('--save_grid', action="store_true", help="Flag for saving the generated results.")
    parser.add_argument('--use_lc', action="store_true", help="Flag for saving the generated results.")
    parser.add_argument('--fresh', action="store_true", help="Flag for whether evaluate the classification result.")
    parser.add_argument('--short_run', action="store_true", help="Flag for whether evaluate the classification result.")
    parser.add_argument('--augment', action="store_true", help="Flag for whether evaluate the classification result.")
    parser.add_argument('--n_sample_steps', type=int, default=1000, help='Flag for refreshing the replay buffer.')
    parser.add_argument('--resume', type=str, default='none')
    parser.add_argument('--init_epoch', type=int, default=0)
    parser.add_argument('--gpu', default=0, type=int)

    opt = parser.parse_args()

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'


    # opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = '{}_{}_lr_{}_decay_{}_buffer_size_{}_lpx_{}_lpxy_{}_trial_{}_epoch_{}_gsteps_{}_step_size'.format(opt.model_s, opt.dataset, opt.learning_rate, opt.weight_decay, opt.capcitiy, opt.lmda_p_x, opt.lmda_p_x_y, opt.trial, opt.epochs, opt.g_steps, opt.step_size)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.device = "cuda:{}".format(opt.gpu)

    if opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'cifar10' or opt.dataset == 'svhn':
        opt.n_cls = 10
    else:
        opt.n_cls = 1000
    return opt

def main():
    opt = parse_option()
    strs = 'img_sample_eval/'
    ckpts = 'img_ckpts/'
    if opt.save_grid:
        strs = strs[:-1] + '_grid/'

    opt.save_dir = os.path.join(opt.save_folder, strs)
    opt.save_ckpt = os.path.join(opt.save_folder, ckpts)

    print(opt.save_dir)
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    if not os.path.exists(opt.save_ckpt):
        os.mkdir(opt.save_ckpt)
    # dataloader
    # model
    # model = model_dict[opt.model](num_classes=opt.n_cls, norm='batch')
    d, w = opt.model_s.split('x')[0][-2:], opt.model_s.split('x')[1]
    if opt.model_s == 'resnet28x10':
        model_score = model_dict[opt.model_s](depth=int(d), widen_factor=int(w), num_classes=opt.n_cls, norm='none')
    else:
        model_score = model_dict[opt.model_s](num_classes=opt.n_cls, norm='none')
    model_score = model_dict['Gen'](model=model_score, n_cls=opt.n_cls)
    if opt.use_lc:
        s = model_dict[opt.model_stu](num_classes=opt.n_cls, norm=opt.norm)
        t = model_dict[opt.model_t](num_classes=opt.n_cls, norm=opt.norm)
        print('==> Loading teacher and student model..')
        s.load_state_dict(torch.load(opt.path_s, map_location=opt.device)['model'])
        t.load_state_dict(torch.load(opt.path_t, map_location=opt.device)['model'])
        model_list = [model_score, s, t]
        s.to(opt.device)
        t.to(opt.device)
        print('==> Done.')
    # model = model_dict['Score'](model=model, n_cls=opt.n_cls)
    # print(model)
    # optimizer = nn.DataParallel(optimizer)
    
    if torch.cuda.is_available():
        model_score = model_score.to(opt.device)
        cudnns.benchmark = True
    
    opt.device = next(model_score.parameters()).device

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # buffer = SampleBuffer(net_T=opt.path_t, max_samples=opt.capcitiy)
    buffer, model_score = get_replay_buffer(opt, model=model_score)
    if opt.use_lc:
        buffer = generation_stage(model_list, buffer, opt)
    else:
        buffer = validate_G(model_score, buffer, opt)
    ckpt_dict = {
                "replay_buffer": buffer
    }
    torch.save(ckpt_dict, os.path.join(opt.save_folder, 'res_buffer.pts'))

    # routine
if __name__ == '__main__':
    main()
