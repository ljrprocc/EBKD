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
from helper.util_gen import get_replay_buffer, getDirichl

from helper.loops import train_generator, train_nce_G
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
    parser.add_argument('--warmup_iters', type=int, default=200, help="number of iters to linearly increase learning rate, -1 set no warmup.")
    parser.add_argument('--data_noise', type=float, default=0.03, help="The adding noise for sampling data point x~p_data.")

    # optimization
    # EBM option
    parser.add_argument('--learning_rate_ebm', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs_ebm', type=str, default='150,180,210,250', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_ebm', type=float, default=0.3, help='decay rate for learning rate')
    parser.add_argument('--weight_decay_ebm', type=float, default=0.0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    # VAE option
    parser.add_argument('--learning_rate_vae', type=float, default=0.005, help='learning rate for vae model')
    parser.add_argument('--weight_decay_vae', type=float, default=0.0, help='weight decay for ebm')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, help= 'gamma for lr_scheduler.')

    # Generator Details
    parser.add_argument('--g_steps', type=int, default=100, help='Updating steps for x')
    parser.add_argument('--g_lr', type=float, default=0.025, help='Start learning rate of update_stpes')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')

    # # I/O
    # parser.add_argument('--save_dir', type=str, default='../save/', help='The directory for saving the generated samples.')

    # model
    parser.add_argument('--model_vae', type=str, default='cvae',
                        choices=['vae', 'cvae'])
    parser.add_argument('--model_s', type=str, default='resnet8x4',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'resnet20x10','resnet26x10', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50','resnet32x10'])
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'batch', 'instance', 'spectral'])
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'leaky', 'swish'])
    
    
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    parser.add_argument('--energy', default='mcmc', type=str, help='Sampling method to update EBM.')
    parser.add_argument('--lmda_v', default=0.01, type=float, help='Hyperparameter for update VAE.')
    parser.add_argument('--steps', default=20, type=int, help='Total MCMC steps for generating images.')
    parser.add_argument('--step_size', default=1, type=float, help='learning rate of MCMC updation.')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--plot_uncond', action="store_true", help="Flag for saving class-conditional samples.")
    parser.add_argument('--plot_cond', action="store_true", help="Flag for saving class-conditional samples")
    parser.add_argument('--n_valid', type=int, default=5000, help='Set validation data.')
    parser.add_argument('--labels_per_class', type=int, default=-1, help='Number of labeled examples per class.')
    parser.add_argument('--cls', type=str, default='cls', choices=['cls', 'mi'])


    # DDP options
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate_ebm = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs_ebm.split(',')
    opt.lr_decay_epochs_ebm = list([])
    for it in iterations:
        opt.lr_decay_epochs_ebm.append(int(it))

    # opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = '{}_{}_lr_{}_decay_{}_lrvae_{}_gamma_{}_energy_mode_{}_trial_{}'.format(opt.model_s, opt.dataset, opt.learning_rate_ebm, opt.weight_decay_ebm, opt.learning_rate_vae, opt.scheduler_gamma, opt.energy, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


# def get_teacher_name(model_path):
#     segments = model_path.split('/')[-2].split('_')
#     if segments[0] != 'wrn':
#         return segments[0]
#     else:
#         return segments[0] + '_' + segments[1] + '_' + segments[2]

# def load_teacher(model_path, opt):
#     print('=====> loading teacher model.')
#     model_t = get_teacher_name(model_path)
#     model = model_dict[model_t](num_classes=opt.n_cls, norm='batch')
#     if opt.dataset == 'imagenet':
#         model = model_dict[model_t](num_classes=opt.n_cls, pretrained=True)
#     else:
#         model.load_state_dict(torch.load(model_path)['model'])
#     print('===> done.')
#     return model

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
    opt.datafree = False
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers, use_subdataset=True)
        opt.n_cls = 100
        img_size = 32
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True, use_subdataset=True)
        opt.n_cls = 1000
        img_size = 256
    else:
        raise NotImplementedError(opt.dataset)
    
    # model
    # model = model_dict[opt.model](num_classes=opt.n_cls, norm='batch')
    # model = load_teacher(opt.path_t, opt)

    model_score = model_dict[opt.model_s](num_classes=opt.n_cls, norm=opt.norm, use_latent=True, img_size=img_size, latent_dim=100)
    model_score = model_dict['Score'](model=model_score, n_cls=opt.n_cls)
    model_vae = model_dict[opt.model_vae](num_classes=opt.n_cls, in_channels=3, latent_dim=100, img_size=img_size)
    # model = model_dict['Score'](model=model, n_cls=opt.n_cls)
    # print(model)
    optimizer = optim.Adam(model_score.parameters(), lr=opt.learning_rate_ebm, betas=[0.9, 0.999],  weight_decay=opt.weight_decay_ebm)
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=opt.learning_rate_vae, weight_decay=opt.weight_decay_vae)
    scheduler_vae = optim.lr_scheduler.ExponentialLR(optimizer_vae, gamma=opt.scheduler_gamma)
    model_list = [model_vae, model_score]
    optimizer_list = [optimizer_vae, optimizer]
    # optimizer = nn.DataParallel(optimizer)
    # criterion = TVLoss()
    if torch.cuda.is_available():
        model_vae = model_vae.cuda()
        model_score = model_score.cuda()
        # criterion = criterion.cuda()
        cudnns.benchmark = False
        cudnns.deterministic = True


    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # buffer = SampleBuffer(net_T=opt.path_t, max_samples=opt.capcitiy)
    # buffer, _ = get_replay_buffer(opt, model=model_score)
    # opt.y = getDirichl(opt.path_t)
    # print(opt.y)
    # routine
    for epoch in range(opt.init_epochs+1, opt.epochs + 1):
        if epoch in opt.lr_decay_epochs_ebm:
            for param_group in optimizer.param_groups:
                new_lr = param_group['lr'] * opt.lr_decay_rate_ebm
                param_group['lr'] = new_lr

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_loss = train_nce_G(epoch, train_loader, model_list, optimizer_list, opt, logger)
        time2 = time.time()
        logger.log_value('train_loss', train_loss, epoch)
        scheduler_vae.step()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # save the best model
        # print('Saving Sampling Buffer'
        
        if epoch % opt.save_freq == 0:
            print('Writing valid samples')
            ckpt_dict = {
                "ebm_state_dict": model_score.state_dict(),
                "vae_state_dict": model_vae.state_dict(),
                "vae_opt_state_dict": optimizer_vae.state_dict(),
                "ebm_opt_state_dict": optimizer.state_dict()
            }
            torch.save(ckpt_dict, os.path.join(opt.save_folder, 'res_epoch_{epoch}.pts'.format(epoch=epoch)))

if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(4321)
    main()
