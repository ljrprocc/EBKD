from __future__ import print_function

import os
import argparse
import socket
import time
import yaml
import json

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnns
from torch import distributed as dist

from models import model_dict

from datasets.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from datasets.imagenet import get_imagenet_dataloader, get_dataloader_sample
from datasets.svhn import get_svhn_dataloaders, get_svhn_dataloaders_sample

from helper.util import adjust_learning_rate, TVLoss
from helper.util_gen import getDirichl, add_dict

from helper.gen_loops import train_generator, train_joint, train_z_G, train_vae, train_coopnet
from helper.pretrain import init
from helper.sampling import get_replay_buffer

# DDP setting
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# from helper.util import SampleBuffer

def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print_frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--warmup_iters', type=int, default=200, help="number of iters to linearly increase learning rate, -1 set no warmup.")
    # parser.add_argument('--data_noise', type=float, default=0.03, help="The adding noise for sampling data point x~p_data.")

    # optimization
    # parser.add_argument('--learning_rate_ebm', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--lr_decay_epochs_ebm', type=str, default='150,180,210,250', help='where to decay lr, can be a list')
    # parser.add_argument('--lr_decay_rate_ebm', type=float, default=0.3, help='decay rate for learning rate')
    # parser.add_argument('--weight_decay_ebm', type=float, default=0.0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'imagenet', 'svhn'], help='dataset')
    parser.add_argument('--config', type=str, default='./configs/jem.yaml')

    # # I/O
    # parser.add_argument('--save_dir', type=str, default='../save/', help='The directory for saving the generated samples.')

    # model
    parser.add_argument('--joint', action="store_true", help='Flag for whether adding l_c term when training EBM.')

    parser.add_argument('--sampling', default='mcmc', type=str, help='Sampling method to update EBM.')
    parser.add_argument('--lmda_ebm', default=0.7, type=float, help='Hyperparameter for update EBM.')
    parser.add_argument('--lmda_l2', default=0.01, type=float, help='Hyperparameter for l2-norm for generated loss')
    parser.add_argument('--lmda_tv', default=2.5e-3, type=float, help='Hyperparameter for total variation loss.')
    # parser.add_argument('--lmda_p_x', default=1., type=float, help='Hyperparameter for building p(x)')
    # parser.add_argument('--lmda_p_x_y', default=0., type=float, help='Hyperparameter for building p(x,y)')
    parser.add_argument('--lmda_e', default=0.1, type=float, help='Hyperparameter for kl divergence of negative student and positive teacher.')
    parser.add_argument('--lc_K', default=5, type=int, help='Sample K steps for policy gradient. ')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--plot_uncond', action="store_true", help="Flag for saving class-conditional samples.")
    parser.add_argument('--plot_cond', action="store_true", help="Flag for saving class-conditional samples")
    parser.add_argument('--load_buffer_path', type=str, default=None, help='If not none, the loading path of replay buffer.')
    parser.add_argument('--n_valid', type=int, default=None, help='Set validation data.')
    parser.add_argument('--labels_per_class', type=int, default=-1, help='Number of labeled examples per class.')
    parser.add_argument('--cls', type=str, default='cls', choices=['cls', 'mi'])
    parser.add_argument('--st', type=int, default=-1, help="Inital sample step for policy gradient. -1 for random sample.")
    # model options
    parser.add_argument('--self_attn', action="store_true")
    parser.add_argument('--multiscale', action="store_true")
    parser.add_argument('--augment', action="store_true")
    # CUDA options
    parser.add_argument('--gpu', default=0, type=int, help='Local device.')

    # DDP options
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world_size', default=4, type=int, help='world size for learning DDP')

    opt = parser.parse_args()
    with open(opt.config, 'r') as f:
        opt_m = yaml.load(f)
        opt_m = add_dict(opt, opt_m)
    # Conditional generation for downstream KD tasks.
    opt.cond = True
    opt.spec_norm = False
    # Setting no data noise in short-run MCMC.
    if opt.short_run:
        opt.data_noise = 0.

    # set different learning rate from these 4 models
    # if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
    #     opt.learning_rate = 0.01
    opt.df_folder = '/data/lijingru/img_sample_eval_10000/'
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
    config_type = opt.config.split('/')[-1].split('.')[0]

    # opt.model_t = get_teacher_name(opt.path_t)
    if config_type == 'jem':
        if opt.joint:
            opt.model_name = '{}_T:{}_S:{}_{}_lr_{}_decay_{}_buffer_size_{}_lpx_{}_lpxy_{}_energy_mode_{}_step_size_{}_trial_{}_k_{}'.format(opt.model_s, opt.model, opt.model_stu, opt.dataset, opt.learning_rate_ebm, opt.weight_decay_ebm, opt.capcitiy, opt.lmda_p_x, opt.lmda_p_x_y, opt.energy, opt.step_size, opt.trial, opt.lc_K)
        else:
            opt.model_name = '{}_{}_lr_{}_decay_{}_buffer_size_{}_lpx_{}_lpxy_{}_energy_mode_{}_step_size_{}_g_steps_{}_trial_{}'.format(opt.model_s, opt.dataset, opt.learning_rate_ebm, opt.weight_decay_ebm, opt.capcitiy, opt.lmda_p_x, opt.lmda_p_x_y, opt.energy, opt.step_size, opt.g_steps, opt.trial)
    elif config_type == 'vae':
        opt.model_name = '{}_lr_{}_decay_{}_ndf_{}_trial_{}'.format(opt.dataset, opt.exp_params['LR'], opt.exp_params['weight_decay'], opt.model_params['latent_dim'], opt.trial)
        opt.batch_size = opt.exp_params['batch_size']

    elif config_type == 'coopnet':
        opt.model_name = opt.model_name = '{}_lr_{}_decay_{}_ndf_{}_trial_{}_{}_{}_lr_{}_decay_{}_buffer_size_{}_lpx_{}_lpxy_{}_energy_mode_{}_step_size_{}_g_steps_{}_trial_{}'.format(opt.dataset, opt.exp_params['LR'], opt.exp_params['weight_decay'], opt.model_params['latent_dim'], opt.trial, opt.model_s, opt.dataset, opt.learning_rate_ebm, opt.weight_decay_ebm, opt.capcitiy, opt.lmda_p_x, opt.lmda_p_x_y, opt.energy, opt.step_size, opt.g_steps, opt.trial)

    else:
        if not opt.joint:
            opt.model_name = '{}_{}_ngf_{}_ndf_{}_elr_{}_glr_{}_trial_{}'.format(opt.dataset, opt.model_s, opt.ngf, opt.ndf, opt.e_lr, opt.g_lr, opt.trial)
        else:
            opt.model_name = '{}_ngf_{}_ndf_{}_elr_{}_glr_{}_lcK_{}_st_{}_trial_{}'.format(opt.dataset, opt.ngf, opt.ndf, opt.e_lr, opt.g_lr, opt.lc_K, opt.st, opt.trial)
    

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
    model = model_dict[model_t](num_classes=opt.n_cls, norm='batch')
    if opt.dataset == 'imagenet':
        model = model_dict[model_t](num_classes=opt.n_cls, pretrained=True)
    else:
        model.load_state_dict(torch.load(model_path, map_location=opt.device)['model'])
    print('===> done.')
    return model

def setup_ranks():
    os.environ['OPM_NUM_THREADS'] = '1'
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    os.environ['OPM_NUM_THREADS'] = '1'
    master_port = os.environ.get("MASTER_PORT", None)
    # print(rank, world_size)
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device

def cleanup():
    dist.destroy_process_group()

def set_optimizers(opt, model_score, model_G=None, config_type='jem', model_s=None):
    if config_type == 'jem' or config_type == 'coopnet':
        optimizer = optim.Adam(model_score.parameters(), lr=opt.learning_rate_ebm, betas=[0.9, 0.999],  weight_decay=opt.weight_decay_ebm)
        if config_type == 'coopnet':
            optimizer_alpha = optim.Adam(model_score.parameters(), lr=opt.exp_params['LR'], betas=[0.9, 0.999],  weight_decay=opt.exp_params['weight_decay'])
            return optimizer, optimizer_alpha
        return optimizer
    elif config_type == 'gz':
        assert model_G is not None, 'Must set generator in latent code mode.'
        optE = torch.optim.Adam(model_score.parameters(), lr=opt.e_lr, weight_decay=opt.e_decay, betas=(opt.e_beta1, opt.e_beta2))
        optG = torch.optim.Adam(model_G.parameters(), lr=opt.g_lr, weight_decay=opt.g_decay, betas=(opt.g_beta1, opt.g_beta2))

        lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optE, opt.e_gamma)
        lr_scheduleG = torch.optim.lr_scheduler.ExponentialLR(optG, opt.g_gamma)
        return optE, optG, lr_scheduleE, lr_scheduleG
    elif config_type == 'vae':
        optimizer = optim.Adam(model_score.parameters(), lr=opt.exp_params['LR'], betas=[0.9, 0.999],  weight_decay=opt.exp_params['weight_decay'])
        optimizer_s = optim.Adam(model_score.parameters(), lr=opt.joint_training['stu_lr'], betas=[0.5, 0.999],  weight_decay=opt.exp_params['weight_decay'])
        return optimizer, optimizer_s
    else:
        raise NotImplementedError('Not implemented at type {}'.format(config_type))



def main_function(gpu, opt):
    
    ddp = opt.dataset == 'imagenet'
    if ddp:
        local_rank, device = setup(gpu, opt.world_size)
    opt.save_dir = os.path.join(opt.save_folder, 'img_samples/')
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    # dataloader
    # print(local_rank, gpu)
    # exit(-1)
    opt.datafree = False
    opt.device = 'cuda:{}'.format(opt.gpu)
    # 
    if opt.dataset == 'cifar100' or opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar100_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers, use_subdataset=True)
        opt.n_cls = 100 if opt.dataset == 'cifar100' else 10
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(opt, batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True, use_subdataset=True)
        opt.n_cls = 1000
    elif opt.dataset == 'svhn':
        train_loader, val_loader = get_svhn_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers, use_subdataset=True)
        opt.n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # print(gpu)
    # model
    # model = model_dict[opt.model](num_classes=opt.n_cls, norm='batch')

    config_type = opt.config.split('/')[-1].split('.')[0]
    assert config_type in ['gz', 'jem', 'vae', 'coopnet']
    if config_type == 'gz':
        model_score = model_dict['ZE'](args=opt, num_classes=opt.n_cls)
        netG = model_dict['ZGc'](args=opt)
    elif config_type == 'jem' or config_type == 'coopnet':
        if opt.model_s == 'resnet28x10':
            d, w = opt.model_s.split('x')[0][-2:], opt.model_s.split('x')[1]
            model_score = model_dict[opt.model_s](depth=int(d), widen_factor=int(w), num_classes=opt.n_cls, norm=opt.norm)
        elif opt.model_s == 'Energy':
            model_score = model_dict[opt.model_s](args=opt, num_classes=opt.n_cls)
        else:
            model_score = model_dict[opt.model_s](num_classes=opt.n_cls, norm=opt.norm, act=opt.act, multiscale=opt.multiscale)
        netG = None
        if config_type == 'coopnet':
            netG = model_dict['cvae'](**opt.model_params)

    else:
        model_score = model_dict['cvae'](**opt.model_params)
        netG = None
    
    if config_type != 'vae':
        model_score = model_dict['Gen'](model=model_score, n_cls=opt.n_cls)
        prior_buffer, _ = get_replay_buffer(opt, model=model_score, config_type=config_type, model_G=netG)
        post_buffer, _ = get_replay_buffer(opt, model=model_score, config_type=config_type, model_G=netG)
        buffer_lists = [prior_buffer, post_buffer]
    
    
    optimizer = set_optimizers(opt, model_score=model_score, model_G=netG, config_type=config_type)
    

    if config_type == 'gz':
        optE, optG, lrE, lrG = optimizer
        optimizer_list = [optG, optE]
        print(opt.path_t)
        model = load_teacher(opt.path_t, opt)
    elif config_type == 'jem':
        opt.y = getDirichl(opt.path_t, device=opt.device)
        if opt.joint:
            model = load_teacher(opt.path_t, opt)
            model_stu = model_dict[opt.model_stu](num_classes=opt.n_cls, norm='batch')
    else:
        opt.y = getDirichl(opt.path_t, device=opt.device)
        model = load_teacher(opt.path_t, opt)
        model_stu = model_dict[opt.model_stu](num_classes=opt.n_cls, norm='batch')
    

    if torch.cuda.is_available():
        if ddp:
            model_score = model_score.to(device)
            # criterion = criterion.to(gpu)
            model_score = DDP(model_score, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

            if opt.joint:
                model_stu = model_stu.to(device)
                model = model.to(device)
                model_stu = DDP(model_stu, device_ids=[local_rank], output_device=local_rank)
                model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        else:
            model_score = model_score.to(opt.device)
            if config_type == 'gz' or config_type == 'coopnet':
                netG = netG.to(opt.device)
                model = model.to(opt.device)
            # criterion = criterion.cuda()
            elif config_type == 'vae':
                model = model.to(opt.device)
                # model = model.to(opt.device)
                model_stu = model_stu.to(opt.device)
            else:
                if opt.joint:
                    model = model.to(opt.device)
                    model_stu = model_stu.to(opt.device)

        cudnns.benchmark = True
        cudnns.enabled = True
    # model = model_dict['Score'](model=model, n_cls=opt.n_cls)
    # print(model)
    if config_type == 'gz':
        model_list = [model_score, netG]
        criterion = torch.nn.MSELoss(reduction='sum')
    elif config_type == 'vae':
        model_list = [model, model_score]

    elif config_type == 'coopnet':
        model_list = [model_score, model, netG]
        optimizer_adjust = optimizer[0]
    else:
        if opt.joint:
            model_list = [model, model_stu, model_score]
        else:
            model_list = [model_score]
        optimizer_adjust = optimizer
    # optimizer = nn.DataParallel(optimizer)
    # criterion = TVLoss()

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # buffer = SampleBuffer(net_T=opt.path_t, max_samples=opt.capcitiy)
    
    
    if (not ddp) or gpu == 0:
        print(opt)
        with open(os.path.join(opt.save_folder, 'hyper'), 'w') as f:
            f.write(str(opt))
    # print(opt.y)
    # routine
    for epoch in range(opt.init_epochs+1, opt.epochs + 1):
        
        if config_type == 'jem' or 'coopnet':
            if epoch in opt.lr_decay_epochs_ebm:
                for param_group in optimizer_adjust.param_groups:
                    new_lr = param_group['lr'] * opt.lr_decay_rate_ebm
                    param_group['lr'] = new_lr
            adjust_learning_rate(epoch, opt, optimizer_adjust)
        if (not ddp) or gpu == 0:
            print("==> training...")

        time1 = time.time()
        if ddp:
            # for loader in train_loader:
            loader, loader_lb = train_loader
            loader.sampler.set_epoch(epoch)
            # loader_lb.sampler.set_epoch(epoch)
        if config_type == 'jem':
            if opt.joint:
                train_loss = train_joint(epoch, train_loader, model_list, optimizer, opt, prior_buffer, logger, device=opt.device)
            else:
                train_loss = train_generator(epoch, train_loader, model_list, optimizer, opt, prior_buffer, logger, local_rank=gpu, device=opt.device)
        elif config_type == 'vae':
            train_loss = train_vae(model_list, optimizer[0], opt, train_loader, logger, epoch, model_s=model_stu, optimizer_s=optimizer[1])
        elif config_type == 'coopnet':
            train_loss = train_coopnet(model_list, optimizer, opt, train_loader, logger, epoch, prior_buffer)
        else:
            train_loss = train_z_G(epoch, buffer_lists, train_loader, model_list, criterion, optimizer_list, opt, logger, local_rank=gpu, device=opt.device, model_cls=model if opt.joint else None)
            lrE.step(epoch=epoch)
            lrG.step(epoch=epoch)
        
        time2 = time.time()
        # logger.log_value('train_loss', train_loss, epoch)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # save the best model
        # print('Saving Sampling Buffer')
        # print('Sample Buffer:')
        # print('length: %d' % (len(buffer)))
        
        if epoch % opt.save_freq == 0:
            print('***********Writing valid samples**************')
            if config_type == 'jem':
                ckpt_dict = {
                    "model_state_dict": model_score.state_dict(),
                    "replay_buffer": prior_buffer if not opt.short_run else None
                }

            elif config_type == 'gz':
                ckpt_dict = {
                    "model_state_dict": model_score.state_dict(),
                    "G_state_dict": netG.state_dict(),
                    "optG": optG.state_dict(),
                    "optE": optE.state_dict()
                }

            elif config_type == 'coopnet':
                ckpt_dict = {
                    "model_state_dict": model_score.state_dict(),
                    "G_state_dict": netG.state_dict(),
                    "optG": optimizer[1].state_dict(),
                    "optE": optimizer[0].state_dict(),
                }
            else:
                ckpt_dict = {
                    "model_state_dict": model_score.state_dict(),
                    "optimizer": optimizer[0].state_dict(),
                    "optimizer_stu": optimizer[1].state_dict()
                }
            if gpu == 0:
                torch.save(ckpt_dict, os.path.join(opt.save_folder, 'res_epoch_{epoch}.pts'.format(epoch=epoch)))

def main():
    opt = parse_option()
    gpu_nums = torch.cuda.device_count()
    ddp = opt.dataset == 'imagenet'
    if ddp and gpu_nums > 1:
        mp.spawn(main_function, nprocs=gpu_nums, args=(opt,))
    else:
        main_function(0, opt)

if __name__ == '__main__':
    import random
    random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    cudnns.benchmark = True
    cudnns.enabled = True
    cudnns.deterministic = True
    main()
