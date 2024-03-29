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

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from datasets.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from datasets.imagenet import get_imagenet_dataloader, get_dataloader_sample
from datasets.svhn import get_svhn_dataloaders, get_svhn_dataloaders_sample

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL, HintLoss, Attention

from helper.loops import train_distill, validate
from helper.pretrain import init

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
    parser.add_argument('--learning_rate_ebm', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs_ebm', type=str, default='150,180,210,250', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_ebm', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay_ebm', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'cifar10', 'svhn'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50', 'ResNet18'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    parser.add_argument('--norm', type=str, default='none', choices=['none', 'batch', 'instance'])
    parser.add_argument('--df_folder', type=str, default='/data/lijingru/img_sample_eval_1000/', help='Root folder of generated dataset for KD.')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'correlation', 'vid', 'crd', 'kdsvd', 'fsp', 'rkd', 'pkt', 'abound', 'factor', 'nst', 'energy', 'ebkd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--datafree', action='store_true')
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])
    
    # Energy options
    parser.add_argument('--G_step', default=1, type=int, help='Iterations of updation for generator.')
    parser.add_argument('--D_step', default=1, type=int, help='Iterations of updation for distillation model.')
    # parser.add_argument('--datafree', action='store_true')
    parser.add_argument('--energy', default='mcmc', type=str, help='Sampling method to update EBM.')
    parser.add_argument('--lmda_ebm', default=0.7, type=float, help='Hyperparameter for update EBM.')

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
        opt.model_path = './save/student_datafree_model'
        opt.tb_path = './save/student_datafree_tensorboards'

    iterations = opt.lr_decay_epochs_ebm.split(',')
    opt.lr_decay_epochs_ebm = list([])
    for it in iterations:
        opt.lr_decay_epochs_ebm.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

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

def load_teacher(model_path, n_cls, opt):
    print('=====> loading teacher model.')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls, norm=opt.norm)
    if opt.dataset == 'imagenet':
        model = model_dict[model_t](num_classes=n_cls, pretrained=True)
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
    best_acc = 0.

    opt = parse_option()

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # Loading data, supporting CIFAR100 and ImageNet
    if opt.dataset == 'cifar100' or opt.dataset == 'cifar10':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size, num_workers=opt.num_workers, k=opt.nce_k, mode=opt.mode)
        train_loader, val_loader = get_cifar100_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 100 if opt.dataset == 'cifar100' else 10
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=True)
        n_cls = 1000
    elif opt.dataset == 'svhn':
        train_loader, val_loader = get_svhn_dataloaders(opt, batch_size=opt.batch_size, num_workers=opt.num_workers)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    use_ddp = opt.dataset == 'imagenet'
    if use_ddp:
        local_rank, device = setup_ranks()

    model_t = load_teacher(opt.path_t, n_cls, opt)
    model_s = model_dict[opt.model_s](num_classes=n_cls, norm=opt.norm)

    if opt.init_epochs > 0:
        print('==> Loading resumed epochs..')
        save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
        pth = torch.load(save_file)
        opt.init_epochs = pth['epoch']
        model_s.load_state_dict(pth['model'])
        print('Done.')

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)
    optimizer_list = []
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    # if opt.mode == 'energy':
    #     score = model_dict['Score'](depth=10, width=4, norm=None, dropout_rate=0, n_classes=n_cls)
    #     module_list.append(score)
        
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    elif opt.distill == 'energy':
        criterion_kd = EnergyKD(num_classes=n_cls, T=opt.kd_T, energy_model=model_s, te_model=model_t)
        energy_optim = optim.Adam(criterion_kd.parameters(), lr=0.001, weight_decay=opt.weight_decay, betas=(0.5, 0.99))
        optimizer_list.append(energy_optim)

    elif opt.distill == 'ebkd':
        criterion_kd = EBKD(T=opt.kd_T, teacher_path=opt.path_t, num_classes=n_cls)

        # trainable_list.append(criterion_kd.linear_block)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate_ebm,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay_ebm)
    
    optimizer_list.append(optimizer)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        if use_ddp:
            module_list = torch.nn.ModuleList([m.to(device) for m in module_list])
            module_list = torch.nn.ModuleList([torch.nn.parallel.DistributedDataParallel(m, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) for m in module_list])
            criterion_list.to(local_rank)
        else:
            module_list = torch.nn.ModuleList([torch.nn.DataParallel(m) for m in module_list])
            module_list.cuda()
            criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)
    # noise = torch.randn(128,3,32,32)
    # buffer = SampleBuffer(net_T=opt.path_t)
    # routine
    for epoch in range(opt.init_epochs + 1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train_distill(epoch, train_loader, module_list, criterion_list, optimizer_list, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        
        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            if (not use_ddp) or torch.distributed.get_rank() == 0:
                torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            if (not use_ddp) or torch.distributed.get_rank() == 0:
                torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        # 'score': score.state_dict()
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    if (not use_ddp) or torch.distributed.get_rank() == 0:
        torch.save(state, save_file)


if __name__ == '__main__':
    main()
