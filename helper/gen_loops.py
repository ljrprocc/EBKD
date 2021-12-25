from __future__ import print_function, division

import sys
import time
import torch
import os
import torch.optim as optim
from torch.autograd import grad
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
import random
import tqdm
import shutil

from math import sqrt
sys.path.append('..')
from datasets.cifar100 import CIFAR100Gen
 
from .util import AverageMeter, accuracy, set_require_grad, print_trainable_paras, inception_score, TVLoss
from .util_gen import update_theta, getDirichl, diag_normal_NLL
from .util_gen import cond_samples
from torch.autograd import Variable
from .sampling import langevin_at_z, langevin_at_x

def train_joint(epoch, train_loader, model_list, optimizer, opt, buffer, logger, device=None):
    '''One epoch for training generator with teacher'''
    model_t, model_s, model = model_list
    model.train()
    model_t.eval()
    model_s.eval()
    # module_list[0].eval()
    # model = module_list[-1]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fpxs = AverageMeter()
    fqxs = AverageMeter()
    fpxys = AverageMeter()
    fqxys = AverageMeter()
    accs = AverageMeter()
    # noise = torch.randn(128, 3, 32, 32)
    train_loader, train_labeled_loader = train_loader
    # criterion is tv loss
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))

    end = time.time()
    sample_q = langevin_at_x(opt, device=device)
    correct = 0
    total_length = 0
    for idx, data in enumerate(train_loader):
        if idx <= opt.warmup_iters:
            lr = opt.learning_rate_ebm * idx / float(opt.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        data_time.update(time.time() - end)
        nldata = train_labeled_loader.__next__()
        x_lab, y_lab = nldata[0].to(device), nldata[1].to(device)

        input, target = data[0], data[1]
        input = input.float()
        # noise = torch.randn(input.shape[0], 100)
        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)
            # noise = noise.to(device)

        # ===================forward=====================
        # input_fake, [mu, logvar] = G(noise, target, return_feat=True)
        # output = model(input_fake)
        loss_ebm = 0
        
        if opt.energy == 'mcmc':
            loss_ebm, cache_p_x, cache_p_y, logit, ls = update_theta(opt, buffer, model_list, input, x_lab, y_lab, mode='joint', y_p=target)
        elif opt.energy == 'ssm':
            loss_ebm, score_x, score_xy = ssm_sample(opt, buffer, model, input, x_lab, y_lab)
        else:
            raise NotImplementedError('Not implemented.')
        optimizer.zero_grad()
        model.zero_grad()
        loss_ebm.backward()
        optimizer.step()
        losses.update(loss_ebm, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if opt.energy == 'mcmc':
            if opt.lmda_p_x > 0:
                fpx, fqx = cache_p_x
                fpxs.update(fpx, input.size(0))
                fqxs.update(fqx, input.size(0))
            if opt.lmda_p_x_y > 0:
                fpxy, fqxy = cache_p_y
                fpxys.update(fpxy, input.size(0))
                fqxys.update(fqxy, input.size(0))
        elif opt.energy == 'ssm':
            if opt.lmda_p_x > 0:
                # print(score_x.shape)
                fqxs.update(score_x.mean(), input.size(0))
            if opt.lmda_p_x_y > 0:
                fqxys.update(score_xy.mean(), input.size(0))
        

        # tensorboard logger
        # l_p_x, l_p_x_y, l_cls, l_c, l_c_k_minus_1, l2_k, l_cls_k, l_e_k, l2_k_1, l_cls_k_1, l_e_k_1 = ls
        l_p_x, l_p_x_y, l_cls, l_c, l_c_k_minus_1, l_cls_k, l_e_k, l_cls_k_1, l_e_k_1 = ls
        acc = torch.sum(torch.argmax(logit, 1) == y_lab).item() / input.size(0)
        accs.update(acc, input.size(0))
        global_iter = epoch * len(train_loader) + idx
        if global_iter % opt.print_freq == 0:
            logger.log_value('l_p_x', l_p_x, global_iter)
            logger.log_value('l_p_x_y', l_p_x_y, global_iter)
            logger.log_value('l_cls', l_cls, global_iter)
            logger.log_value('l_image_c_k', l_c, global_iter)
            logger.log_value('l_image_c_k_1', l_c_k_minus_1, global_iter)
            # logger.log_value('l_2_k', l2_k, global_iter)
            logger.log_value('l_cls_k', l_cls_k, global_iter)
            logger.log_value('l_e_k', l_e_k, global_iter)
            # logger.log_value('l_2_k_1', l2_k_1, global_iter)
            logger.log_value('l_cls_k_1', l_cls_k_1, global_iter)
            logger.log_value('l_e_k_1', l_e_k_1, global_iter)
            logger.log_value('accuracy', acc, global_iter)
        
        accs.update(acc, input.size(0))
        # print info
        if idx % opt.print_freq == 0:
            string = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses)
            if opt.lmda_p_x > 0:
                if opt.energy != 'ssm':
                    string += 'p(x) f(x+) {fpx.val:.4f} ({fpx.avg:.4f})\t'.format(fpx=fpxs)
                # print(fqxs.count)
                string += 'f(x-) {fqx.val:.4f} ({fqx.avg:.4f})\n'.format(fqx=fqxs)
            if opt.lmda_p_x_y > 0:
                if opt.energy != 'ssm':
                    string += 'p(x, y) f(x+) {fpxy.val:.4f} ({fpxy.avg:.4f})\t'.format(fpxy=fpxys)
                string += 'f(x-) {fqxy.val:.4f} ({fqxy.avg:.4f})\n'.format(fqxy=fqxys)
            string += 'Acc: {accs.val:.4f} ({accs.avg:.4f})\n'.format(accs=accs)
            print(string)
            sys.stdout.flush()
            if opt.plot_uncond:
                y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(input.device)
                x_q, _ = sample_q(model, buffer, y=y_q)
                plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q)
            if opt.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, opt.n_cls).to(input.device)
                # print(y.shape)
                x_q_y, _ = sample_q(model, buffer, y=y)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)

    return losses.avg


def train_generator(epoch, train_loader, model_list, optimizer, opt, buffer, logger, local_rank=None, device=None):
    '''One epoch for training generator with teacher'''
    '''One epoch for training generator with teacher'''
    # model_t, model = model_list
    model = model_list[0]
    model.train()
    # model_t.eval()
    # module_list[0].eval()
    # model = module_list[-1]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fpxs = AverageMeter()
    fqxs = AverageMeter()
    fpxys = AverageMeter()
    fqxys = AverageMeter()
    accs = AverageMeter()
    # noise = torch.randn(128, 3, 32, 32)
    train_loader, train_labeled_loader = train_loader
    # criterion is tv loss
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))

    end = time.time()
    sample_q = langevin_at_x(opt, device=device)
    correct = 0
    total_length = 0
    for idx, data in enumerate(train_loader):
        # print(len(data))
        input = data[0]
        target = data[1]
        if idx <= opt.warmup_iters:
            lr = opt.learning_rate_ebm * idx / float(opt.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        data_time.update(time.time() - end)
        nldata = train_labeled_loader.__next__()
        x_lab, y_lab = nldata[0], nldata[1] 
        x_lab, y_lab = x_lab.to(device), y_lab.to(device)

        input = input.float()
        # noise = torch.randn(input.shape[0], 100)
        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)
            # noise = noise.to(device)

        # ===================forward=====================
        # input_fake, [mu, logvar] = G(noise, target, return_feat=True)
        # output = model(input_fake)
        loss_ebm = 0
        
        if opt.energy == 'mcmc':
            loss_ebm, cache_p_x, cache_p_y, logit, ls = update_theta(opt, buffer, model_list, input, x_lab, y_lab, y_p=target)    
        elif opt.energy == 'ssm':
            loss_ebm, score_x, score_xy = ssm_sample(opt, buffer, model, input, x_lab, y_lab)
        else:
            raise NotImplementedError('Not implemented.')
        optimizer.zero_grad()
        model.zero_grad()
        loss_ebm.backward()
        optimizer.step()
        losses.update(loss_ebm, input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if opt.energy == 'mcmc':
            if opt.lmda_p_x > 0:
                fpx, fqx = cache_p_x
                fpxs.update(fpx, input.size(0))
                fqxs.update(fqx, input.size(0))
            if opt.lmda_p_x_y > 0:
                fpxy, fqxy = cache_p_y
                fpxys.update(fpxy, input.size(0))
                fqxys.update(fqxy, input.size(0))
        elif opt.energy == 'ssm':
            if opt.lmda_p_x > 0:
                # print(score_x.shape)
                fqxs.update(score_x.mean(), input.size(0))
            if opt.lmda_p_x_y > 0:
                fqxys.update(score_xy.mean(), input.size(0))
        

        # tensorboard logger
        l_p_x, l_p_x_y, l_cls = ls
        acc = torch.sum(torch.argmax(logit, 1) == y_lab).item() / input.size(0)
        accs.update(acc, input.size(0))
        global_iter = epoch * len(train_loader) + idx
        if global_iter % opt.print_freq == 0:
            logger.log_value('l_p_x', l_p_x, global_iter)
            logger.log_value('l_p_x_y', l_p_x_y, global_iter)
            logger.log_value('l_cls', l_cls, global_iter)
            logger.log_value('accuracy', acc, global_iter)
        
        accs.update(acc, input.size(0))
        ddp = (opt.dataset == 'imagenet')
        # print info
        if idx % opt.print_freq == 0:
            string = 'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses)
            if opt.lmda_p_x > 0:
                if opt.energy != 'ssm':
                    string += 'p(x) f(x+) {fpx.val:.4f} ({fpx.avg:.4f})\t'.format(fpx=fpxs)
                # print(fqxs.count)
                string += 'f(x-) {fqx.val:.4f} ({fqx.avg:.4f})\n'.format(fqx=fqxs)
            if opt.lmda_p_x_y > 0:
                if opt.energy != 'ssm':
                    string += 'p(x, y) f(x+) {fpxy.val:.4f} ({fpxy.avg:.4f})\t'.format(fpxy=fpxys)
                string += 'f(x-) {fqxy.val:.4f} ({fqxy.avg:.4f})\n'.format(fqxy=fqxys)
            string += 'Acc: {accs.val:.4f} ({accs.avg:.4f})\n'.format(accs=accs)
            if (not ddp) or local_rank == 0:
                print(string)
            sys.stdout.flush()
            if opt.plot_uncond:
                y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(input.device)
                x_q, _ = sample_q(model, buffer, y=y_q)
                plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q)
            if opt.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, opt.n_cls).to(device)
                # print(y.device, device)
                # print(y.shape)
                x_q_y, _ = sample_q(model, buffer, y=y)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)

    return losses.avg

def train_z_G(epoch, buffer, train_loader, model_list, criterion, optimizer, opt, logger, local_rank=None, device=None, model_cls=None):
    '''
    Sampling from z~p_{\theta}(z), x~p_{\alpha}(x|z)
    '''
    model_E, model_G = model_list
    optG, optE = optimizer

    model_G.train()
    model_E.train()
    train_loader, train_labeled_loader = train_loader
    # normalize = lambda x: ((x.float() / 255.) - .5) * 2.
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))
    sample_langevin_prior_z, sample_langevin_post_z, sample_p_0 = langevin_at_z(opt, device=device)
    # z_fixed = sample_p_0()
    x_fixed = next(iter(train_loader))[0].to(device)
    y_fixed = next(iter(train_loader))[1].to(device)
    prior_buffer, post_buffer = buffer

    # Future for combining with function update_theta
    for i, (x, y) in enumerate(train_loader, 0):
            # print(x.max(), x.min())
            x = x.to(device)
            batch_size = x.shape[0]
            y = y.to(device)
            nldata = train_labeled_loader.__next__()
            # x_lab, y_lab = nldata[0], nldata[1] 
            # x_lab, y_lab = x_lab.to(device), y_lab.to(device)
            
            # Initialize chains
            z_g_0, g_inds = sample_p_0(post_buffer, n=batch_size)
            z_e_0, e_inds = sample_p_0(prior_buffer, n=batch_size)
            # print(x.device, z_g_0.device, z_e_0.device)
            
            # Langevin posterior and prior
            z_g_k, samples, _, _= sample_langevin_post_z(replay_buffer_post=post_buffer, buffer_inds=g_inds, netE=model_E, z=Variable(z_g_0), x=x, args=opt, netG=model_G, y=y, verbose=i<5)
            z_e_k, _, _ = sample_langevin_prior_z(replay_buffer_prior=prior_buffer, buffer_inds=e_inds, netE=model_E, z=Variable(z_e_0), args=opt, y=y)

            # Learn generator
            optG.zero_grad()
            x_hat = model_G(z_g_k.detach())
            # print(x_hat.min(), x_hat.max(), x.max(), x.min())
            
            loss_g = criterion(x_hat, x) / batch_size
            # print(x_hat.mean(), x.mean())
            # print(loss_g.item())
            # exit(-1)
            loss_g.backward()
            # grad_norm_g = get_grad_norm(net.netG.parameters())
            # if args.g_is_grad_clamp:
            torch.nn.utils.clip_grad_norm(model_G.parameters(), opt.g_max_norm)
            optG.step()

            # Learn prior EBM
            optE.zero_grad()
            model_G.eval()
            en_neg = model_E(z_e_k.detach())[0].mean() # TODO(nijkamp): why mean() here and in Langevin sum() over energy? constant is absorbed into Adam adaptive lr
            en_pos = model_E(z_g_k.detach())[0].mean()
            # print(z_g_k.mean())
            loss_e = en_pos - en_neg
            K = opt.lc_K
            # print(K)
            if opt.joint:
                model_cls.eval()
                if opt.st == -1:
                    # Randomly sample the start point
                    st = random.randint(0, opt.g_l_steps - K)
                else:
                    # Sample from the given start point.
                    st = opt.st
                for sample_at_k in samples[st:st+K]:
                    z_k, z_k_minus_1, noise = sample_at_k
                    x_k = model_G(z_k)
                    x_k_minus_1 = model_G(z_k_minus_1.detach())
                    logit_k = model_cls(x_k)
                    logit_k_minus_1 = model_cls(x_k_minus_1)
                    # Only consider classification result
                    l_c_k = torch.nn.CrossEntropyLoss()(logit_k, y)
                    l_c_k_minus_1 = torch.nn.CrossEntropyLoss()(logit_k_minus_1, y)
                    mu = z_k - noise
                    sigma = opt.g_l_step_size * torch.ones_like(z_k)
                    nll = diag_normal_NLL(torch.flatten(z_k, 1), torch.flatten(mu, 1), 2*torch.flatten(sigma, 1).log()).mean(1)
                    l_b = ((l_c_k_minus_1 - l_c_k) * nll).mean()
                    loss_e += opt.g_l_steps / opt.lc_K * l_b

            
            loss_e.backward()
            if loss_e.abs().item() > 1e5:
                print(loss_e.item(), en_pos.item(), en_neg.item())
                raise ValueError('Not converge.')
            # grad_norm_e = get_grad_norm(net.netE.parameters())
            # if args.e_is_grad_clamp:
            #    torch.nn.utils.clip_grad_norm_(net.netE.parameters(), args.e_max_norm)
            optE.step()
            global_iter = epoch * len(train_loader) + i
            # Printout
            if i % opt.print_freq == 0:
                batch_size_fixed = x_fixed.shape[0]
                z_g_0p, g_inds0 = sample_p_0(replay_buffer=post_buffer,n=batch_size_fixed)
                z_e_0p, e_inds0 = sample_p_0(replay_buffer=prior_buffer, n=batch_size_fixed)
                z_g_kp, _, _= sample_langevin_post_z(replay_buffer_post=post_buffer, buffer_inds=g_inds0, netE=model_E, z=Variable(z_g_0p), x=x_fixed, args=opt, netG=model_G, y=None, verbose=False)
                z_e_kp, _ = sample_langevin_prior_z(replay_buffer_prior=prior_buffer,netE=model_E, z=Variable(z_e_0p), buffer_inds=e_inds0, args=opt, y=None)
                with torch.no_grad():
                    logger.log_value('l_g', loss_g, global_iter)
                    logger.log_value('l_e', loss_e, global_iter)
                    if opt.joint:
                        logger.log_value('l_b', l_b, global_iter)
                        logger.log_value('l_cls_k', l_c_k, global_iter)
                        logger.log_value('l_cls_k_minus_1', l_c_k_minus_1, global_iter)
                    logger.log_value('z_e_k', z_e_k.mean(), global_iter)
                    logger.log_value('z_g_k', z_g_k.mean(), global_iter)
                    
                    
                    x_0 = model_G(z_e_0p)
                    x_k = model_G(z_e_kp)
                    x_g_kp = model_G(z_g_kp)
                    x_0_kp = model_G(z_g_0)
                    plot('{}/x_k_{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_k)
                    plot('{}/x_0p_{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_0_kp)
                    plot('{}/x_kp_{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_g_kp)
                    # plot('{}/x_fixed_{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_fixed)
                    plot('{}/x_0_{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_0)
                                  
                    # if opt.plot_uncond:
                    #     y = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(input.device)
                    #     z_e_k = sample_langevin_prior_z(netE=model_E, z=Variable(z_e_0), args=opt, y=y)
                    #     # x_q, _ = sample_q(model, buffer, y=y_q)
                    #     # plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_q)
                    # if opt.plot_cond:  # generate class-conditional samples
                    #     y = torch.arange(0, opt.n_cls).to(input.device)
                        # print(y.shape)
                        # x_q_y, _ = sample_q(model, buffer, y=y)
                        # plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, i), x_q_y)

                    en_neg_2 = model_E(z_e_kp)[0].mean()
                    en_pos_2 = model_E(z_g_kp)[0].mean()
                    if opt.joint:
                        logit = model_cls(x_g_kp).argmax(1)
                        correct = (logit == y_fixed).sum()
                        pretrained_acc_moments = 'Pretrained accuracy: {:.2f} / {:.2f} = {:.2f} '.format(correct, batch_size_fixed, correct / batch_size_fixed)

                    prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_e_k.mean(), z_e_k.std(), z_e_k.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_g_k.mean(), z_g_k.std(), z_g_k.abs().max())
                    
                    str = '{:5d}/{:5d} {:5d}/{:5d} '.format(epoch, opt.epochs, i, len(train_loader)) + 'loss_g={:8.3f}, '.format(loss_g) +'loss_e={:8.3f}, '.format(loss_e) +'en_pos=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_pos, en_pos_2, en_pos_2-en_pos) +'en_neg=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_neg, en_neg_2, en_neg_2-en_neg) +'|z_g_0|={:6.2f}, '.format(z_g_0.view(batch_size, -1).norm(dim=1).mean()) + '|z_g_k|={:6.2f}, '.format(z_g_k.view(batch_size, -1).norm(dim=1).mean()) +'|z_e_0|={:6.2f}, '.format(z_e_0.view(batch_size, -1).norm(dim=1).mean()) +'|z_e_k|={:6.2f}, '.format(z_e_k.view(batch_size, -1).norm(dim=1).mean()) + 'z_e_disp={:6.2f}, '.format((z_e_k-z_e_0).view(batch_size, -1).norm(dim=1).mean()) +'z_g_disp={:6.2f}, '.format((z_g_k-z_g_0).view(batch_size, -1).norm(dim=1).mean()) + 'x_e_disp={:6.2f}, '.format((x_k-x_0).view(batch_size, -1).norm(dim=1).mean()) +'prior_moments={}, '.format(prior_moments) + 'posterior_moments={}, '.format(posterior_moments)
                    if opt.joint:
                        str += pretrained_acc_moments
                    print(str)

                    
    return loss_e


def train_vae(model_list, optimizer, opt, train_loader, logger, epoch, model_s=None, optimizer_s=None):
    model_t, model_vae = model_list
    model_vae.train()
    model_t.eval()
    joint = opt.joint_training['open']
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))
    train_loader, train_labeled_loader = train_loader

    x_fixed, y_fixed = next(train_labeled_loader)
    x_fixed = x_fixed.to(opt.device)
    y_fixed = y_fixed.to(opt.device)
    train_imgs = len(train_loader.dataset)
    for (idx, batch) in enumerate(train_loader):
        real_img, labels = batch
        curr_device = opt.device
        real_img = real_img.to(curr_device)
        labels = labels.to(curr_device)
        optimizer.zero_grad()

        results = model_vae(real_img, labels = labels)
        total_loss = 0.
        train_loss = model_vae.loss_function(*results, M_N = opt.batch_size / train_imgs, optimizer_idx = 0, batch_idx = idx)
        total_loss += train_loss['loss']
        if opt.joint:
            sampled_imgs = model_vae.sample(num_samples=opt.batch_size, current_device=curr_device, labels=labels, train=False)
            logit_t = model_t(sampled_imgs)
            l_cls = torch.nn.CrossEntropyLoss()(logit_t, labels)
            logit_t_pos = model_t(real_img)
            p_pos = F.log_softmax(logit_t_pos, 1)
            p_neg = F.softmax(logit_t, 1)
            kl_loss = F.kl_div(p_pos, p_neg)
            acc = (logit_t.argmax(1) == labels).sum() / opt.batch_size
            if joint:
                assert model_s is not None
                model_s.eval()
                logit_s = model_s(sampled_imgs)
                p_stu = F.softmax(logit_s / 4., 1)
                p_tea = F.log_softmax(logit_t / 4., 1)
                kd_div = F.kl_div(p_tea, p_stu)
                total_loss -= 0.1 * kd_div

            # tv_loss = TVLoss()(sampled_imgs).mean()
            total_loss += opt.lamda_cls * l_cls + opt.lamda_kl * kl_loss

        total_loss.backward()
        optimizer.step()
        if joint:
            model_s.train()
            # model_vae.eval()
            model_s.zero_grad()
            sampled_imgs = model_vae.sample(num_samples=opt.batch_size, current_device=curr_device, labels=labels, train=False)
            logit_t = model_t(sampled_imgs)
            logit_s = model_s(sampled_imgs)
            stu_cls = torch.nn.CrossEntropyLoss()(logit_s, labels)
            p_stu = F.softmax(logit_s / 4, 1)
            p_tea = F.log_softmax(logit_t / 4, 1)
            kd_div = F.kl_div(p_tea, p_stu)
            kd_loss = opt.joint_training['alpha'] * stu_cls + opt.joint_training['beta'] * kd_div
            logit_s_real = model_s(x_fixed)
            stu_acc = (logit_s_real.argmax(1) == y_fixed).sum() / opt.batch_size
            kd_loss.backward()
            optimizer_s.step()
            
            
        global_iter = len(train_loader) * epoch + idx
        if idx % opt.print_freq == 0:
            logger.log_value('total_loss', train_loss['loss'], global_iter)
            logger.log_value('Rec_loss', train_loss['Reconstruction_Loss'], global_iter)
            logger.log_value('KL_Loss', train_loss['KLD'], global_iter)

            print_str = 'Epoch: {} / {}, Data: {} / {}, total_loss = {:.4f}\t Reconstruction_loss = {:.4f}\t KL_loss = {:.4f}\t'.format(epoch, opt.epochs, idx, len(train_loader), train_loss['loss'], train_loss['Reconstruction_Loss'], train_loss['KLD'])
            if opt.joint:
                logger.log_value('l_cls', l_cls, global_iter)
                logger.log_value('l_kl', kl_loss)
                logger.log_value('acc', acc)
                print_str += 'Classification Loss = {:.4f}\t +/- divergence = {:.4f}\t Teacher Accuracy = {:.4f}\t'.format(l_cls, kl_loss, acc)
                if joint:
                    print_str += 'Student Accuracy = {:.4f}\t'.format(stu_acc)
                    logger.log_value('stu acc', stu_acc)
            print(print_str)
            if opt.plot_uncond:
                y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(curr_device)
                x_q = model_vae.sample(opt.batch_size, curr_device, labels = y_q)
                plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q)
            if opt.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, opt.n_cls).to(curr_device)
                # print(y.shape)
                x_q_y = model_vae.sample(opt.n_cls, curr_device, labels = y, train=False)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)

            x_rec = model_vae.generate(x_fixed, labels=y_fixed)
            plot('{}/x_rec_fixed{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_rec)

    return train_loss['loss']

def sample_vae(model, opt, n_samples=30000):
    num_img_per_class = n_samples // opt.n_cls
    y = torch.arange(0, opt.n_cls).to(opt.device)
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))
    if not os.path.exists('{}/vae_samples/'.format(opt.save_folder)):
        os.mkdir('{}/vae_samples/'.format(opt.save_folder))
    for j in tqdm.tqdm(range(num_img_per_class)):
        temp_x = model.sample(opt.n_cls, opt.device, labels=y, train=False)
        for i, x in enumerate(temp_x):
            plot('{}/vae_samples/samples_label_{}_{}.png'.format(opt.save_folder, i, j), x.unsqueeze(0))

def train_coopnet(model_list, optimizer_list, opt, train_loader, logger, epoch, buffer):
    model_s, model_t, model_vae = model_list
    model_vae.eval()
    model_t.eval()
    model_s.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    fpxs = AverageMeter()
    fqxs = AverageMeter()
    fpxys = AverageMeter()
    fqxys = AverageMeter()
    accs = AverageMeter()
    # joint = opt.joint_training['open']
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))
    train_loader, train_labeled_loader = train_loader

    x_fixed, y_fixed = next(train_labeled_loader)
    x_fixed = x_fixed.to(opt.device)
    y_fixed = y_fixed.to(opt.device)
    train_imgs = len(train_loader.dataset)
    optimizer_theta, optimizer_alpha = optimizer_list
    end = time.time()
    sample_q = langevin_at_x(opt, device=opt.device)
    for (idx, batch) in enumerate(train_loader):
        real_img, labels = batch
        curr_device = opt.device
        real_img = real_img.to(curr_device)
        labels = labels.to(curr_device)

        # update slow sampler(ebm)
        # optimizer_theta.zero_grad()
        if idx <= opt.warmup_iters:
            lr = opt.learning_rate_ebm * idx / float(opt.warmup_iters)
            for param_group in optimizer_theta.param_groups:
                param_group['lr'] = lr
        data_time.update(time.time() - end)
        nldata = train_labeled_loader.__next__()
        x_lab, y_lab = nldata[0], nldata[1] 
        x_lab, y_lab = x_lab.to(curr_device), y_lab.to(curr_device)

        loss_ebm, cache_p_x, cache_p_y, logit, ls = update_theta(opt, buffer, model_list, real_img, x_lab, y_lab, y_p=labels, mode='coopnet')   
        optimizer_theta.zero_grad()
        losses.update(loss_ebm, real_img.size(0))
        model_s.zero_grad()
        loss_ebm.backward()
        optimizer_theta.step()
        if opt.lmda_p_x > 0:
            fpx, fqx = cache_p_x
            fpxs.update(fpx, real_img.size(0))
            fqxs.update(fqx, real_img.size(0))
        if opt.lmda_p_x_y > 0:
            fpxy, fqxy = cache_p_y
            fpxys.update(fpxy, real_img.size(0))
            fqxys.update(fqxy, real_img.size(0))

        l_p_x, l_p_x_y, l_cls = ls

        acc = torch.sum(torch.argmax(logit, 1) == y_lab).item() / real_img.size(0)
        accs.update(acc, real_img.size(0))
        global_iter = epoch * len(train_loader) + idx
        if global_iter % opt.print_freq == 0:
            logger.log_value('l_p_x', l_p_x, global_iter)
            logger.log_value('l_p_x_y', l_p_x_y, global_iter)
            logger.log_value('l_cls', l_cls, global_iter)
            logger.log_value('accuracy', acc, global_iter)
        # update fast initializer(vae)
        optimizer_alpha.zero_grad()
        model_vae.train()
        model_s.eval()

        results = model_vae(real_img, labels = labels)
        total_loss = 0.
        train_loss = model_vae.loss_function(*results, M_N = opt.batch_size / train_imgs, optimizer_idx = 0, batch_idx = idx)
        total_loss += train_loss['loss']

        total_loss.backward()
        optimizer_alpha.step()
            
            
        global_iter = len(train_loader) * epoch + idx
        if idx % opt.print_freq == 0:
            model_vae.eval()
            logger.log_value('total_loss', train_loss['loss'], global_iter)
            logger.log_value('Rec_loss', train_loss['Reconstruction_Loss'], global_iter)
            logger.log_value('KL_Loss', train_loss['KLD'], global_iter)

            print_str = 'Epoch: {} / {}, Data: {} / {}, total_loss = {:.4f}\t Reconstruction_loss = {:.4f}\t KL_loss = {:.4f}\t'.format(epoch, opt.epochs, idx, len(train_loader), train_loss['loss'], train_loss['Reconstruction_Loss'], train_loss['KLD'])
            print_str += 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\n'.format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses)
            if opt.lmda_p_x > 0:
                # print(fqxs.count)
                print_str += 'p(x) f(x+) {fpx.val:.4f} ({fpx.avg:.4f})\t'.format(fpx=fpxs)
                print_str += 'f(x-) {fqx.val:.4f} ({fqx.avg:.4f})\n'.format(fqx=fqxs)
            if opt.lmda_p_x_y > 0:
                print_str += 'p(x) f(x+) {fpx.val:.4f} ({fpx.avg:.4f})\t'.format(fpx=fpxys)
                print_str += 'f(x-) {fqxy.val:.4f} ({fqxy.avg:.4f})\n'.format(fqxy=fqxys)
            print_str += 'Acc: {accs.val:.4f} ({accs.avg:.4f})\n'.format(accs=accs)
            
            # print(string)
            print(print_str)
            if opt.plot_uncond:
                y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(curr_device)
                x_q = model_vae.sample(opt.batch_size, curr_device, labels = y_q, train=False)
                x_q = sample_q(model_s, buffer, y=y_q, init_x=x_q)
                plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q)
            if opt.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, opt.n_cls).to(curr_device)
                # print(y.shape)
                x_q_y = model_vae.sample(opt.n_cls, curr_device, labels = y, train=False)
                x_q_y, _ = sample_q(model_s, buffer, y=y, init_x=x_q_y)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)

            # x_rec = model_vae.generate(x_fixed, labels=y_fixed)
            # plot('{}/x_rec_fixed{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_rec)

    return train_loss['loss']


def validate_G(model, replay_buffer, opt, eval_loader=None):
    """validation for generation stage. Also returns inception score(IS), Fischer Inception Distance(FID). Designed for further inference."""
    replay_buffer = cond_samples(model, replay_buffer, device=opt.device, opt=opt, fresh=opt.fresh)
    if not opt.save_grid:
        test_folder = opt.save_dir
        dataset = CIFAR100Gen(
            root=test_folder,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        )
        mean, var = inception_score(dataset, resize=True, device=opt.device, splits=5)
        print(mean, var)
        f = open(os.path.join(opt.save_folder, 'is.txt'), 'w')
        f.write('{:.4f} +- {:.4f}'.format(mean, var))
        f.close()
    
    return replay_buffer

def generation_stage(model_lists, replay_buffer, opt, logger=None):
    import numpy as np
    import matplotlib.pyplot as plt
    model, s, t = model_lists

    device = opt.device
    sample_q = langevin_at_x(opts=opt, device=device)
    # replay_buffer = ckpt_energy['replay_buffer'].cpu()
    #replay_buffer = init_random((opt.buffer_size, 3, 32, 32))
    # print(replay_buffer.shape)
    y = torch.arange(0, opt.n_cls).to(device)

    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))

    freshh_epochs = opt.epochs
    for i in range(freshh_epochs):
        replay_buffer = replay_buffer.cpu()
        samples, _ = sample_q(model, replay_buffer, y=y, other_models=(s, t), logger=logger, global_epoch=i)
        if i % 200 == 0:
            if os.path.exists(opt.save_dir):
                # os.rmdir(opt.save_dir)\
                shutil.rmtree(opt.save_dir)
                os.mkdir(opt.save_dir)
            print('Epoch {} / {} ****'.format(i, freshh_epochs))
            # buffer_size = len(replay_buffer)
            # inds = torch.randint(0, buffer_size, (25,))
            plot(os.path.join(opt.save_folder, "res_epoch_{}.jpg".format(i)), samples)
            # img = np.transpose(imgs, (1,2,0))
            replay_buffer = cond_samples(model, replay_buffer, device, opt, use_buffer=True)
            test_folder = opt.save_dir
            dataset = CIFAR100Gen(
                root=test_folder,
                transform=T.Compose([
                    T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            )
            mean, var = inception_score(dataset, device, resize=True, splits=3, batch_size=8)
            print('IS: {} +- {}'.format(mean, var))
            logger.log_value('Inception Score', mean, i)
            # ckpt_dict = {
            #     "replay_buffer": replay_buffer
            # }
            # torch.save(ckpt_dict, os.path.join(opt.save_ckpt, 'res_buffer_{}.pts'.format(i)))

    return replay_buffer
