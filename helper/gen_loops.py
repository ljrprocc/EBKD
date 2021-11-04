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
from math import sqrt
sys.path.append('..')
from datasets.cifar100 import CIFAR100Gen
 
from .util import AverageMeter, accuracy, set_require_grad, print_trainable_paras, inception_score, TVLoss
from .util_gen import update_theta, getDirichl 
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
    sample_q = langevin_at_x(opt)
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
        l_p_x, l_p_x_y, l_cls, l_c, l_c_k_minus_1, l2_k, l_cls_k, l_e_k, l2_k_1, l_cls_k_1, l_e_k_1 = ls
        acc = torch.sum(torch.argmax(logit, 1) == y_lab).item() / input.size(0)
        accs.update(acc, input.size(0))
        global_iter = epoch * len(train_loader) + idx
        if global_iter % opt.print_freq == 0:
            logger.log_value('l_p_x', l_p_x, global_iter)
            logger.log_value('l_p_x_y', l_p_x_y, global_iter)
            logger.log_value('l_cls', l_cls, global_iter)
            logger.log_value('l_image_c_k', l_c, global_iter)
            logger.log_value('l_image_c_k_1', l_c_k_minus_1, global_iter)
            logger.log_value('l_2_k', l2_k, global_iter)
            logger.log_value('l_cls_k', l_cls_k, global_iter)
            logger.log_value('l_e_k', l_e_k, global_iter)
            logger.log_value('l_2_k_1', l2_k_1, global_iter)
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
    sample_q = langevin_at_x(opt)
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
                y = torch.arange(0, opt.n_cls).to(input.device)
                # print(y.shape)
                x_q_y, _ = sample_q(model, buffer, y=y)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)

    return losses.avg

def train_z_G(epoch, buffer, train_loader, model_list, criterion, optimizer, opt, logger, local_rank=None, device=None):
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
    prior_buffer, post_buffer = buffer

    # Future for combining with function update_theta
    for i, (x, y) in enumerate(train_loader, 0):
            # print(x.max(), x.min())
            x = x.to(device)
            batch_size = x.shape[0]
            y = y.to(device)
            
            
            # Initialize chains
            z_g_0, g_inds = sample_p_0(post_buffer, n=batch_size)
            z_e_0, e_inds = sample_p_0(prior_buffer, n=batch_size)
            # print(x.device, z_g_0.device, z_e_0.device)
            
            # Langevin posterior and prior
            z_g_k, _, _, _= sample_langevin_post_z(replay_buffer_post=post_buffer, buffer_inds=g_inds, netE=model_E, z=Variable(z_g_0), x=x, args=opt, netG=model_G, y=y, verbose=i<10)
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
            en_neg = model_E(z_e_k.detach())[0].mean() # TODO(nijkamp): why mean() here and in Langevin sum() over energy? constant is absorbed into Adam adaptive lr
            en_pos = model_E(z_g_k.detach())[0].mean()
            # print(z_g_k.mean())
            loss_e = en_pos - en_neg
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

                    prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_e_k.mean(), z_e_k.std(), z_e_k.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_g_k.mean(), z_g_k.std(), z_g_k.abs().max())
                    str = '{:5d}/{:5d} {:5d}/{:5d} '.format(epoch, opt.epochs, i, len(train_loader)) + 'loss_g={:8.3f}, '.format(loss_g) +'loss_e={:8.3f}, '.format(loss_e) +'en_pos=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_pos, en_pos_2, en_pos_2-en_pos) +'en_neg=[{:9.4f}, {:9.4f}, {:9.4f}], '.format(en_neg, en_neg_2, en_neg_2-en_neg) +'|z_g_0|={:6.2f}, '.format(z_g_0.view(batch_size, -1).norm(dim=1).mean()) + '|z_g_k|={:6.2f}, '.format(z_g_k.view(batch_size, -1).norm(dim=1).mean()) +'|z_e_0|={:6.2f}, '.format(z_e_0.view(batch_size, -1).norm(dim=1).mean()) +'|z_e_k|={:6.2f}, '.format(z_e_k.view(batch_size, -1).norm(dim=1).mean()) + 'z_e_disp={:6.2f}, '.format((z_e_k-z_e_0).view(batch_size, -1).norm(dim=1).mean()) +'z_g_disp={:6.2f}, '.format((z_g_k-z_g_0).view(batch_size, -1).norm(dim=1).mean()) + 'x_e_disp={:6.2f}, '.format((x_k-x_0).view(batch_size, -1).norm(dim=1).mean()) +'prior_moments={}, '.format(prior_moments) + 'posterior_moments={}, '.format(posterior_moments)
                    print(str)

                    
    return loss_e
                    # logger.info()


    


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