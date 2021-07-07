from __future__ import print_function, division

import sys
import time
import torch
import torch.optim as optim
from torch.autograd import grad
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from math import sqrt
sys.path.append('..')
from datasets.cifar100 import CIFAR100Gen
 
from .util import AverageMeter, accuracy, set_require_grad, print_trainable_paras, inception_score
from .util_gen import get_replay_buffer, update_theta, ssm_sample, getDirichl
from .util_gen import get_sample_q, cond_samples, estimate_h, diag_normal_NLL, diag_standard_normal_NLL


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        if opt.dataset == 'cifar100':
            input, target = data
        else:
            input, target, idx = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss, output = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def train_nce_G(epoch, train_loader, model_list, optimizer_list, opt, logger):
    model_vae, model = model_list
    opt_alpha, opt_theta = optimizer_list
    model.train()
    model_vae.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_theta = AverageMeter()
    vae_losses = AverageMeter()
    recon_losses = AverageMeter()
    fpxys = AverageMeter()
    fqxys = AverageMeter()
    accs = AverageMeter()
    # noise = torch.randn(128, 3, 32, 32)
    train_loader, train_labeled_loader = train_loader
    # criterion is tv loss
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=int(sqrt(x.size(0))))

    end = time.time()
    sample_q, _ = get_sample_q(opt)
    correct = 0
    total_length = 0
    d = 0
    N_train_sets = len(train_loader.dataset)
    next_mode = 'ebm'
    for idx, (input, target) in enumerate(train_loader):
        # if idx <= opt.warmup_iters:
        #     lr = opt.learning_rate * idx / float(opt.warmup_iters)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        data_time.update(time.time() - end)
        global_iter = epoch * len(train_loader) + idx
        x_lab, y_lab = train_labeled_loader.__next__()
        x_lab, y_lab = x_lab.cuda(), y_lab.cuda()
        # print(next_mode)

        input = input.float()
        # noise = torch.randn(input.shape[0], 100)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        if next_mode == 'ebm':
        # Update theta, conditional ebm
            # raise ValueError('Invalid next mode.')
            set_require_grad(model, True)
            set_require_grad(model_vae, False)
            opt_theta.zero_grad()
            # with torch.no_grad():
            # with torch.no_grad():
            #     result = 
            # out_result = model_vae(x_lab, labels=y_lab)
            loss_theta, log_probs_ebm_pos, log_probs_ebm_neg, _, next_mode = estimate_h(x_lab, y_lab, model_vae, model, mode='ebm', batch_size=opt.batch_size)
            if loss_theta.abs().item() > 1e8:
                raise ValueError('Not Converged.')
            fpxys.update(log_probs_ebm_pos.mean().item(), x_lab.size(0))
            fqxys.update(log_probs_ebm_neg.mean().item(), x_lab.size(0))
            # loss_k_theta = -(log_probs_ebm_pos - log_probs_ebm_neg).mean()
            # loss_theta += loss_k_theta

            losses_theta.update(loss_theta.item(), x_lab.size(0))
            loss_theta = -(loss_theta)
            # print(model.c)
            # print(loss_theta, mode)
            # one = torch.FloatTensor([1]).to(x_lab.device)
            # mone = one * -1
            loss_theta.backward()
            opt_theta.step()
            next_mode='vae'

            if idx % opt.print_freq == 0:
                string = 'Epoch: [{0}][{1}/{2}]\tStep: [EBM]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses_theta)
                if opt.energy != 'ssm':
                    string += 'p(x, y) f(x+) {fpxy.val:.4f} ({fpxy.avg:.4f})\t'.format(fpxy=fpxys)
                string += 'f(x-) {fqxy.val:.4f} ({fqxy.avg:.4f})\n'.format(fqxy=fqxys)
                print(string)
                sys.stdout.flush()
            
            if d == 1:
                next_mode = 'vae'
                d = 0
            else:
                d += 1
                next_mode = 'ebm'

        # Logger for update of ebm
        # Including: tensorboard logger and print information

        # Update alpha, conditional vae
        if next_mode == 'vae':
            set_require_grad(model, False)
            set_require_grad(model_vae, True)
            opt_alpha.zero_grad()
            # out_result = model_vae(x_lab, labels=y_lab)
            loss_V, log_probs_ebm_pos, log_probs_ebm_neg, out_result, next_mode = estimate_h(x_lab, y_lab, model_vae, model, mode='vae', batch_size=opt.batch_size)
            x_rec, x_ori, mu, logvar, rez = out_result
            train_loss = model_vae.loss_function(*out_result, M_N=opt.batch_size/N_train_sets)
            # print()
            loss_alpha = train_loss['loss']
            # loss_rec = torch.nn.functional.mse_loss(x_rec, x_ori)
            # logpz = diag_standard_normal_NLL(rez).mean()
            # logqzx = diag_normal_NLL(rez, mu, logvar).mean()
            # loss_kl = -logqzx + logpz
            # ratio = opt.batch_size / N_train_sets
            # loss_alpha =  loss_rec + ratio * loss_kl
            # print(loss_alpha)
            # print(train_loss)
            if loss_alpha.mean().item() > 1e8:
                raise ValueError('Not Converged.')
            
            # log_probs_ebm_neg = model(x=x_neg, y=y_lab)[0]
            # x_neg, _, mu, log_var, _ = out_result
            # # Guassian distribution of q_{\alpha}(x)
            # dist = torch.distributions.Normal(mu, log_var.exp() / 2)
            # log_probs_noise_neg = dist.log_prob(x_neg)
            # log_probs_noise_pos = dist.log_prob(x_lab)
            # fpxys.update(log_probs_ebm_pos.mean().item(), input.size(0))
            # fqxys.update(log_probs_ebm_neg.mean().item(), input.size(0))
            recon_losses.update(train_loss['Reconstruction_Loss'].item(), input.size(0))
            vae_losses.update(train_loss['KLD'].item(), input.size(0))
            # recon_losses.update(loss_rec.item(), input.size(0))
            # vae_losses.update(loss_kl.item(), input.size(0))
            # torch.nn.utils.clip_grad_norm_(model_vae.parameters(), 0.01)
            
            # print(-train_loss['KLD'].item())
            # h_ebm = torch.sigmoid(log_probs_ebm_pos - log_probs_ebm_neg)
            # h_noise = torch.sigmoid(log_probs_noise_pos-log_probs_noise_neg)
            # loss_V = h_ebm.log().mean() + (1 - h_noise).log().mean()
            # loss_V = -loss_V
            # print(loss_alpha)
            loss_alpha = loss_alpha + opt.lmda_v*loss_V
            # print(loss_alpha)
            loss_alpha.backward()
            if log_probs_ebm_neg.mean().abs().item() > 10000:
                print('Diverge detected. The logs:')
                print('1. The log probability of noise sampled negative sample by ebm model. Also called negative energy value.')
                print(log_probs_ebm_neg.mean().item(), log_probs_ebm_pos.mean().item())
                
                print('2. Norm of energy model parameter and grads.')
                for p in model.named_parameters():
                    print('Name: {}'.format(p[0]))
                    print(p[1].mean())
                    # print(type(p[1].grad))
                    print(p[1].grad.mean())
                print('3. Norm of vae Encoder and decoder parameter and grads.')
                for p in model_vae.named_parameters():
                    print('Name: {}'.format(p[0]))
                    print(p[1].mean())
                    print(p[1].grad.mean())
                raise ValueError('Not converged. Please check the possible divergence reason above to check: EBM, VAE or logistic loss function.')
            losses.update(loss_alpha.item(), input.size(0))
            opt_alpha.step()
            batch_time.update(time.time() - end)
            end = time.time()
            next_mode = 'ebm'

            if idx % opt.print_freq == 0:
                string = 'Epoch: [{0}][{1}/{2}]\tStep: [VAE]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\tRec Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\tKLD {kld.val:.4f} ({kld.avg:.4f})'.format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, recon_loss=recon_losses, kld=vae_losses)

                # string += 'Acc: {accs.val:.4f} ({accs.avg:.4f})\n'.format(accs=accs)
                print(string)
                sys.stdout.flush()
        
        # next_mode = 'vae'
        # print(next_mode)
        if idx % opt.print_freq == 0:
            if opt.plot_uncond:
                y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(input.device)
                x_q = model_vae.sample(num_samples=opt.batch_size, current_device=input.device, labels=y_q, train=False)
                plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q)
            if opt.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, opt.n_cls).to(input.device)
                # print(y.shape)
                x_q_y = model_vae.sample(num_samples=opt.n_cls, current_device=input.device, labels=y, train=False)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)
        

    return losses.avg




def train_generator(epoch, train_loader, model_list, criterion, optimizer, opt, buffer, logger):
    '''One epoch for training generator with teacher'''
    model_t, model = model_list
    model.train()
    model_t.eval()
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
    sample_q, _ = get_sample_q(opt)
    correct = 0
    total_length = 0
    for idx, (input, target) in enumerate(train_loader):
        if idx <= opt.warmup_iters:
            lr = opt.learning_rate * idx / float(opt.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        x_lab, y_lab = train_labeled_loader.__next__()
        x_lab, y_lab = x_lab.cuda(), y_lab.cuda()

        input = input.float()
        # noise = torch.randn(input.shape[0], 100)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            # noise = noise.cuda()

        # ===================forward=====================
        # input_fake, [mu, logvar] = G(noise, target, return_feat=True)
        # output = model(input_fake)
        loss_ebm = 0
        optimizer.zero_grad()
        model.zero_grad()
        if opt.energy == 'mcmc':
            loss_ebm, cache_p_x, cache_p_y, logit, ls = update_theta(opt, buffer, model, input, x_lab, y_lab, model_t=model_t)    
        elif opt.energy == 'ssm':
            loss_ebm, score_x, score_xy = ssm_sample(opt, buffer, model, input, x_lab, y_lab)
        else:
            raise NotImplementedError('Not implemented.')
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
                x_q = sample_q(model, buffer, y=y_q)
                plot('{}/x_q_{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q)
            if opt.plot_cond:  # generate class-conditional samples
                y = torch.arange(0, opt.n_cls).to(input.device)
                # print(y.shape)
                x_q_y = sample_q(model, buffer, y=y)
                plot('{}/x_q_y{}_{:>06d}.png'.format(opt.save_dir, epoch, idx), x_q_y)

    return losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.module.train()
    # set teacher as eval()
    module_list[-1].module.eval()

    if opt.distill == 'abound':
        module_list[1].module.eval()
    elif opt.distill == 'factor':
        module_list[2].module.eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    # train_loader.sampler.set_epoch(epoch)
    # with torch.autograd.set_detect_anomaly(True):
    for idx, data in enumerate(train_loader):
        
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        # print(input.shape)
        data_time.update(time.time() - end)
        bs = input.shape[0]
        # input_fake = model_G(noise)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
        
        
        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
            # input = fake_input
        # input_fake = model_G(noise, target)
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        elif opt.distill == 'energy':
            fs = feat_s[-1]
            ft = feat_t[-1]
            loss_ssm = criterion_kd(input, ft)
            loss_kd = 0
        elif opt.distill == 'ebkd':
            fs = logit_s
            ft = logit_t
            loss_kd = criterion_kd(fs, ft)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        # print(loss_cls, loss_div)
        losses.update((loss).item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer[-1].zero_grad()
        # loss_cls.backward(retain_graph=True)
        loss.backward()
        optimizer[-1].step()
        # if opt.

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

            # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg
    

def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            noise = torch.randn(input.shape[0], 100)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                noise = noise.cuda()

            # compute output
            output = model(input)

            # print(torch.argmax(output, 1), target)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def validate_G(model, replay_buffer, opt, eval_loader=None):
    """validation for generation stage. Also returns inception score(IS), Fischer Inception Distance(FID). Designed for further inference."""
    cond_samples(model, replay_buffer, opt)
    if not opt.save_grid:
        test_folder = opt.save_dir
        dataset = CIFAR100Gen(
            root=test_folder,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        mean, var = inception_score(dataset, resize=True)
        print(mean, var)
    if opt.jem_cls:
        model.eval()
