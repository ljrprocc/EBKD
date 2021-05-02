from __future__ import print_function, division

import sys
import time
import torch
import torch.optim as optim
from torch.autograd import grad
import torch.nn.functional as F
 
from .util import AverageMeter, accuracy, set_require_grad, sample_buffer, sample_mcmc, clip_grad, SGLD


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

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


def train_generator(epoch, train_loader, module_list, criterion, optimizer, opt, buffer):
    '''One epoch for training generator with teacher'''
    for module in module_list:
        module.train()
    # module_list[0].eval()
    # model = module_list[-1]
    model_G = module_list[0]
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    noise = torch.randn(128, 3, 32, 32)

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        # noise = torch.randn(input.shape[0], 100)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            noise = noise.cuda()

        # ===================forward=====================
        # input_fake, [mu, logvar] = G(noise, target, return_feat=True)
        # output = model(input_fake)
        loss_ebm = 0
        if opt.energy == 'mcmc':
            neg_img, neg_id = sample_buffer(buffer, input.shape[0], num_classes=model_G.n_cls)
            set_require_grad(model_G, False)
            model_G.eval()
            x_k = torch.autograd.Variable(neg_img, requires_grad=True)
            neg_img = SGLD(model_G, neg_img, neg_id, noise)
            buffer.buffer[neg_id] = neg_img.cpu()
            optimizer.zero_grad()
            pos_out = model_G(input, target)
            neg_out = model_G(neg_img, neg_id)

            loss_ebm = 3 * (pos_out ** 2 + neg_out ** 2)
            # print(loss_ebm.mean(), (pos_out - neg_out).mean())
            loss_ebm = loss_ebm + (pos_out - neg_out)
            loss_ebm = loss_ebm.mean()

            # print(torch.mean(neg_img, 0).norm(), torch.mean(input, 0).norm())
            loss_ebm.backward()

            clip_grad(model_G.parameters(), optimizer)
            optimizer.step()

            # buffer.push(neg_img, neg_id)
        # buffer.push(input, target)
        # loss, output = criterion(output, target)
        # loss_kl = torch.mean(-0.5*torch.sum(1+logvar-mu**2-logvar.exp(), dim=1), dim=0)
        # print(loss_kl)
        # loss = loss + 20 * loss_kl

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_ebm, input.size(0))
        # top1.update(acc1[0], input.size(0))
        # top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #   .format(top1=top1, top5=top5))

    return losses.avg




def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

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
        if opt.mode != 'energy':
            feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        else:
            feat_s, logit_s = model_s.classify(input, is_feat=True, preact=preact)
        
        with torch.no_grad():
            # if opt.mode != 'energy':
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
        if top1.val > 98:
            print(top1.val)
            print('Too high Top-1 accuracy!')
            print(torch.softmax(logit_s, 1))
            print(target)
            exit(-1)
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


def train_distill_G(epoch, train_loader, module_list, criterion_list, optimizer, opt, buffer):
    """One epoch distillation, Data-Free mode"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_ebms = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    optimizer_ebm = optimizer[-2]
    noise = torch.randn(128, 3, 32, 32)
    

    # with torch.autograd.set_detect_anomaly(True):
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)
        bs = input.shape[0]

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            noise = noise.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        preact = False
        if opt.distill in ['abound']:
            preact = True

    set_require_grad(model_t, False)
    if opt.energy == 'mcmc':
        neg_img_raw, neg_id = sample_buffer(buffer, y=target, batch_size=input.shape[0], num_classes=model_G.n_cls)
        set_require_grad(model_s, False)
        model_s.eval()
        # print(neg_img_raw.shape)
        x_k = torch.autograd.Variable(neg_img_raw, requires_grad=True)
        neg_img = SGLD(model_G, neg_img=x_k, y=neg_id)
        neg_img.clamp_(-1, 1)
        # print(neg_img)
        # print(neg_id)
        buffer.push(neg_img, neg_id)
        optimizer_ebm.zero_grad()
        pos_out_cls = model_s.classify(input)
        # print(pos_out_cls)
        # loss_clf.backward()
        # loss_ebm = 0.
        neg_out = model_s(neg_img, neg_id).mean()
        pos_out = model_s(input, target).mean()
        # print(pos_out, neg_out)
        loss_ebm = -(pos_out - neg_out) + (pos_out ** 2 + neg_out ** 2)
        loss_ebm = loss_ebm.mean()

        # print(loss_clf.mean())
        lossa = loss_ebm + loss_clf  
        lossa.backward()
        optimizer_ebm.step()               
    elif opt.energy == 'ssm':
        pass
    else:
        raise NotImplementedError('Son of a bitch.')
    set_require_grad(model_s, True)
    
    neg_img, neg_id = sample_buffer(buffer, y=target, batch_size=input.shape[0], p=1, num_classes=model_G.n_cls)
    
    feat_s, logit_s = model_s(neg_img, is_feat=True, preact=preact)
    with torch.no_grad():
        feat_t, logit_t = model_t(neg_img, is_feat=True, preact=preact)
        feat_t = [f.detach() for f in feat_t]
    loss_cls = criterion_cls(logit_s, neg_id)
    loss_div = criterion_div(logit_s, logit_t)
    # print(logit_s.shape)
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
    # print(loss_cls, loss_div)
    acc1, acc5 = accuracy(logit_s, neg_id, topk=(1, 5))
    if (top1.val == 0 and top5.val >= 40):
        # print(top1.val)
        print('Too high Top-1 accuracy!')
        print(torch.max(torch.softmax(logit_s, 1), 1))
        print(logit_s)
        exit(-1)
    losses.update((loss).item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))
    loss_ebms.update((lossa).item(), input.size(0))


    # ===================backward=====================
    optimizer[-1].zero_grad()
    # loss_cls.backward(retain_graph=True)
    loss.backward()
    optimizer[-1].step()
    # print((model_G(input, target) - feat_real).mean())

    # ===================meters=====================
    batch_time.update(time.time() - end)
    end = time.time()
    # exit(-1)
        # print info
    if idx % opt.print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'EBM Loss {loss_ebm.val:.4f} ({loss_ebm.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_ebm=loss_ebms, top1=top1, top5=top5))
        sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
   

    return top1.avg, losses.avg



def validate(val_loader, model, criterion, opt, teacher_mode=False):
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
            if teacher_mode or opt.mode != 'energy':
                output = model(input)
            else:
                output = model.classify(input)

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

def validate_G(val_loader, model_list, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    for model in model_list:
        model.eval()
    g = model_list[0]
    t = model_list[1]
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
            input_noise = g(noise, target)
            output = t(input_noise)
            # print(criterion)
            loss, _ = criterion(output, target)
            # print(len(criterion.get_logit(output)))
            # output, _ = criterion.get_logit(output)

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
                       idx, len(val_loader), batch_time=batch_time, loss=losses, loss_ebm=loss_ebm,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg