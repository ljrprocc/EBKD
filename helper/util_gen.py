from __future__ import print_function

import torch
import os
import tqdm
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions.dirichlet import Dirichlet
import torchvision.utils as vutils
from torchvision import transforms
from helper.sampling import freshh, langevin_at_x
# from util import set_require_grad
import cv2

def add_dict(c, d):
    # Register new key-value pairs
    for k, v in d.items():
        c.__dict__[k] = v


def diag_normal_NLL(z, mu, log_var):
    '''
    Directly calculate the log probability without distribution sampling. It can modeling the sample process q_{\phi}(z | x)
    Input: [b, nz]
    '''
    nll = 0.5 * log_var + 0.5 * ((z - mu) * (z - mu) / (1e-6 + log_var.exp()))
    return nll.squeeze()

def diag_standard_normal_NLL(z):
    '''
    Sample from q_{\alpha}(z) ~ N(0, I)
    '''
    nll = 0.5 * (z * z)
    return nll.squeeze()

def getDirichl(net_path, scale=1, sim_scale=1, device=None):
    sim_matrix = create_similarity(net_path, scale=sim_scale, device=device)
    # X = torch.zeros(bs, num_classes).to(sim_matrix.device)
    c_n = (sim_matrix - sim_matrix.min(1)[0]) / (sim_matrix.max(1)[0] - sim_matrix.min(1)[0])
    c_n = c_n * scale + 0.000001
    diri = Dirichlet(c_n)
    X = diri.rsample()
    return X.mean(0) 

def cond_samples(model, replay_buffer, device, opt, fresh=False, use_buffer=False):
    sqrt = lambda x: int(torch.sqrt(torch.tensor([x])))
    plot = lambda p,x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
    n_cls = opt.n_cls
    meta_buffer_size = opt.buffer_size // n_cls
    model.eval()
    log_dir = os.path.join(opt.save_folder, 'log.txt')
    f = open(log_dir, 'w')
    if fresh:
        replay_buffer = freshh(model, opt, save=opt.open_debug, device=device)
    
    
    n_cls = opt.n_cls
    n_it = opt.buffer_size // 100
    n_range = opt.buffer_size // n_cls
    all_y = []
    # n_cls = opt.n_cls
    for i in range(n_it):
        x = replay_buffer[i * 100: (i + 1) * 100].to(device) 
        y = model(x, cls_mode=True).max(1)[1]
        # y = torch.LongTensor([i] * 100).to(device)
        all_y.append(y)
    
    all_y = torch.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(n_cls)]
    imgs = []
    # energys =[[] for  _ in range(n_cls)]
    for i in tqdm.tqdm(range(n_cls)):
        energys = []
        y = torch.LongTensor([i]).to(device)
        if opt.save_grid:
            plot('{}/samples_label_{}.png'.format(opt.save_dir, i), each_class[i])
            
        else:
            for j, im in enumerate(each_class[i]):
                output = model(im.unsqueeze(0).to(device))[0].mean()
                output_xy = model(im.unsqueeze(0).to(device), y=y)[0].mean()
                # if output < -10 and output_xy < -10:
                #     continue
                plot('{}/samples_label_{}_{}.png'.format(opt.save_dir, i, j), im)
                energys.append((output_xy - output).cpu().item())
                
                write_str = 'samples_label_{}_{}\tf(x):{:.4f}\tf(x,y):{:.4f}\n'.format(i, j, output, output_xy)
                f.write(write_str)

            random_seed = torch.FloatTensor(energys).argsort()[-10:]
            # print(each_class[random_seed].shape)
            # print()
            imgs.append(each_class[i][random_seed])

    print('Successfully saving the generated result of replay buffer.')
    f.close()
    if not opt.save_grid:
        imgs = torch.cat(imgs, 0)
        if opt.dataset != 'cifar100':             
            plot('{}/sample_10_perclass.png'.format(opt.save_folder), imgs)
            print('Successfully save the result.')

    return replay_buffer

def update_lc_theta(opt, x_q, t_logit, y_gt, s_logit, t_logit_true):
    # l_tv = get_image_prior_losses(x_q)
    n_cls = opt.n_cls
    y_one_hot = torch.eye(n_cls)[y_gt].to(x_q.device)
    l_cls = -torch.sum(torch.log_softmax(t_logit, 1) * y_one_hot, 1)
    bs = x_q.size(0)
    l_2 = torch.norm(x_q.view(bs, -1), dim=-1)
    # print(l_cls.shape, l_2.shape, l_tv.shape)
    # KL(p_t(y|x) || p_s(y|x))
    l_e = torch.sum(torch.softmax(t_logit, 1) * (torch.log_softmax(s_logit, 1)- torch.log_softmax(t_logit_true, 1)), 1)
    lc = opt.lmda_l2 * l_2 + 0.1*l_cls + opt.lmda_e * l_e
    # print(lc.mean(), (lc - lc.mean()).mean())
    # c = lc.mean()
    return lc, (l_2, l_cls, l_e)

def update_theta(opt, replay_buffer, models, x_p, x_lab, y_lab, mode='sep', y_p=None):
    L = 0
    if mode == 'joint':
        model_t, model_s, model = models
        model_t.eval()
        model_s.eval()
    else:
        model = models[0]
        # model_t.eval()
    sample_q = langevin_at_x(opt, x_p.device)
    cache_p_x = None
    cache_p_x_y = None
    ls = []
    # P(x) modeling
    if opt.lmda_p_x > 0:
        y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(x_p.device)
        # The process of get x_q~p_{\theta}(x), stochastic process of x*=argmin_{x}(E_{\theta}(x))
        # print(replay_buffer.shape, y_q.shape)
        x_q, samples = sample_q(model, replay_buffer, y=y_q, train=True)
        # print(x_p.device, opt.y.device)
        f_p = model(x_p, py=opt.y)[0]
        f_q = model(x_q, py=opt.y)[0]
        fp = f_p.mean()
        fq = f_q.mean()
        # print(fp, fq)
        l_p_x = -(fp-fq)
        L += opt.lmda_p_x * l_p_x
        ls.append(l_p_x)
        cache_p_x = (fp, fq)
        x_pos, y_pos = x_p, y_q
    else:
        ls.append(0.0)
    # Here use x_labeled and y_labeled, to unifying the label information.
    # Followed by the idea of jem

    if opt.lmda_p_x_y > 0:
        x_q_lab, samples = sample_q(model, replay_buffer, y=y_lab, train=True)
        # -E_{\theta}, bigger better.
        fpxys = model(x_lab, y_lab, py=opt.y)[0]
        fqxys = model(x_q_lab, y_lab, py=opt.y)[0]
        
        fpxy = fpxys.mean()
        fqxy = fqxys.mean()
        l_p_x_y  = -(fpxy-fqxy)
        cache_p_x_y = (fpxy, fqxy)
        L += opt.lmda_p_x_y * l_p_x_y
        ls.append(l_p_x_y)
        x_pos, y_pos = x_lab, y_lab
    else:
        ls.append(0.0)

    # P(y | x). Here needs the update of x.
    logit = model(x_lab, cls_mode=True)
    # print(logit.shape, x_lab.shape)
    l_cls = torch.nn.CrossEntropyLoss()(logit, y_lab)
    
    L += l_cls
    # l_cls.backward
    # l_cls = -torch.log_softmax(logit, 1).mean() - math.log(opt.n_cls)
    ls.append(l_cls)
    if mode == 'joint':
        # x_pos = x_lab
        logit_t_pos = model_t(x_pos)
        l_b_s = 0.
        K = opt.lc_K
        # print(K)
        if opt.st == -1:
            # Randomly sample the start point
            st = random.randint(0, opt.g_steps - K)
        else:
            # Sample from the given start point.
            st = opt.st
        for sample_at_k in samples[st:st+K]:
            x_k, x_k_minus_1, noise = sample_at_k
            logit_s = model_s(x_k)
            logit_t = model_t(x_k)
            l_c_k, cache_l_k = update_lc_theta(opt, x_k, logit_t, y_pos, logit_s, logit_t_pos) # l_c_k.requires_grad = False
            l2_k, l_cls_k, l_e_k = cache_l_k
            # print(x_k_minus_1.requires_grad, x_k.requires_grad, noise.requires_grad)
            # exit(-1)
            with torch.no_grad():
                logit_s = model_s(x_k_minus_1)
                logit_t = model_t(x_k_minus_1)
                l_c_k_minus_1, cache_l_k_1 = update_lc_theta(opt, x_k_minus_1, logit_t, y_pos, logit_s, logit_t_pos) # l_c_{k-1}.requires_grad = False
                l2_k_1, l_cls_k_1, l_e_k_1 = cache_l_k_1
            # lc target updation
            mu = x_k - noise # mu.requires_grad = True
            sigma = 0.01 * torch.ones_like(x_k)

            nll = diag_normal_NLL(torch.flatten(x_k, 1), torch.flatten(mu, 1), 2*torch.flatten(sigma, 1).log()).mean(1)
            l_b = ((l_c_k_minus_1 - l_c_k) * nll).mean()
            l_b_s += opt.g_steps / opt.lc_K * l_b 
            # print(l_b, (l_c_k_minus_1).mean(), l_c_k.mean())
            # L += l_b * 100
        # ls.append(l_b)
        
        # l_b_s.backward()
        L += l_b_s
        ls.append(l_c_k.mean())
        ls.append(l_c_k_minus_1.mean())
        ls.append(l2_k.mean())
        ls.append(l_cls_k.mean())
        ls.append(l_e_k.mean())
        ls.append(l2_k_1.mean())
        ls.append(l_cls_k_1.mean())
        ls.append(l_e_k_1.mean())

    # print(l_cls)
    # L += l_cls
    
    
    # print(l_p_x, l_cls)
    if L.abs().item() > 1e8:
        print('Bad Result.')
        print(cache_p_x, cache_p_x_y)
        print(L.item(), l_p_x.item(), l_p_x_y.item())
        raise ValueError('Not converged.')
    # print(L)
    return L, cache_p_x, cache_p_x_y, logit, ls


def create_similarity(netT_path, scale=1, device=None):
    # Motivated by Zero-shot KD, Nayak et. al, ICML 2019
    weight = torch.load(netT_path, map_location=device)['model']['fc.weight']
    K, _ = weight.shape
    # K: num_classes
    weight_normed = weight / torch.norm(weight, dim=1).unsqueeze(-1)
    # print(weight_normed)
    return torch.matmul(weight_normed, weight_normed.T) / scale