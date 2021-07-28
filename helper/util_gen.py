from __future__ import print_function

import torch
import os
import numpy as np
import tqdm
import random
import math
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.distributions.dirichlet import Dirichlet
import torchvision.utils as vutils
import cv2

def init_random(s):
    # print(s)
    return torch.FloatTensor(*s).uniform_(-1, 1)

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

def estimate_h(x_lab, y_lab, model_vae, model, mode='ebm', batch_size=128):
    results = model_vae(x_lab, labels=y_lab)
    x_rec, mu, log_var, z = results[0], results[2], results[3], results[4]
    if mode == 'ebm':
        mu = mu.detach()
        log_var = log_var.detach()
        z = z.detach()
    # print(mu, log_var)
    
    dist = torch.distributions.Normal(mu, (log_var / 2).exp())
    dist_neg = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    logqz = dist.log_prob(z).mean(1)

    logqxgivenz = -torch.mean((x_rec - x_lab) ** 2, (1,2,3))
    log_probs_noise_pos = logqz + logqxgivenz
    
    # print(z, mu)
    x_neg, z_neg, logqxgivenz_neg = model_vae.sample(num_samples=batch_size, current_device=0, labels=y_lab, train=True)
    if mode == 'ebm':
        z_neg = z_neg.detach()
        x_neg = x_neg.detach()
    logqz_neg = dist_neg.log_prob(z_neg).mean(1)
    # print(x_neg.requires_grad, x_neg_rec.requires_grad, mode)
    
    # logqz
    log_probs_noise_neg = logqz_neg

    # print(mu_neg.shape, log_var_neg.shape, mu.shape, log_var.shape)
    # print(log_var.mean().item(), log_var_neg.mean().item())
    
    # dist_neg = torch.distributions.Normal(mu_neg, (log_var_neg/2).exp() / 2)
    # log_probs_noise_neg = dist.log_prob(z_neg).mean(1)
    # print(model.c)
    log_probs_ebm_pos = model(x=x_lab, y=y_lab, z=z)[0]
    log_probs_ebm_neg = model(x=x_neg, y=y_lab, z=z_neg)[0]
    
    logit_ebm = torch.cat([log_probs_ebm_pos, log_probs_ebm_neg], 1)
    if mode == 'vae':
        logit_ebm = logit_ebm.detach()
    logit_noise = torch.cat([log_probs_noise_pos.unsqueeze(1), log_probs_noise_neg.unsqueeze(1)], 1)
    logit_true = logit_ebm - logit_noise
    label = torch.zeros_like(logit_ebm)
    label[:, 0] = 1
    loss_theta = torch.nn.BCEWithLogitsLoss(reduction='none')(logit_true, label).sum(1)
    
    h_pos = torch.sigmoid(log_probs_ebm_pos.squeeze() - log_probs_noise_pos)
    h_neg = torch.sigmoid(log_probs_ebm_neg.squeeze() - log_probs_noise_neg)
    # print((h_pos > 0.5).shape)
    # print(( log_probs_noise_pos).mean().item(), (log_probs_noise_neg).mean().item())
    acc = ((h_pos > 0.5).sum() + (h_neg < 0.5).sum()) / (len(x_lab) + len(x_neg))
    if acc >= 0.5:
        next_mode = 'vae'
    else:
        next_mode = 'ebm'

    return -loss_theta.mean(), log_probs_ebm_pos, log_probs_ebm_neg, results, next_mode

def get_replay_buffer(opt, model=None):
    bs = opt.capcitiy
    nc = 3
    if opt.dataset == 'cifar100':
        im_size = 32
    else:
        im_size = 224
    if not opt.load_buffer_path:
        # replay_buffer = init_random((bs, opt.latent_dim))
        replay_buffer = init_random((bs, nc, im_size, im_size))
    else:
        print('Loading replay buffer from local..')
        ckpt_dict = torch.load(opt.load_buffer_path)
        replay_buffer = ckpt_dict["replay_buffer"]
        if model:
            model.load_state_dict(ckpt_dict["model_state_dict"])
    return replay_buffer, model

def getDirichl(net_path, scale=1, sim_scale=1):
    sim_matrix = create_similarity(net_path, scale=sim_scale)
    # X = torch.zeros(bs, num_classes).to(sim_matrix.device)
    c_n = (sim_matrix - sim_matrix.min(1)[0]) / (sim_matrix.max(1)[0] - sim_matrix.min(1)[0])
    c_n = c_n * scale + 0.000001
    diri = Dirichlet(c_n)
    X = diri.rsample()
    return X.mean(0)

def get_sample_q(opts, device=None, open_debug=False, l=None):
    # bs = opt.capcitiy
    nc = 3
    if opts.dataset == 'cifar100':
        im_size = 32
        n_cls = 100
    else:
        im_size = 224
        n_cls = 1000
    sqrt = lambda x: int(torch.sqrt(torch.tensor([x])))
    plot = lambda p,x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
    def sample_p_0(replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random((bs, nc, im_size, im_size)), []
            # return init_random((bs, l)), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // opts.n_cls
        # print(replay_buffer)
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
        buffer_samples = replay_buffer[inds]
        random_samples = init_random((bs, nc, im_size, im_size))
        # s = (bs, opts.latent_dim)
        # random_samples = init_random(s)
        choose_random = (torch.rand(bs) < opts.reinit_freq).float()[:, None, None, None]
        # choose_random = (torch.rand(bs) < opts.reinit_freq).float()[:, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.cuda(), inds

    def sample_q(f, replay_buffer, y=None, n_steps=opts.g_steps, open_debug=False, open_clip_grad=None):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = opts.batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        samples = []
        now_step_size = opts.step_size
        x_k_pre = init_sample
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
            noise = 0.01 * torch.randn_like(x_k)
            x_k = x_k + now_step_size * f_prime + noise
            # now_step_size *= 0.99
            # print((now_step_size * f_prime + noise).mean())
            samples.append((x_k.detach(), x_k_pre, noise))
            # if open_clip_grad:
            #     torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=open_clip_grad)
                # plot('{}/debug_{}.png'.format(opts.save_folder, k))
                # exit(-1)
            x_k_pre = x_k.detach()
            
            
        f.train()
        # exit(-1)
        # idx = random.randint(1, n_steps - 1)
        # sample_at_k = samples[idx]
        # print(idx, n_steps)
        # sample_at_k = x_k, init_sample, noise
        # print((x_k - init_sample).mean())
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        if open_debug:
            return final_samples, samples
        else:
            return final_samples, samples

    def sample_q_xy(f, replay_buffer, y=None, n_steps=opts.g_steps):
        f.eval()
        bs = opts.batch_size if y is None else y.size(0)
        init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)

        now_step_size = opts.step_size
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k, cls_mode=True).sum(0), [x_k], grad_outputs=torch.ones_like(opt.n_cls))
            print(f_prime.shape)
            x_k.data += now_step_size * f_prime + 0.01 * torch.randn_like(x_k)
            now_step_size *= 0.99
        
        f.train()
        final_samples = x_k.detach()

        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples

    return sample_q, sample_q_xy
    
def sliced_VR_score_matching(energy_net, samples, noise=None, detach=False, noise_type='radermacher', y=None):
    # Single MCMC step, for the matching of score function, i.e.,
    # s_m(x;\theta) = \nabla_{x} \log p_{\theta}(x)
    dup_samples = samples.unsqueeze(0).expand(1, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    logp = energy_net(dup_samples, y=y)[0].sum()
    grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(1, -1).mean(dim=0)
    loss2 = loss2.view(1, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss, logp

def ssm_sample(opt, replay_buffer, model, x_p, x_lab, y_lab):
    sample_q = get_sample_q(opt, x_p.device)
    cache_p_x = None
    cache_p_y = None
    L = 0
    score_qx = 0
    score_q_xy = 0
    if opt.lmda_p_x > 0:
        y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(x_p.device)
        x_q = sample_q(model, replay_buffer, y=y_q, open_clip_grad=0.1)
        loss_x, score_qx = sliced_VR_score_matching(model, x_q, open_clip_grad=0.1)
        L += loss_x * opt.lmda_p_x
    if opt.lmda_p_x_y > 0:
        x_q_lab = sample_q(model, replay_buffer, y=y_lab)
        loss_xy, score_q_xy = sliced_VR_score_matching(model, x_q_lab, y=y_lab)
        L += loss_xy * opt.lmda_p_x_y
    L.backward()
    return L, score_qx, score_q_xy

def get_image_prior_losses(inputs):
    bs = inputs.size(0)
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
    diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1.view(bs, -1), dim=-1) + torch.norm(diff2.view(bs, -1), dim=-1) + torch.norm(diff3.view(bs, -1), dim=-1) + torch.norm(diff4.view(bs, -1), dim=-1)

    return loss_var_l2

def cond_samples(model, replay_buffer, opt, fresh=False, use_buffer=True):
    sqrt = lambda x: int(torch.sqrt(torch.tensor([x])))
    plot = lambda p,x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))
    n_cls = opt.n_cls
    meta_buffer_size = replay_buffer.size(0) // n_cls
    model.eval()
    log_dir = os.path.join(opt.save_folder, 'log.txt')
    f = open(log_dir, 'w')
    
    for i in tqdm.tqdm(range(n_cls)):
        if opt.save_grid:
            plot('{}/samples_label_{}.png'.format(opt.save_dir, i), replay_buffer[i*meta_buffer_size:(i+1)*meta_buffer_size])
        else:
            for j in range(meta_buffer_size):
                global_idx = i*meta_buffer_size+j
                x = replay_buffer[global_idx].unsqueeze(0).cuda()
                if use_buffer:
                    logit = model(x, cls_mode=True)
                    y = logit.argmax(1)
                else:
                    y = torch.LongTensor([i]).cuda()
                plot('{}/samples_label_{}_{}.png'.format(opt.save_dir, i, j), replay_buffer[global_idx])
                output = model(x)[0].mean()
                output_xy = model(x, y=y)[0].mean()
                write_str = 'samples_label_{}_{}\tf(x):{:.4f}\tf(x,y):{:.4f}\n'.format(i, j, output, output_xy)
                f.write(write_str)


    
    print('Successfully saving the generated result of replay buffer.')
    f.close()
        

    # if fresh:
    #     pass
    
    # n_it = replay_buffer.size(0) // 100
    # all_y = []
    # n_cls = opt.n_cls
    # for i in range(n_it):
    #     x = replay_buffer[i * 100: (i + 1) * 100].cuda()
    #     y = model(x, cls_mode=True).max(1)[1]
    #     all_y.append(y)
    
    # all_y = torch.cat(all_y, 0)
    # each_class = [replay_buffer[all_y == l] for l in range(n_cls)]
    # print([len(c) for c in each_class])
    # for i in range(100):
    #     this_im = []
    #     for l in range(n_cls):
    #         this_l = each_class[l][i*n_cls:(i+1)*n_cls]
    #         this_im.append(this_l)
    #     this_im = torch.cat(this_im, 0)
    #     print(this_im)
    #     # print(this_im.size(0))
    #     if this_im.size(0) > 0:
    #         plot('{}/samples_{}.png'.format(opt.save_dir, i), this_im)

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
    lc = opt.lmda_l2 * l_2 + l_cls + opt.lmda_e * l_e
    # print(lc.mean(), (lc - lc.mean()).mean())
    # c = lc.mean()
    return lc

def kl_div(mu1, mu2, std1, std2):
    kl_1 = -0.5 * torch.sum(1 + std1 - mu1 ** 2 - std1.exp(), dim = 1)
    kl_2 = -0.5 * torch.sum(1 + std2 - mu2 ** 2 - std2.exp(), dim = 1)
    # p = torch.distributions.Normal(mu1, std1)
    # q = torch.distributions.Normal(mu2, std2)

    # log_qxz = q.log_prob(z)
    # log_pz = p.log_prob(z)
    # kl = (log_qxz - log_pz)
    # kl = kl.sum(-1)
    return kl_1 + kl_2

def update_theta(opt, replay_buffer, models, x_p, x_lab, y_lab, mode='sep'):
    L = 0
    if mode == 'joint':
        model_t, model_s, model = models
        model_t.eval()
        model_s.eval()
    else:
        model_t, model = models
        model_t.eval()
    sample_q, sample_q_xy = get_sample_q(opt, x_p.device)
    cache_p_x = None
    cache_p_x_y = None
    ls = []
    # P(x) modeling
    if opt.lmda_p_x > 0:
        y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(x_p.device)
        # The process of get x_q~p_{\theta}(x), stochastic process of x*=argmin_{x}(E_{\theta}(x))
        # print(replay_buffer.shape, y_q.shape)
        x_q, samples = sample_q(model, replay_buffer, y=y_q,open_clip_grad=0.1)
        f_p = model(x_p)[0]
        f_q = model(x_q)[0]
        # stdp = torch.exp(logp / 2)
        # stdq = torch.exp(logq / 2)
        # print(stdp, stdq)
        fp = f_p.mean()
        fq = f_q.mean()
        l_p_x = -(fp-fq)
        L += opt.lmda_p_x * l_p_x
        ls.append(l_p_x)
        cache_p_x = (fp, fq)
    else:
        ls.append(0.0)
    # Here use x_labeled and y_labeled, to unifying the label information.
    # Followed by the idea of jem

    if opt.lmda_p_x_y > 0:
        x_q_lab, samples = sample_q(model, replay_buffer, y=y_lab, open_clip_grad=0.1)
        # -E_{\theta}, bigger better.
        f_p = model(x_lab, y_lab, py=opt.y)[0]
        f_q = model(x_q_lab, y_lab, py=opt.y)[0]
        fp = f_p.mean()
        fq = f_q.mean()
        l_p_x_y  = -(fp-fq)
        cache_p_x_y = (fp, fq)
        L += opt.lmda_p_x_y * l_p_x_y
        ls.append(l_p_x_y)
    else:
        ls.append(0.0)

    # P(y | x). Here needs the update of x.
    
    if opt.cls == 'cls':
        # print(len(logit))
        logit = model(x_lab, cls_mode=True)
        l_cls = torch.nn.CrossEntropyLoss()(logit, y_lab)
        L += l_cls
    else:
        logit = model(x_q_lab, cls_mode=True)
        l_cls = -torch.gather(torch.log_softmax(logit, 1), 1, y_lab[:, None]).mean() - math.log(opt.n_cls)
        L += l_cls
        # l_cls.backward
    # l_cls = -torch.log_softmax(logit, 1).mean() - math.log(opt.n_cls)
    ls.append(l_cls)
    if mode == 'joint':
        x_pos = x_lab
        logit_t_pos = model_t(x_lab)
        l_b_s = []
        for sample_at_k in samples[3:8]:
            x_k, x_k_minus_1, noise = sample_at_k
            # x_k lc calculation
            # x_neg = x_k
            # print(x_k_minus_1.shape)
            logit_s = model_s(x_k)
            logit_t = model_t(x_k)
            # print(logit_s.requires_grad, logit_t.requires_grad)
            # f_q_k = model(x_neg, y=y_lab, py=opt.y)[0]
            l_c_k = update_lc_theta(opt, x_k, logit_t, y_lab, logit_s, logit_t_pos) # l_c_k.requires_grad = False
            # x_{k - 1} lc calculation
            # x_neg = x_k_minus_1
            with torch.no_grad():
                logit_s = model_s(x_k_minus_1)
                logit_t = model_t(x_k_minus_1)
                
                # f_q_k_minus_1 = model(x_neg, y=y_lab, py=opt.y)[0]
                l_c_k_minus_1 = update_lc_theta(opt, x_k_minus_1, logit_t, y_lab, logit_s, logit_t_pos) # l_c_{k-1}.requires_grad = False
            # lc target updation
            # Here p_{\theta} ~ N(x_{k - 1} + \sigma_{k-1}^2 s_{\theta}(x_{k - 1}), \sigma_{k-1}^2)
            # The mu_{\theta} here using minus operation, to avoid the second calculation of gradient.
            mu = x_k - noise # mu.requires_grad = True
            sigma = 0.01 * torch.ones_like(x_k)

            nll = diag_normal_NLL(torch.flatten(x_k, 1), torch.flatten(mu, 1), torch.flatten(sigma, 1)).mean(1)
            l_b = ((l_c_k_minus_1 - l_c_k) * nll).mean()
            l_b_s.append(l_b * 20)
            # print(l_b, (l_c_k_minus_1).mean(), l_c_k.mean())
            # L += l_b * 100
        # ls.append(l_b)
        sum(l_b_s).backward()
        ls.append(l_c_k.mean())
        ls.append(l_c_k_minus_1.mean())
    # print(l_cls)
    # L += l_cls
    
    L.backward()
    # print(l_p_x, l_cls)
    if L.abs().item() > 1e8:
        print('Bad Result.')
        raise ValueError('Not converged.')
    # print(L)
    return L, cache_p_x, cache_p_x_y, logit, ls


def create_similarity(netT_path, scale=1):
    # Motivated by Zero-shot KD, Nayak et. al, ICML 2019
    weight = torch.load(netT_path)['model']['fc.weight']
    K, _ = weight.shape
    # K: num_classes
    weight_normed = weight / torch.norm(weight, dim=1).unsqueeze(-1)
    # print(weight_normed)
    return torch.matmul(weight_normed, weight_normed.T) / scale

def update_x(model, neg_img, opt, y=None, global_iters=0):
    n_cls = model.n_cls
    criterion_cls = torch.nn.CrossEntropyLoss()
    optim_X = optim.Adam([neg_img], lr=opt.g_lr, betas=[0.5, 0.99], eps=1e-8)

    with torch.autograd.set_detect_anomaly(True):
        for i in range(opt.g_steps):
            logit_out, neg_out = model(neg_img, return_logit=True)
            if i == opt.g_steps - 1:
                loss_out = neg_out.sum()
                grad_now = autograd.grad(loss_out, [neg_img], retain_graph=True, only_inputs=True)[0]
            optim_X.zero_grad()
            model.zero_grad()
            loss_cls = criterion_cls(logit_out, y)
            loss_tv = get_image_prior_losses(neg_img)
            loss_l2 = neg_img.view(neg_img.shape[0], -1).norm(1).mean()
            loss_aux = loss_cls + opt.lmda_tv * loss_tv + opt.lmda_l2 * loss_l2
            loss_aux.backward()
            optim_X.step()
            global_iters += 1
    return neg_img.detach(), grad_now, global_iters



def SGLD_autograd(model, neg_img, opt, y=None, global_iters=None):
    n_cls = model.n_cls
    criterion_cls = torch.nn.CrossEntropyLoss()
    # optimizer for update of X
    optim_X = optim.Adam([neg_img], lr=0.05, betas=[0.5, 0.999], eps=1e-8)
    for _ in range(opt.steps):
        # noise = torch.randn_like(neg_img).to(neg_img.device)
        logit_out, neg_out = model(neg_img, return_logit=True)
        optim_X.zero_grad()
        model.zero_grad()
        loss_cls = criterion_cls(logit_out, y)
        loss_tv = get_image_prior_losses(neg_img)
        loss_l2 = neg_img.view(neg_img.shape[0], -1).norm(1).mean()
        loss_aux = loss_cls + opt.lmda_tv * loss_tv + opt.lmda_l2 * loss_l2
        # print(loss_aux)
        loss_aux.backward()
        optim_X.step()
        global_iters += 1
    
    set_require_grad(model, True)
    model.train()
    return neg_img.detach(), global_iters


def SGLD(model, neg_img, opt, y=None):
    # neg_img.requires_grad = True
    # print(neg_img)
    for _ in range(opt.steps):
        noise = torch.randn_like(neg_img).to(neg_img.device)
        logit_out, neg_out = model(neg_img, return_logit=True)
        loss_out = neg_out.sum()
        # print(loss_tv, loss_l2, loss_out)
        # print(neg_out)
        # print(neg_out.mean())
        grad_x = autograd.grad(loss_out, [neg_img], retain_graph=True)[0]
        
        # print(loss_out)
        neg_img.data = neg_img.data - opt.step_size * grad_x + 0.01 * noise
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        # neg_img.clamp_(-1, 1)
    # print('****************')
    set_require_grad(model, True)
    model.train()
    # print(neg_img.shape)
    # neg_img.clamp_(-1, 1)
    return neg_img.detach()

def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]

                if 'step' not in state or state['step'] < 1:
                    continue

                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']

                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))

class SampleBuffer:
    def __init__(self, max_samples=10000, net_T=None):
        self.max_samples = max_samples
        self.buffer = []
        sim_mat = create_similarity(net_T, 1)
        self.sim_mat = sim_mat

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, class_ids=None, global_iters=None):
        samples = samples.detach().to('cpu')
        class_ids = class_ids.detach().to('cpu')
        global_iters = global_iters.detach().to('cpu')

        for sample, class_id, global_iter in zip(samples, class_ids, global_iters):
            self.buffer.append((sample.detach(), class_id, global_iter))

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples, device='cuda'):
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids, global_iters = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        global_iters = torch.tensor(global_iters)
        # class_ids = torch.stack(class_ids, 0)
        samples = samples.to(device)
        class_ids = class_ids.to(device)
        global_iters = global_iters.to(device)

        return samples, class_ids, global_iters


def sample_buffer(buffer, y, batch_size=128, p=0.95, device='cuda', num_classes=100):
    
    if len(buffer) < 1:
        return (
            init_random(batch_size, 3, 32, 32).to(device),
            torch.randint(0, num_classes, (batch_size,)).to(device),
            torch.zeros(batch_size).to(device)
            # y,
        )

    n_replay = (np.random.rand(batch_size) < p).sum()

    replay_sample, replay_id, global_iters = buffer.get(n_replay)
    random_sample = init_random(batch_size - n_replay, 3, 32, 32).to(device)
    random_id = torch.randint(0, num_classes, (batch_size - n_replay,)).to(device)
    random_global_iter = torch.zeros((batch_size - n_replay, )).to(device)

    return (
        torch.cat([replay_sample, random_sample], 0),
        torch.cat([replay_id, random_id], 0),
        torch.cat([global_iters, random_global_iter], 0)
    )

def NT_XentLoss(z1, z2, temperature=0.5):
    '''
    SimCLR NT-Xent Loss.
    '''
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)
