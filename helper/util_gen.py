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

def init_random(s1, s2, s3, s4):
    return torch.FloatTensor(s1, s2, s3, s4).uniform_(-1, 1)

def get_replay_buffer(opt, model=None):
    bs = opt.capcitiy
    nc = 3
    if opt.dataset == 'cifar100':
        im_size = 32
    else:
        im_size = 224
    if not opt.load_buffer_path:
        replay_buffer = init_random(bs, nc, im_size, im_size)
    else:
        print('Loading replay buffer from local..')
        ckpt_dict = torch.load(opt.load_buffer_path)
        replay_buffer = ckpt_dict["replay_buffer"]
        if model:
            model.load_state_dict(ckpt_dict["model_state_dict"])
    return replay_buffer, model

def getDirichl(bs, num_classes, sim_matrix, scale):
    X = torch.zeros(bs, num_classes).to(sim_matrix.device)
    for i in range(num_classes):
        alpha = sim_matrix[i, :]
        alpha_normalized = (alpha - torch.min(alpha)) / (torch.max(alpha) - torch.min(alpha))
        alpha_normalized = alpha_normalized * scale + 0.000001
        # print(alpha_normalized.device)
        diri = Dirichlet(alpha_normalized)
        X += diri.rsample((bs, ))

    return X / num_classes

def get_sample_q(opts, device=None):
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
            return init_random(bs, nc, im_size, im_size), []
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // opts.n_cls
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds
        buffer_samples = replay_buffer[inds]
        random_samples = init_random(bs, nc, im_size, im_size)
        choose_random = (torch.rand(bs) < opts.reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.cuda(), inds

    def sample_q(f, replay_buffer, y=None, n_steps=opts.g_steps, open_debug=False):
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
        now_step_size = opts.step_size
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
            x_k.data += now_step_size * f_prime + 0.01 * torch.randn_like(x_k)
            now_step_size *= 0.99
            if open_debug:
                plot('{}/debug_{}.png'.format(opts.save_folder, k))
                exit(-1)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q
    
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
        x_q = sample_q(model, replay_buffer, y=y_q)
        loss_x, score_qx = sliced_VR_score_matching(model, x_q)
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

def cond_samples(model, replay_buffer, opt, fresh=False):
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
                y = torch.LongTensor([i]).cuda()
                plot('{}/samples_label_{}_{}.png'.format(opt.save_dir, i, j), replay_buffer[global_idx])
                output = model(replay_buffer[global_idx].unsqueeze(0).cuda())[0].mean()
                output_xy = model(replay_buffer[global_idx].unsqueeze(0).cuda(), y=y)[0].mean()
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

def update_lc_theta(opt, neg_out, x_q, logit, y_gt):
    l_tv = get_image_prior_losses(x_q)
    n_cls = opt.n_cls
    y_one_hot = torch.eye(n_cls)[y_gt].to(x_q.device)
    l_cls = -torch.sum(torch.log_softmax(logit, 1) * y_one_hot, 1)
    bs = x_q.size(0)
    l_2 = torch.norm(x_q.view(bs, -1), dim=-1)
    # print(l_cls.shape, l_2.shape, l_tv.shape)
    lc = opt.lmda_l2 * l_2 + opt.lmda_tv * l_tv + l_cls
    l_backward = (lc * neg_out).mean() - lc.mean() * neg_out.mean()
    return l_backward

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

def update_theta(opt, replay_buffer, model, x_p, x_lab, y_lab, model_t=None):
    L = 0
    sample_q = get_sample_q(opt, x_p.device)
    cache_p_x = None
    cache_p_x_y = None
    ls = []
    # P(x) modeling
    if opt.lmda_p_x > 0:
        y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(x_p.device)
        # The process of get x_q~p_{\theta}(x), stochastic process of x*=argmin_{x}(E_{\theta}(x))
        # print(replay_buffer.shape, y_q.shape)
        x_q = sample_q(model, replay_buffer, y=y_q)
        f_p, (mup, logp) = model(x_p, return_kl=True)
        f_q, (muq, logq) = model(x_q, return_kl=True)
        # stdp = torch.exp(logp / 2)
        # stdq = torch.exp(logq / 2)
        # print(stdp, stdq)
        kl = kl_div(mup, muq, logp, logq)
        fp = f_p.mean()
        fq = f_q.mean()
        l_p_x = -(fp-fq) + opt.lmda_kl * kl.mean()
        L += opt.lmda_p_x * l_p_x
        ls.append(l_p_x)
        cache_p_x = (fp, fq)
    else:
        ls.append(0.0)
    # Here use x_labeled and y_labeled, to unifying the label information.
    # Followed by the idea of jem

    if opt.lmda_p_x_y > 0:
        x_q_lab = sample_q(model, replay_buffer, y=y_lab)
        # -E_{\theta}, bigger better.
        fp, (mup, logp) = model(x_lab, y_lab, return_kl=True)
        fq, (muq, logq) = model(x_q_lab, y_lab, return_kl=True)
        # stdp = torch.exp(logp / 10)
        # stdq = torch.exp(logq / 10)
        # print(logp, logq)
        # unbiased estimation of variance
        # logvarp = fp.var().log()
        # logvarq = fq.var().log()
        # kl = kl_div(mup, muq, logp, logq)
        # kl = logvarp - logvarq
        # print(kl)
        fp = fp.mean()
        fq = fq.mean()
        l_p_x_y  = -(fp-fq)
        # l_ssm, _ = sliced_VR_score_matching(model, x_lab, y=y_lab)
        # l_p_x_y += opt.lmda_kl * l_ssm.mean()
        cache_p_x_y = (fp, fq)
        L += opt.lmda_p_x_y * l_p_x_y
        ls.append(l_p_x_y)
    else:
        ls.append(0.0)

    # P(y | x). Here needs the update of x.
    logit = model(x_lab, cls_mode=True)
    # print(len(logit))
    l_cls = torch.nn.CrossEntropyLoss()(logit, y_lab)
    # l_cls = -torch.log_softmax(logit, 1).mean() - math.log(opt.n_cls)
    ls.append(l_cls)
    if model_t:
        l_c = update_lc_theta(opt, fq, x_q_lab, logit, y_lab)
        L += l_c
    # print(l_cls)
    L += l_cls
    
    L.backward()
    # print(l_p_x, l_cls)
    if L.abs().item() > 1e8:
        print('Bad Result.')
        raise ValueError('Not converged.')
    # print(L)
    return L, cache_p_x, cache_p_x_y, ls


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
