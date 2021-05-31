from __future__ import print_function

import torch
import numpy as np
import random
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.dirichlet import Dirichlet
import torchvision.utils as vutils
import cv2

def init_random(s1, s2, s3, s4):
    return torch.FloatTensor(s1, s2, s3, s4).uniform_(-1, 1)

def get_replay_buffer(opt):
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
    return replay_buffer

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

    def sample_q(f, replay_buffer, y=None, n_steps=opts.g_steps):
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
        for k in range(n_steps):
            f_prime = torch.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += opts.step_size * f_prime + 0.01 * torch.randn_like(x_k)
        f.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
    return sample_q

def update_theta(opt, replay_buffer, model, x_p, x_lab, y_lab):
    L = 0
    sample_q = get_sample_q(opt, x_p.device)
    cache_p_x = None
    cache_p_x_y = None
    # P(x) modeling
    if opt.lmda_p_x > 0:
        y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(x_p.device)
        # The process of get x_q~p_{\theta}(x), stochastic process of x*=argmin_{x}(E_{\theta}(x))
        x_q = sample_q(model, replay_buffer, y=y_q)
        
        f_p = model(x_p)
        f_q = model(x_q)
        fp = f_p.mean()
        fq = f_q.mean()
        l_p_x = -(fp-fq)
        L += opt.lmda_p_x * l_p_x
        cache_p_x = (fp, fq)
    # Here use x_labeled and y_labeled, to unifying the label information.
    # Followed by the idea of jem

    if opt.lmda_p_x_y > 0:
        x_q_lab = sample_q(model, replay_buffer, y=y_lab)
        fp, fq = model(x_lab, y_lab).mean(), model(x_q_lab, y_lab).mean()
        l_p_x_y  = -(fp-fq)
        cache_p_x_y = (fp, fq)
        L += opt.lmda_p_x_y * l_p_x_y

    # P(y | x). Here needs the update of x.
    logit = model(x_lab, cls_mode=True)
    # print(len(logit))
    l_cls = torch.nn.CrossEntropyLoss()(logit, y_lab)
    # print(l_cls)
    L += 0.1 * l_cls
    L.backward()
    # print(L)
    return L, cache_p_x, cache_p_x_y


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