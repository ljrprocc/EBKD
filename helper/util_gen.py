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
# from util import set_require_grad
import cv2

def init_random(s):
    # print(s)
    return torch.FloatTensor(*s).uniform_(-1, 1)

def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.4*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=0.2)
            color_distort = transforms.Compose([
                rnd_color_jitter,
                rnd_gray])
            return color_distort

def augment(dataset, sample):
    color_transform = get_color_distortion()
    if dataset == "cifar10" or "cifar100":
        transform = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == "continual":
        color_transform = get_color_distortion(0.1)
        transform = transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.7, 1.0)), color_transform, transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == "celeba":
        transform = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == "imagenet":
        transform = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.01, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == "object":
        transform = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.01, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == "lsun":
        transform = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.08, 1.0)), transforms.RandomHorizontalFlip(), color_transform, transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == "mnist":
        transform = None
    elif dataset == "moving_mnist":
        transform = None
    else:
        assert False

    im = sample.permute(1, 2, 0)
    im = transform(Image.fromarray(np.uint8((im + 1) / 2 * 255)))
    # print(im.max(),  im.min())
    return im



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

def get_replay_buffer(opt, model=None):
    
    nc = 3
    if opt.dataset == 'cifar100' or opt.dataset == 'cifar10' or opt.dataset == 'svhn':
        im_size = 32
    else:
        im_size = 224
    if not opt.load_buffer_path:
        bs = opt.capcitiy
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
    elif opts.dataset == 'imagenet':
        im_size = 224
        n_cls = 1000
    else:
        im_size = 32
        n_cls = 10
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
        if opts.augment:
            samples = []
            for sample in buffer_samples:
                res_samples = augment(opts.dataset, sample)
                samples.append(res_samples)
            buffer_samples = torch.stack(samples, 0)
        random_samples = init_random((bs, nc, im_size, im_size))
        # s = (bs, opts.latent_dim)
        # random_samples = init_random(s)
        choose_random = (torch.rand(bs) < opts.reinit_freq).float()[:, None, None, None]
        # choose_random = (torch.rand(bs) < opts.reinit_freq).float()[:, None]
        # print(random_samples.shape, buffer_samples.shape)
        # print(random_samples.device, buffer_samples.device)
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        if device:
            return samples.to(device), inds
        else:
            return samples.cuda(), inds

    def sample_q(f, replay_buffer, y=None, n_steps=opts.g_steps, open_debug=False, open_clip_grad=None, train=False):
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
            f_prime = torch.autograd.grad(f(x_k, y=y, multiscale=opts.multiscale and train)[0].sum(), [x_k], retain_graph=True)[0]
            noise = 0.01 * torch.randn_like(x_k)
            x_k = x_k + now_step_size * f_prime + noise
            # now_step_size *= 0.99
            # print((now_step_size * f_prime + noise).mean())
            if y is not None:
                samples.append((x_k, x_k_pre, noise))
            # if open_clip_grad:
            #     torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=open_clip_grad)
                # plot('{}/debug_{}.png'.format(opts.save_folder, k))
                # exit(-1)
            x_k_pre = x_k.detach()
            
            
        f.train()
        
        final_samples = x_k.detach()
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        if y is not None:
            return final_samples, samples
        else:
            return final_samples

    return sample_q

def freshh(model, opt, device, replay_buffer=None, save=True):
    sample_q, _ = get_sample_q(opts=opt, device=device)
    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: vutils.save_image(torch.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    replay_buffer = torch.FloatTensor(opt.buffer_size, 3, 32, 32).uniform_(-1, 1)
    print(replay_buffer.shape)
    y = torch.arange(0, opt.n_cls).to(device)
    if opt.resume != 'none':
        ckpt = torch.load(opt.resume)
        replay_buffer = ckpt['replay_buffer']
    for i in tqdm.tqdm( range(opt.init_epoch,opt.n_sample_steps)):
        samples, _ = sample_q(model, replay_buffer, y=y)
        # if i == 0:
        #     ckpt_dict = {
        #         "model_state_dict": model.state_dict(),
        #         "replay_buffer": replay_buffer
        #     }
        #     torch.save(ckpt_dict, os.path.join(opt.save_folder, 'res_buffer_0.pts'))
        if i % opt.print_every == 0 and save:
            plot('{}/samples_{}.png'.format(opt.save_folder, i), samples)
            ckpt_dict = {
                "model_state_dict": model.state_dict(),
                "replay_buffer": replay_buffer
            }
            torch.save(ckpt_dict, os.path.join(opt.save_ckpt, 'res_buffer_{}.pts'.format(i)))
        # print(i)
    return replay_buffer

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
    sample_q = get_sample_q(opt, x_p.device)
    cache_p_x = None
    cache_p_x_y = None
    ls = []
    # P(x) modeling
    if opt.lmda_p_x > 0:
        y_q = torch.randint(0, opt.n_cls, (opt.batch_size,)).to(x_p.device)
        # The process of get x_q~p_{\theta}(x), stochastic process of x*=argmin_{x}(E_{\theta}(x))
        # print(replay_buffer.shape, y_q.shape)
        x_q, samples = sample_q(model, replay_buffer, y=y_q, train=True)
        f_p = model(x_p, py=opt.y, multiscale=opt.multiscale)[0]
        f_q = model(x_q, py=opt.y, multiscale=opt.multiscale)[0]
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
        fpxys = model(x_lab, y_lab, py=opt.y, multiscale=opt.multiscale)[0]
        fqxys = model(x_q_lab, y_lab, py=opt.y, multiscale=opt.multiscale)[0]
        
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
    # print(len(logit))
    logit = model(x_lab, cls_mode=True)
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
                
                # f_q_k_minus_1 = model(x_neg, y=y_lab, py=opt.y)[0]
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


def create_similarity(netT_path, scale=1):
    # Motivated by Zero-shot KD, Nayak et. al, ICML 2019
    weight = torch.load(netT_path)['model']['fc.weight']
    K, _ = weight.shape
    # K: num_classes
    weight_normed = weight / torch.norm(weight, dim=1).unsqueeze(-1)
    # print(weight_normed)
    return torch.matmul(weight_normed, weight_normed.T) / scale

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
