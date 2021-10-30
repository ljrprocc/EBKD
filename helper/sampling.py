# Sampling functions for choosing appropriate sampling strategies.
# Including: Langevin Sampling(image level), Langevin Sampling(latent level), Entropy Regulariation, Variational Inference.

import torch
import numpy as np
from torch.autograd import Variable
import torchvision.utils as vutils
import tqdm
from torchvision import transforms
from PIL import Image
from torch import nn
import os

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


def init_random(s):
    # print(s)
    return torch.FloatTensor(*s).uniform_(-1, 1)

def get_replay_buffer(opt, model=None, local_rank=None):
    
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
        ddp = (opt.dataset == 'imagenet')
        if ddp:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        ckpt_dict = torch.load(opt.load_buffer_path)
        replay_buffer = ckpt_dict["replay_buffer"]
        if model:
            model.load_state_dict(ckpt_dict["model_state_dict"])
    return replay_buffer, model

def langevin_at_x(opts, device=None):
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
            f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
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


def get_replay_buffer(opt, model=None, local_rank=None):
    
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
        ddp = (opt.dataset == 'imagenet')
        if ddp:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        ckpt_dict = torch.load(opt.load_buffer_path)
        replay_buffer = ckpt_dict["replay_buffer"]
        if model:
            model.load_state_dict(ckpt_dict["model_state_dict"])
    return replay_buffer, model


def freshh(model, opt, device, replay_buffer=None, save=True):
    sample_q, _ = langevin_at_x(opts=opt, device=device)
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



def langevin_at_z(args, device=None):
    def sample_p_0(n=args.batch_size, sig=args.e_init_sig):
        return sig * torch.randn(*[n, args.nz, 1, 1]).to(device)

    def sample_langevin_prior_z(netE, z, verbose=False, y=None):
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.g_steps):
            en = netE(z, y=y)
            z_grad = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * args.step_size * args.step_size * (z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
            if args.e_l_with_noise:
                z.data += args.step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == args.g_steps - 1) and verbose:
                print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, args.e_l_steps, en.sum().item()))

            z_grad_norm = z_grad.view(args.batch_size, -1).norm(dim=1).mean()

        return z.detach(), z_grad_norm

    def sample_langevin_post_z(netE, z, args, netG, x, verbose=False, y=None):

        mse = nn.MSELoss(reduction='sum')

        z = z.clone().detach()
        z.requires_grad = True
        for i in range(args.g_steps):
            x_hat = netG(z)
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat, x)
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            en = netE(z, y=y)
            z_grad_e = torch.autograd.grad(en.sum(), z)[0]

            z.data = z.data - 0.5 * args.step_size * args.step_size * (z_grad_g + z_grad_e + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
            if args.g_l_with_noise:
                z.data += args.step_size * torch.randn_like(z).data

            if (i % 5 == 0 or i == args.g_steps - 1) and verbose:
                print('Langevin posterior {:3d}/{:3d}: MSE={:8.3f}'.format(i+1, args.g_steps, g_log_lkhd.item()))

            z_grad_g_grad_norm = z_grad_g.view(args.batch_size, -1).norm(dim=1).mean()
            z_grad_e_grad_norm = z_grad_e.view(args.batch_size, -1).norm(dim=1).mean()

        return z.detach(), z_grad_g_grad_norm, z_grad_e_grad_norm

    return sample_langevin_prior_z, sample_langevin_post_z, sample_p_0