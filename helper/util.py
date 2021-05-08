from __future__ import print_function

import torch
import numpy as np
import random
import torch.autograd as autograd
from torch.distributions.dirichlet import Dirichlet
import cv2


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        prob, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_img(g, save_path, label):
    # ten = ten[0]
    label = torch.IntTensor([label])
    g.eval()
    ten = torch.randn(1,100)
    img = g(ten, label) 
    img = img[0].detach().numpy()
    img = np.transpose(img, (1,2,0))
    img = img / 2 + 0.5
    img = (img * 255).astype(np.int)
    cv2.imwrite(save_path, img)

def set_require_grad(net, flag):
    for p in net.named_parameters():
        if p[0].split('.')[0] != 'energy_output':
            p[1].requires_grad = flag

def init_random(s1, s2, s3, s4):
    return torch.FloatTensor(s1, s2, s3, s4).uniform_(-1, 1)

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

def create_similarity(netT_path, scale=1):
    # Motivated by Zero-shot KD, Nayak et. al, ICML 2019
    weight = torch.load(netT_path)['model']['fc.weight']
    K, _ = weight.shape
    # K: num_classes
    weight_normed = weight / torch.norm(weight, dim=1).unsqueeze(-1)
    # print(weight_normed)
    return torch.matmul(weight_normed, weight_normed.T) / scale


def sample_mcmc(model, neg_img, neg_id, noise, steps=20, step_size=0.1, detachs=True):
    neg_img.requires_grad = True
    for k in range(steps):
        # noise = torch.randn_like(neg_img)
        if noise.shape[0] != neg_img.shape[0]:
            noise = torch.randn(neg_img.shape[0], 3, 32, 32, device=neg_img.device)

        noise.normal_(0, 0.01)
        # print(neg_img.shape, noise.shape)
        neg_img.data.add_(noise.data)

        neg_out = model(neg_img, neg_id)
        neg_out.sum().backward()
        neg_img.grad.data.clamp_(-0.01, 0.01)

        neg_img.data.add_(-step_size, neg_img.grad.data)

        neg_img.grad.detach_()
        neg_img.grad.zero_()

        # neg_img.data.clamp_(-1, 1)

    # neg_img = neg_img.detach()
    return neg_img

def SGLD(model, neg_img, steps=20, step_size=2, y=None):
    # neg_img.requires_grad = True
    # print(neg_img)
    for _ in range(steps):
        noise = torch.randn_like(neg_img).to(neg_img.device)
        neg_out = model(neg_img, y=y)
        loss_out = neg_out.sum()
        # print(neg_out.mean())
        grad_x = autograd.grad(loss_out, [neg_img], retain_graph=True)[0]
        
        # print(loss_out)
        neg_img.data += step_size * grad_x + 0.01 * noise
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

    def push(self, samples, class_ids=None):
        samples = samples.detach().to('cpu')
        class_ids = class_ids.detach().to('cpu')

        for sample, class_id in zip(samples, class_ids):
            self.buffer.append((sample.detach(), class_id))

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples, device='cuda'):
        items = random.choices(self.buffer, k=n_samples)
        samples, class_ids = zip(*items)
        samples = torch.stack(samples, 0)
        class_ids = torch.tensor(class_ids)
        # class_ids = torch.stack(class_ids, 0)
        samples = samples.to(device)
        class_ids = class_ids.to(device)

        return samples, class_ids


def sample_buffer(buffer, y, batch_size=128, p=0.95, device='cuda', num_classes=100):
    
    if len(buffer) < 1:
        return (
            init_random(batch_size, 3, 32, 32).to(device),
            y,
        )

    n_replay = (np.random.rand(batch_size) < p).sum()

    replay_sample, replay_id = buffer.get(n_replay)
    random_sample = init_random(batch_size - n_replay, 3, 32, 32).to(device)
    random_id = torch.randint(0, num_classes, (batch_size - n_replay,)).to(device)

    return (
        torch.cat([replay_sample, random_sample], 0),
        torch.cat([replay_id, random_id], 0),
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



if __name__ == '__main__':

    pass
