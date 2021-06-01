from __future__ import print_function

import torch
import numpy as np
import random
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.dirichlet import Dirichlet
import torchvision.utils as vutils
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

class TVLoss(torch.nn.Module):
    def __init__(self, tv_weight=1):
        super(TVLoss, self).__init__()
        self.tv_weight = tv_weight
    
    def forward(self, img):
        w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2), (1,2,3))
        h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2), (1,2,3))
        loss = self.tv_weight * (h_variance + w_variance)
        return loss


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
            # print(p[0])
            p[1].requires_grad = flag

def print_trainable_paras(net):
    for p in net.named_parameters():
        if p[1].requires_grad:
            print(p[0])

if __name__ == '__main__':

    pass
