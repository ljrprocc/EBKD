from __future__ import print_function
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
# import torch.autograd.Function as Function

def sample_langevin(x, model, stepsize, n_steps, noise_scale=None, intermediate_samples=False):
    if noise_scale is None:
        noise_scale = np.sqrt(stepsize*2)
    # x.requires_grad = True
    if not x.requires_grad:
        raise
    for i in range(n_steps):
        noise = torch.randn_like(x) * noise_scale
        # noise = noise.to(x.device)
        # in original model, p_{\theta}(x) represents another seperate model
        # By analyzing the logits produced by teacher and student model separately, we treat the soft target
        # Treat student model as energy model.
        # of teacher output as real_data distribution, and here p_{\theta} represents the learnt distribution
        # of student model.
        # The original code:
        #     out = model(x)
        out = model(x)
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        # print(torch.mean(grad))
        dynamics = stepsize*grad + noise
        x = x + dynamics
        # print(i)

    return x.detach().to(x.device)
# class EBMSampling(nn.Module):
#     def __init__(self):
#         super(EBMSampling, self).__init__()



def sliced_score_estimation_vr(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    # print(grad1)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

def naive_score_matching(score_net, data_net, sample):
    # sample.requires_grad = True
    if not sample.requires_grad:
        sample.requires_grad_(True)
    pos_output = torch.sum(-data_net(sample), -1).sum()
    grad_pos = autograd.grad(pos_output, sample, create_graph=True)[0]

    neg_output = torch.sum(-score_net(sample), -1).sum()
    grad_neg = autograd.grad(neg_output, sample, create_graph=True)[0]

    loss = (grad_pos-grad_neg) ** 2 / 2.
    loss = loss.mean(0).mean()
    return loss


# def implicit_sample_distribution(distribution, inputs):
#     '''
#     Implement the sample operation x~p_t with p_t  
#     '''


class EnergyKD(nn.Module):
    '''
    Energy-Based Knowledge Distillation
    Optimize the learnt distribution by student and teacher by Energy-based Models. The learnt energy map/value also shows the performance of our method.
    mode: the update sampling method for the updation of energy function, including:
    - mcmc: Langevin MCMC
    - sm: Score matching. Only support Sliced Score Matching now.
    - nce: Noise Contrastive Estimation
    '''
    def __init__(self, T, mode='sm', alpha=1, num_classes=10, stepsize=0.1, n_step=100, energy_model=None, te_model=None, image_size=32):
        super(EnergyKD, self).__init__()
        self.T = T
        self.mode = mode
        self.alpha = alpha
        self.stepsize = stepsize
        self.n_steps = n_step
        self.model = energy_model
        self.te_model = te_model
        self.linear_block = nn.Sequential(
            *[
                nn.Linear(256, image_size**2 * 3),
                # nn.LeakyReLU(0.1, True),
                # nn.Linear(512, ),
                nn.Tanh()
            ]
        )
        # Energy model to save 
        # self.model = nn.Sequential(
        #     *[nn.Linear(256, 256), 
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(128, 1)]
        # )
        self.init = True
        self.image_size = image_size
        for p in self.te_model.parameters():
            p.requires_grad = False
    
    def forward(self, f_s, f_t):
        # print(f_s.shape, f_t.shape)
        
        if self.mode == 'mcmc':
            if self.init:
                self.init = False
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            pos_out = self.model(f_t)
            neg_out = self.model(f_s)
            loss = (pos_out - neg_out) + self.alpha * (pos_out ** 2 + neg_out ** 2)
            # print(torch.mean(pos_out), torch.mean(neg_out))
            loss = loss.mean()
        elif self.mode == 'sm':
            f_t_project = self.linear_block(f_t).view(f_t.shape[0], 3, self.image_size, self.image_size)
            f_t = (f_s + f_t_project) / 2.
            # print(f_t.shape)
            # loss, *_ = sliced_score_matching_vr(self.model, f_t,  1)
            loss = naive_score_matching(self.model, self.te_model, f_t)
            # print(loss*120)
            # print(loss*120)
        return loss * 120

    def getEnergy(self, f_s, f_t):
        with torch.no_grad():
            energy_t = self.model(f_t)
            energy_s = self.model(f_s)
        return energy_s, energy_t
        

        
        
