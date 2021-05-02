from __future__ import print_function
import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
# import torch.autograd.Function as Function



def expectation(X, y, T=1):
    '''
    X: Energy logit of model. shape: (B, K)
    y: Given probability distribution of dirichlet.  shape: (K, )
    Calculate E_{y~Dir(alpha)}[f(x)].
    Here f(x) = e^{x}
    '''
    f_x = torch.exp(-X / T) # Shape: (B, K)
    # print(torch.mean(f_x))
    y = y.unsqueeze(0)
    E_y_x = torch.sum(f_x * y, 1) + 0.0000001
    return E_y_x

def expectation_delta(E_t, E_s, y, T=1):
    '''
    E_t: Energy logit of teacher model. shape: (B, K)
    E_s: Energy logit of student model. shape: (B, K)
    y: Given probability distribution of dirichlet.  shape: (K, )
    Calculate E_{y~Dir(alpha)}[f(x)].
    Here f(E_t, E_s) = e^{E_t}(E_t - E_s + log)
    '''
    # print(E_t / T)
    f_x = torch.exp(-E_t / T)*(E_t - E_s)
    y = y.unsqueeze(0)
    E_y_x = torch.sum(f_x * y, 1) + 0.00000001
    return E_y_x


class EBKD(nn.Module):
    '''
    Energy-Based Knowledge Distillation
    Optimize the learnt distribution by student and teacher by Energy-based Models. The learnt energy map/value also shows the performance of our method.
    mode: the update sampling method for the updation of energy function, including:
    - mcmc: Langevin MCMC
    - sm: Score matching. Only support Sliced Score Matching now.
    - nce: Noise Contrastive Estimation
    implementing JEM.
    '''
    def __init__(self, T):
        super(EBKD, self).__init__()
        self.T = T
        # self.num_classes = num_classes
        # self.mode = mode
        # )
    
    def forward(self, f_s, f_t):
        '''
        f_s: logit of student model, get E_s(x)
        f_t: logit of teacher model, get E_t(x)
        '''
        # print(f_s.shape, f_t.shape)
        p_s = F.log_softmax(f_s/self.T, dim=1)
        p_t = F.softmax(f_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


    def getEnergy(self, f_s, f_t):
        with torch.no_grad():
            energy_t = self.model(f_t)
            energy_s = self.model(f_s)
        return energy_s, energy_t


class EBLoss(nn.Module):
    def __init__(self, n_cls=100, T=4, teacher_path=None):
        super(EBLoss, self).__init__()
        self.embed = nn.Embedding(n_cls, n_cls)
        sim_matrix = create_similarity(teacher_path, scale=0.1)
        self.y = getDirichl(100, num_classes=n_cls, sim_matrix=sim_matrix, scale=1.)
        self.y = torch.mean(self.y, 0)
        # self.label = torch.arange(n_cls)
        self.criterion = nn.NLLLoss()
    
    def forward(self, Elogit, target):
        E_y_given_x, E_x_y = self.get_logit(Elogit, target)
        # print(E_y_given_x)
        # Get the loss
        loss = self.criterion(torch.log(E_y_given_x), target)
        if torch.isnan(loss):
            print('Error: Nan loss detected.')
            exit(-1)
        return loss, E_y_given_x

    
    def _cosine_attention(self, label_embedding, Elogit):
        le_normed = label_embedding / torch.norm(label_embedding, dim=-1).unsqueeze(-1)
        E_normed = Elogit / torch.norm(Elogit, dim=-1).unsqueeze(-1)
        # print(E_normed)
        dot_res = torch.matmul(E_normed, le_normed.T)
        Elogit = torch.matmul(dot_res, le_normed)
        return Elogit


    def get_logit(self, Elogit, target=None):
        # self.label = self.label.to(Elogit.device)
        # print(torch.softmax(Elogit, 1))
        if target is not None:
            label_embedding = self.embed(target) # (B, K)
            Elogit = self._cosine_attention(label_embedding, Elogit)
            Elogit /= 0.2
        # Get joint distribution, of EBM
        # Unnormalized
        # print(self.y.unsqueeze(1).shape, Elogit.shape)
        E_x_y = torch.exp(Elogit) / self.y.unsqueeze(0)
        # Normalized
        E_x_y = torch.softmax(Elogit, 0)
        # print(E_x_y)
        # Get posterior distribution
        E_y_given_x = E_x_y / (torch.sum(E_x_y, -1).unsqueeze(-1))
        return E_y_given_x, E_x_y
