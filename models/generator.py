import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm
import math
from .widerresnet import Wide_ResNet

def weight_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


class FF(nn.Module):
    def __init__(self, model, n_cls=10):
        super(FF, self).__init__()
        self.f = model
        # self.mlp_cls_head = nn.Sequential(
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(self.f.last_dim, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(0.2),
        # )
        self.cls_head = nn.Linear(self.f.last_dim, n_cls)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        # self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.n_cls = n_cls
        # self.is_feat = is_feat

    def classify(self, x, is_feat=False, preact=False):
        if is_feat:
            feats, penult_z = self.f(x, is_feat=is_feat, preact=preact)
            # print(penult_z.requires_grad)
            # print(self.f(x, is_feat=is_feat, preact=preact))
            return feats, penult_z
        else:
            penult_z = self.f(x, is_feat=is_feat, preact=preact)
            return penult_z
    
    def forward(self, x):
        feats, penult_z = self.f(x, is_feat=True)
        # print(feats[-1].requires_grad)
        if not feats[-1].requires_grad:
            feat = feats[-1].detach()
        else:
            feat = feats[-1]
        # feat = self.mlp_cls_head(ori_feat)
        return self.energy_output(feat).squeeze()


class CCF(FF):
    def __init__(self, model, n_cls=10):
        super(CCF, self).__init__(model=model, n_cls=n_cls)
        # self.is_feat = is_feat

    def forward(self, x, y=None, cls_mode=False, is_feat=False, preact=False, py=None):
        #print(cls_mode, is_feat, preact, y)
        
        feats, logits = super().classify(x, is_feat=True, preact=preact)
        
        # feat = feats[-1]
        if py is not None:
            logits = py.log() + logits
        if cls_mode:
            # print(is_feat)
            if not is_feat:
                return logits
            else:
                return feats, logits
        return_list = []
        if y is not None:
            p = torch.gather(logits, 1, y[:, None])
        else:
            p = logits.logsumexp(1)
            
        if is_feat:
            return_list = [feats, p]
        else:
            return_list = [p]
     
        return return_list
