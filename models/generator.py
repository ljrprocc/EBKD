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
        # self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.n_cls = n_cls
        # self.is_feat = is_feat

    def forward(self, x, y=None, cls_mode=False, is_feat=False, preact=False, return_feat=False, z=None):
        if cls_mode:
            if is_feat:
                feats, penult_z = self.f(x, is_feat=is_feat, preact=preact, z=z)
                # print(penult_z.requires_grad)
                # print(self.f(x, is_feat=is_feat, preact=preact))
                return feats, penult_z
            else:
                penult_z = self.f(x, is_feat=is_feat, preact=preact)
                return penult_z
        else:
            feats, penult_z = self.f(x, is_feat=True)
            # print(feats[-1].requires_grad)
            if not feats[-1].requires_grad:
                feat = feats[-1].detach()
            else:
                feat = feats[-1]
            ori_feat = feat
            # feat = self.mlp_cls_head(ori_feat)
            if return_logit:
                return ori_feat, self.energy_output(feat).squeeze()
            return self.energy_output(feat).squeeze()


class CCF(FF):
    def __init__(self, model, n_cls=10):
        super(CCF, self).__init__(model=model, n_cls=n_cls)

    def forward(self, x=None, y=None, cls_mode=False, is_feat=False, preact=False, py=None, z=None):
        
        feats, logits = super().forward(x, y=None, cls_mode=True, is_feat=True, preact=preact, z=z)
        
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
            ne = torch.gather(logits, 1, y[:, None])
        else:
            ne = torch.log(logits.exp().sum(1))
        if is_feat:
            return_list = [feats, ne]
        else:
            return_list = [ne]
        return return_list


class ZNEnergy(CCF):
    def __init__(self, model, n_cls=10):
        super(ZNEnergy, self).__init__(model=model, n_cls=n_cls)
    
    def forward(self, z, y=None, py=None):
        return_list = super().forward(x=None, y=y, cls_mode=False, is_feat=False, preact=False, py=py, z=z)
        return return_list

class netE(nn.Module):
    """
    discriminator is based on x and z jointly
    x is first go through conv-layers
    z is first go through conv-layers
    then (x, z) is concatenate along the axis (both ndf*4), then go through several layers
    """
    def __init__(self, nc, nz, nez, ndf=64, z_mode=True, x_mode=True):
        # FOR CIFAR10
        super(netE, self).__init__()
        self.z_mode = z_mode
        self.x_mode = x_mode

        # spectral_norm = lambda x: x
        if x_mode:
            self.x = nn.Sequential(
                spectral_norm(nn.Conv2d(nc, ndf, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf, ndf, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf, ndf*2, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*2, ndf*2, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*2, ndf*4, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*8, ndf*8, 4, 1, 0)),
            )

        ## z
        if z_mode:
            self.z = nn.Sequential(

                spectral_norm(nn.Conv2d(nz, ndf, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf, ndf , 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf, ndf * 2, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*2, ndf * 2, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf *4, ndf * 4, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*8, ndf*8, 1, 1, 0)),
            )
        if x_mode and z_mode:

            self.xz = nn.Sequential(

                spectral_norm(nn.Conv2d(ndf*16 , ndf*16, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*16, nez, 1, 1, 0)),

            )
        elif x_mode and (not z_mode):
            self.x_single = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf*8, ndf*8, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*8, nez, 1, 1, 0))
            )
        elif z_mode and (not x_mode):
            self.z_single = nn.Sequential(
                spectral_norm(nn.Conv2d(ndf*8, ndf*8, 1, 1, 0)),
                nn.LeakyReLU(0.1, inplace=True),
                spectral_norm(nn.Conv2d(ndf*8, nez, 1, 1, 0))
            )
        else:
            raise ValueError('Must support at least one mode of x or z.')

    def forward(self, x=None, z=None, leak=0.1):
        if self.x_mode:
            ox = self.x(x)
        if self.z_mode:
            oz = self.z(z)
        if self.z_mode and self.x_mode:
            oxz = torch.cat([ox, oz], 1)
            oE_outxz = self.xz(oxz)
        elif self.x_mode and (not self.z_mode):
            oE_outxz = self.x_single(ox)
        elif self.z_mode and (not self.x_mode):
            oE_outxz = self.z_single(oz)
        return oE_outxz