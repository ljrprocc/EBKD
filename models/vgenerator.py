import torch
from torch import nn
import torch.nn.functional as F

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)

def get_activation(name, args):
    return {'gelu': GELU(), 'lrelu': nn.LeakyReLU(args.e_activation_leak), 'mish': Mish(), 'swish': Swish()}[name]

class _netG(nn.Module):
    def __init__(self, args):
        super().__init__()

        f = get_activation(args.g_activation, args)

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf*16, 4, 1, 0, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*16, args.ngf*8, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.gnf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*8, args.ngf*4, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.gnf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.gnf*2) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*1) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*1, args.nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)