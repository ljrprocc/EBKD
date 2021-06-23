import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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

class MLPGenerator(nn.Module):
    def __init__(self, latent_dim=100, out_channel=3, out_size=32, num_classes=100):
        super(MLPGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.models = []
        self.models.append(nn.Linear(latent_dim+num_classes, 128))
        self.models.append(nn.LeakyReLU(0.2, True))

        for i in range(3):
            self.models.append(nn.Linear(2**(7+i), 2**(8+i)))
            self.models.append(nn.BatchNorm1d(2**(8+i)))
            self.models.append(nn.LeakyReLU(0.2, True))

        self.models.append(nn.Linear(1024, out_channel*out_size*out_size))
        self.models.append(nn.Tanh())

        self.models = nn.Sequential(*self.models)
        self.out_channel = out_channel
        self.out_size = out_size
        self.apply(weight_init('kaiming'))
        print(self.models)

    
    def forward(self, x, y):
        joint_input = torch.cat([x, self.label_embedding(y)], 1)
        # print(joint_input.shape)
        # print(self.models)
        out = self.models(joint_input)
        bs = x.shape[0]
        out = out.view(bs, self.out_channel, self.out_size, self.out_size)
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channel=3, out_size=32, num_classes=100):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        hidden_dim = 512
        self.hidden_dim = hidden_dim
        self.input_models = []
        self.input_models.append(nn.Linear(latent_dim+num_classes, hidden_dim*2*2))
        self.input_models = nn.Sequential(*self.input_models)

        depth_layer = int(math.log2(out_size)) - 1

        self.model = []
        for i in range(depth_layer - 1):
            next_dim = max(hidden_dim // 2, 64)
            self.model.append(nn.ConvTranspose2d(hidden_dim, next_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
            self.model.append(nn.BatchNorm2d(next_dim))
            self.model.append(nn.LeakyReLU(0.2, True))
            hidden_dim = next_dim

        self.model = nn.Sequential(*self.model)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        self.out_channel = out_channel
        self.out_size = out_size
        self.mu = nn.Linear(self.hidden_dim*2*2, latent_dim)
        self.logvar = nn.Linear(self.hidden_dim*2*2, latent_dim)
        print(self.input_models)
        print(self.model)
        print(self.final_layer)
        self.apply(weights_init)


    def get_mu_var(self, x):
        mu = self.mu(x)
        log_var = self.logvar(x)

        return [mu, log_var]


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps*std + mu
    
    def forward(self, x, y, return_feat=False):
        joint_input = torch.cat([x, self.label_embedding(y)], 1)
        out = self.input_models(joint_input)
        out_para = self.get_mu_var(out)
        out = out.view(-1, self.hidden_dim, 2, 2)
        out = self.model(out)
        # print(out.shape)
        out = self.final_layer(out)
        # print(out.shape)
        if return_feat:
            return out, out_para
        else:
            return out



class Blocks(nn.Module):
    def __init__(self, in_c, out_c, downsample=False, num_classes=100):
        super(Blocks, self).__init__()
        class_embed = nn.Embedding(num_classes, out_c * 2 * 2)
        class_embed.weight.data[:, : out_c * 2] = 1
        class_embed.weight.data[:, out_c * 2 :] = 0
        self.class_embed = class_embed
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 2 if downsample else 1, 1)
        # self.bn2 = nn.BatchNorm2d(out_c)
        
    def forward(self, x, y=None):
        out = x
        # print(out.shape)
        out = self.conv1(out)
        if y is not None:
            embed = self.class_embed(y).view(x.size(0), -1, 1, 1)
            # print(embed.shape)
            weight1, weight2, bias1, bias2 = embed.chunk(4, 1)
            out = weight1 * out + bias1
        # out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        if y is not None:
            out = weight2 * out + bias2
        # out = self.bn2(out)
        out = F.leaky_relu(out, 0.2)
        return out



class Energy(nn.Module):
    '''
    Modeling E_{\theta}(x, y=i)
    '''
    def __init__(self, hidden_dim=64, latent_dim=1, image_size=32, num_classes=100, joint=False):
        super(Energy, self).__init__()
        self.nef = hidden_dim
        self.latent_dim = latent_dim
        
        if joint:
            in_channel = 6
        else:
            in_channel = 3

        self.models = nn.ModuleList([
            Blocks(3, hidden_dim, downsample=True, num_classes=num_classes),
            Blocks(hidden_dim, hidden_dim, num_classes=num_classes),
            Blocks(hidden_dim, hidden_dim*2, downsample=True, num_classes=num_classes),
            Blocks(hidden_dim*2, hidden_dim*2, num_classes=num_classes),
            Blocks(hidden_dim*2, hidden_dim*4, downsample=True, num_classes=num_classes),
            Blocks(hidden_dim*4, hidden_dim*8, downsample=True, num_classes=num_classes),
        ])

        self.flatten = nn.Sequential(
            nn.Linear((image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
            nn.Softplus()
        )
        self.n_cls = num_classes
        self.score = nn.Linear(hidden_dim*8, self.latent_dim)
        self.classi = nn.Linear(hidden_dim*8, self.n_cls)
        self.image_size = image_size
        
        print(self)
        # self.apply(weight_init('kaiming'))


    def forward(self, x, y=None):
        # if y is not None:
        #     z = self.label_embedding(y)
        #     z = self.label_mlp(z)
        #     z = z.view(x.size(0), 3, self.image_size, self.image_size)
        #     x = torch.cat([x, z], 1)
        h = x
        for i, model in enumerate(self.models):
            h = model(h, y)
        h = h.view(h.shape[0], h.shape[1], -1).sum(2)
        h = self.flatten(h)
        score = self.score(h)
        return score

    def classify(self, x):
        h = x
        for i, model in enumerate(self.models):
            h = model(h)
        # print(h.shape)
        h = h.view(h.shape[0], -1)
        # print(h.shape)
        h = self.flatten(h)
        logit = self.classi(h)
        return logit


class CCG(Energy):
    def __init__(self, hidden_dim=64, latent_dim=1, image_size=32, num_classes=100, joint=False):
        super(CCG, self).__init__(hidden_dim, latent_dim, image_size, num_classes, joint)
    
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)
        else:
            return torch.gather(logits, 1, y[:, None])


class FF(nn.Module):
    def __init__(self, model, n_cls=10):
        super(FF, self).__init__()
        self.f = model
        self.mlp_cls_head = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(self.f.last_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
        )
        self.cls_head = nn.Linear(64, n_cls)
        self.energy_output = nn.Linear(64, 1)
        # self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.n_cls = n_cls
        # self.is_feat = is_feat

    def forward(self, x, y=None, cls_mode=False, is_feat=False, preact=False, return_feat=False):
        if cls_mode:
            if is_feat:
                feats, penult_z = self.f(x, is_feat=is_feat, preact=preact)
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
        self.logvar_fc = nn.Linear(self.f.last_dim, 1)
        self.mu_fc = nn.Linear(self.f.last_dim, 100)
        # self.is_feat = is_feat

    def kl_div(self, z, mu, std):
        p = torch.distributions.Normal(torch.zero_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def forward(self, x, y=None, cls_mode=False, is_feat=False, preact=False, return_kl=False, py=None):
        #print(cls_mode, is_feat, preact, y)
        
        feats, logits = super().forward(x, y=None, cls_mode=True, is_feat=True, preact=preact)
        
        # feat = feats[-1]
        if py is not None:
            logits = py.log() / 10 * logits
        if cls_mode:
            # print(is_feat)
            if not is_feat:
                return logits
            else:
                return feats, logits
        return_list = []
        if y is not None:
            pxy = torch.gather(logits, 1, y[:, None])
        else:
            px = torch.log(logits.exp().sum(1))
        if is_feat:
            if y is None:
                return_list = [feats, px]
            else:
                return_list = [feats, pxy]
        else:
            # logits = super().forward(x=x, y=y, cls_mode=False)
            # print(logits.requires_grad)
            if y is None:
                return_list =[px]
            else:
                return_list =[pxy]

        if return_kl:
            log_var = self.logvar_fc(F.relu(feats[-1]))
            mu = self.mu_fc(F.relu(feats[-1]))
            # std = torch.exp(log_var / 2)
            # q = torch.distributions.Normal(mu, std)
            # z = q.rsample()
            # kl_loss = self.kl_div(z, mu, std)
            return_list.append((mu, log_var))
            # return_list.append(log_var)
        return return_list


