'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.util import get_act, get_norm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False, norm='none', act='relu'):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        norm_layer = get_norm(norm)
        act_layer = get_act(act)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = act_layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = self.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False, norm='none', act='relu'):
        super(Bottleneck, self).__init__()
        norm_layer = get_norm(norm)
        act_layer = get_act(act)
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(self.expansion * planes)
        self.relu = act_layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = self.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False, norm='none', act='relu', img_size=224, multiscale=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.norm_layer = get_norm(norm)
        self.act_layer = get_act(act)
        self.num_blocks = num_blocks
        self.block = block
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act, norm=norm)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=act, norm=norm)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=act, norm=norm)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, act=act, norm=norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.last_dim = 512 * block.expansion
        if multiscale:
            self.last_dim += 256 * block.expansion
            self.last_dim += 128 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def set_mid_model(self, act, norm):
        block = self.block
        num_blocks = self.num_blocks
        num_classes = self.num_classes
        self.mid_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.mid_bn1 = self.norm_layer(64)
        self.mid_layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act, norm=norm)
        self.mid_layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=act, norm=norm)
        self.mid_layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=act, norm=norm)
        self.mid_linear = nn.Linear(512 * block.expansion, num_classes)

    def mid_forward(self, x):
        out = F.relu(self.mid_bn1(self.mid_conv1(x)))
        out, _ = self.mid_layer1(out)
        out, _ = self.mid_layer2(out)
        out, _ = self.mid_layer3(out)
        # out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        # out = self.mid_linear(out)
        return out
    
    def set_small_model(self, act, norm):
        block = self.block
        num_blocks = self.num_blocks
        num_classes = self.num_classes
        self.small_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.small_bn1 = self.norm_layer(64)
        self.small_layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, act=act, norm=norm)
        self.small_layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, act=act, norm=norm)
        self.small_layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, act=act, norm=norm)
        self.small_linear = nn.Linear(512 * block.expansion, num_classes)

    def small_forward(self, x):
        out = F.relu(self.small_bn1(self.small_conv1(x)))
        out, _ = self.small_layer1(out)
        out, _ = self.small_layer2(out)
        out, _ = self.small_layer3(out)
        # out = self.avgpool(out)
        # out = out.view(out.size(0), -1)
        # out = self.small_linear(out)
        return out

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        feat_m.append(self.layer4)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]

    def _make_layer(self, block, planes, num_blocks, stride, act='relu', norm='none'):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1, act=act, norm=norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False, preact=False, multiscale=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out, f4_pre = self.layer4(out)
        f4 = out
        if multiscale:
            mid_out = self.mid_forward(x)
            small_out = self.small_forward(x)
            out = torch.cat([f4, mid_out, small_out], 1)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)
        
        if is_feat:
            if preact:
                return [[f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out]
            else:
                return [f0, f1, f2, f3, f4, f5], out
        else:
            return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    net = ResNet18(num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
