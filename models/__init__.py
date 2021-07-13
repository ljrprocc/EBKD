from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4, resnet26x10, resnet20x10, resnet32x10
# If you want to train resnet for ImageNet from sractch
from .resnetv2 import ResNet50, ResNet18
from .pretrained_resnet import resnet50, resnet18
# from torchvision.models.resnet import resnet50, resnet18
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .generator import CCF, FF, ZNEnergy, netE
from .cvae import ConditionalVAE, VanillaVAE, CVAEEncoder, CVAEDecoder

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnet20x10': resnet20x10,
    'resnet26x10': resnet26x10,
    'resnet32x10': resnet32x10,
    'ResNet50': resnet50,
    'ResNet18': resnet18,
    'ResNet50cifar100': ResNet50,
    'ResNet18cifar100': ResNet18,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'Score': netE,
    'Gen': CCF,
    'vae': VanillaVAE,
    'cvae': ConditionalVAE,
    'enc': CVAEEncoder,
    'dec': CVAEDecoder
}
