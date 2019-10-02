import torch.nn as nn
import torchvision.transforms as transforms
import math
from quantize import QConv2d, QLinear
import tensorflow as tf

__all__ = ['resnet_quantized_float_bn']

FLAGS = tf.app.flags.FLAGS


NUM_BITS = 8
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act
is_quantized = FLAGS.is_quantized

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(3,3), stride=stride,
                   padding=1, bias=False, is_quantized=is_quantized, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)


def init_model(model):
    for m in model.modules():
        if isinstance(m, QConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = QConv2d(in_channels=inplanes, out_channels=planes, kernel_size=(1,1), stride=stride,
                             bias=False, is_quantized=is_quantized, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(in_channels=planes, out_channels=planes, kernel_size=(3,3), stride=stride,
                             padding=1, bias=False, is_quantized=is_quantized,
                             num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d(in_channels=planes, out_channels=planes * 4, kernel_size=(1,1), bias=False,
                             is_quantized=is_quantized, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv2d(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=(1,1),
                        stride=stride, bias=False, is_quantized=self.is_quantized, num_bits_weight=n_bits_wt,
                        num_bits_act=n_bits_act),

                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, eta=0.):
        x = self.conv1(x, eta)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = QConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                             bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT,
                             num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = QLinear(512 * block.expansion, num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT,
                          num_bits_grad=NUM_BITS_GRAD)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 30, 'lr': 1e-2},
            {'epoch': 60, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 90, 'lr': 1e-4}
        ]


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()

        if n_bits_wt<=2:
            self.inflate = 5

        elif n_bits_wt <= 4:
            self.inflate = 2
        else:
            self.inflate = 1

        self.is_quantized = FLAGS.is_quantized
        self.noise = FLAGS.noise_model

        self.inplanes = 16 * self.inflate

        n = int((depth - 2) / 6)
        self.conv1 = QConv2d(in_channels=3, out_channels=16*self.inflate, kernel_size=(3,3), stride=1, padding=1,
                             bias=False, is_quantized=self.is_quantized, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)

        self.bn1 = nn.BatchNorm2d(16 * self.inflate)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16 * self.inflate, n)
        self.layer2 = self._make_layer(block, 32 * self.inflate, n, stride=2)
        self.layer3 = self._make_layer(block, 64 * self.inflate, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = QLinear(64 * self.inflate, num_classes, is_quantized=self.is_quantized, noise=self.noise,
                           num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]


def resnet_quantized_float_bn(**kwargs):
    #TODO:fix for imagenet (and resnet34)

    # num_classes, depth, dataset = map(
    #     kwargs.get, ['num_classes', 'depth', 'dataset'])
    # if dataset == 'imagenet':
    #     num_classes = num_classes or 1000
    #     depth = depth or 50
    #     if depth == 18:
    #         return ResNet_imagenet(num_classes=num_classes,
    #                                block=BasicBlock, layers=[2, 2, 2, 2])
    #     if depth == 34:
    #         return ResNet_imagenet(num_classes=num_classes,
    #                                block=BasicBlock, layers=[3, 4, 6, 3])
    #     if depth == 50:
    #         return ResNet_imagenet(num_classes=num_classes,
    #                                block=Bottleneck, layers=[3, 4, 6, 3])
    #     if depth == 101:
    #         return ResNet_imagenet(num_classes=num_classes,
    #                                block=Bottleneck, layers=[3, 4, 23, 3])
    #     if depth == 152:
    #         return ResNet_imagenet(num_classes=num_classes,
    #                                block=Bottleneck, layers=[3, 8, 36, 3])
    #
    # elif dataset == 'cifar10':
    #     num_classes = num_classes or 10
    #     depth = depth or 56
    #     return ResNet_cifar10(num_classes=num_classes,
    #                           block=BasicBlock, depth=depth)

    return ResNet_cifar10(num_classes=10, block=BasicBlock)
