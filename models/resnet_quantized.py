import torch.nn as nn
import torchvision.transforms as transforms
import math
from quantize import QConv2d, QLinear
import tensorflow as tf

__all__ = ['resnet_quantized_float_bn']

FLAGS = tf.app.flags.FLAGS


n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act

def conv3x3(is_quantized, in_planes, out_planes, stride=1):
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

    def __init__(self, is_quantized, inplanes, planes, stride=1, downsample_conv=None, downsample_bn=None,
                 activation=None, bn_pos=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(is_quantized, inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(is_quantized, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample_conv = downsample_conv
        self.downsample_bn = downsample_bn
        self.stride = stride

        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation

        if bn_pos=='pre_res':
            self.pre_res_bn=True
            self.post_res_bn=False
        elif bn_pos=='post_res':
            self.pre_res_bn = False
            self.post_res_bn = True
        else:
            self.pre_res_bn = False
            self.post_res_bn = False

    def forward(self, x):
        residual = x

        if self.downsample_conv is not None:
            residual = self.downsample_conv(x)
            residual = self.downsample_bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)

        if self.post_res_bn:
            out += residual
            out = self.bn2(out)
            out = self.activation(out)
        elif self.pre_res_bn:
            out = self.bn2(out)
            out += residual
            out = self.activation(out)
        else:
            out += residual

        # if self.do_bntan:
        #     out = self.bn2(out)
        #     out = self.tanh2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, is_quantized, inplanes, planes, stride=1, downsample=None):
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
        self.relu = nn.ReLU()
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

    def _make_layer(self, block, planes, blocks, stride=1, bn_pos=None):
        ds_conv = None
        ds_bn = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_conv = QConv2d(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=(1,1),
                        stride=stride, bias=False, is_quantized=self.is_quantized, num_bits_weight=n_bits_wt,
                        num_bits_act=n_bits_act)

            ds_bn = nn.BatchNorm2d(planes * block.expansion)

        layers = []
        layers.append(block(self.is_quantized, self.inplanes, planes, stride, ds_conv, ds_bn, bn_pos=bn_pos))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.is_quantized, self.inplanes, planes, bn_pos=bn_pos))

        return nn.Sequential(*layers)

    def forward(self, x, eta=0.):
        x = self.conv1(x, eta)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10,
                 block=BasicBlock, depth=18, is_quantized=None, inflate=None, activation=None):
        super(ResNet_cifar10, self).__init__()

        if n_bits_wt<=2 and is_quantized:
            if inflate is None:
                self.inflate=4
            else:
                self.inflate = FLAGS.inflate
        else:
            self.inflate = 2

        if is_quantized is None:
            self.is_quantized = FLAGS.is_quantized
        else:
            self.is_quantized = is_quantized

        self.noise = FLAGS.noise_model

        self.inplanes = 16 * self.inflate

        n = int((depth - 2) / 6)
        self.conv1 = QConv2d(in_channels=3, out_channels=16*self.inflate, kernel_size=(3,3), stride=1, padding=1,
                             bias=False, is_quantized=self.is_quantized, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)

        self.bn1 = nn.BatchNorm2d(16 * self.inflate)

        if FLAGS.activation is None:
            self.activation = nn.ReLU()
            bn_pos='pre_res'
        elif FLAGS.activation=='tanh':
            self.activation = activation
            bn_pos='post_res'
            # print('IM HERE')
            # exit()

        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16 * self.inflate, n, bn_pos=bn_pos)
        self.layer2 = self._make_layer(block, 32 * self.inflate, n, stride=2, bn_pos=bn_pos)
        self.layer3 = self._make_layer(block, 64 * self.inflate, n, stride=2, bn_pos=bn_pos)

        if FLAGS.activation=='tanh':
            self.layer4 = self._make_layer(block, 128 * self.inflate, n, stride=2, bn_pos=None)
        else:
            self.layer4 = self._make_layer(block, 128 * self.inflate, n, stride=2, bn_pos=bn_pos)

        self.avgpool = nn.AvgPool2d(4)
        self.fc = QLinear(128 * self.inflate, num_classes, is_quantized=self.is_quantized, noise=self.noise,
                           num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]

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
