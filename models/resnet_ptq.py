import pdb
import torch.nn as nn
import torchvision.transforms as transforms
import math
from quantize_ptq import PTQConv2d, PTQLinear
import tensorflow as tf

__all__ = ['resnet_quantized_ptq_float_bn']

FLAGS = tf.app.flags.FLAGS
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return PTQConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(3,3), stride=stride,
                   padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, PTQConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class BasicBlockPTQ(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample_conv=None, downsample_bn=None,
                 activation=None, bn_pos=None, layer_list=None, layer_name=None):
        super(BasicBlockPTQ, self).__init__()
        self.conv1 = PTQConv2d(in_channels=inplanes, out_channels=planes, kernel_size=(3,3), stride=stride,
                   padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #second conv block always has stride 1, downsampling happens in first self.conv1 layer of res block
        self.conv2 = PTQConv2d(in_channels=planes, out_channels=planes, kernel_size=(3,3), padding=1, bias=False)
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

        layer_list[layer_name + '_conv1'] = self.conv1
        layer_list[layer_name + '_bn1'] = self.bn1
        layer_list[layer_name + '_conv2'] = self.conv2
        layer_list[layer_name + '_bn2'] = self.bn2
        if self.downsample_conv is not None:
            layer_list[layer_name + '_dsconv'] = self.downsample_conv
            layer_list[layer_name + '_dsbn'] = self.downsample_bn

    def forward(self, x, is_quantized=False, n_bits_wt=8, n_bits_act=8, STE=False):
        residual = x

        if self.downsample_conv is not None:
            residual = self.downsample_conv(x, is_quantized=is_quantized, num_bits_act=n_bits_act,
                                            num_bits_wt=n_bits_wt, STE=STE)
            residual = self.downsample_bn(residual)

        out = self.conv1(x, is_quantized=is_quantized, num_bits_act=n_bits_act,
                         num_bits_wt=n_bits_wt, STE=STE)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out, is_quantized=is_quantized, num_bits_act=n_bits_act,
                         num_bits_wt=n_bits_wt, STE=STE)

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
        return out


class ResidualBlockPTQ(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
    #              dilation=1, groups=1, bias=True, biprecision=False):
    def __init__(self, block, inplanes, planes, blocks, stride=1, bn_pos=None, layer_list=None, layer_name=None):
        super(ResidualBlockPTQ, self).__init__()
        ds_conv = None
        ds_bn = None
        self.inplanes = inplanes
        self.layers = nn.ModuleList()
        if stride != 1 or self.inplanes != planes * block.expansion:
            self.ds_conv = PTQConv2d(in_channels=self.inplanes, out_channels=planes * block.expansion,
                                     kernel_size=(1, 1), stride=stride, bias=False)
            self.ds_bn = nn.BatchNorm2d(planes * block.expansion)
        else:
            self.ds_conv=None
            self.ds_bn=None

        _layer_name = layer_name + '_block1'
        self.layers.append(block(self.inplanes, planes, stride, self.ds_conv, self.ds_bn, bn_pos=bn_pos,
                                 layer_list=layer_list, layer_name=_layer_name))
        self.inplanes = planes * block.expansion

        self.blocks = blocks
        for i in range(1, self.blocks):
            _layer_name = layer_name + '_block%d' % i
            self.layers.append(block(self.inplanes, planes, bn_pos=bn_pos,
                                     layer_list=layer_list, layer_name=_layer_name))

    def forward(self, input, is_quantized=False, num_bits_act=8, num_bits_wt=8, STE=False):
        for block in self.layers:
            # pdb.set_trace()
            input = block(input, is_quantized=is_quantized, n_bits_wt=num_bits_act,
                           n_bits_act=num_bits_wt, STE=STE)
        out = input
        return out


class ResNetPTQ(nn.Module):

    def __init__(self):
        super(ResNetPTQ, self).__init__()

    def old_forward(self, x, is_quantized=False, n_bits_wt=8, n_bits_act=8, STE=False):
        x = self.conv1(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.layer2(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.layer3(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.layer4(x, is_quantized, n_bits_act, n_bits_wt, STE)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, is_quantized, n_bits_act, n_bits_wt, STE)
        # x = self.activation(x)
        # x = self.fc2(x)
        return x


    def forward(self, x, is_quantized=False, n_bits_wt=8, n_bits_act=8, STE=False):
        # pdb.set_trace()

        # if self.
        #     for name, m in self.named_modules():
        #         if 'linear' in name.lower() or 'conv' in name.lower():
        #             m.update_activation_min_max()

        x = self.conv1(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.layer2(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.layer3(x, is_quantized, n_bits_act, n_bits_wt, STE)
        x = self.layer4(x, is_quantized, n_bits_act, n_bits_wt, STE)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_end(x, is_quantized, n_bits_act, n_bits_wt, STE)
        # x = self.activation(x)
        # x = self.fc2(x)
        return x


class ResNet_cifar10_ptq(ResNetPTQ):

    def __init__(self, num_classes=10,
                 block=BasicBlockPTQ, depth=18, inflate=None, activation=None):
        super(ResNet_cifar10_ptq, self).__init__()

        self.layer_list = {}

        if inflate is None:
            self.inflate=2
        else:
            self.inflate = FLAGS.inflate

        self.noise = FLAGS.noise_model
        self.inplanes = 16 * self.inflate

        n = int((depth - 2) / 6)
        self.conv1 = PTQConv2d(in_channels=3, out_channels=16*self.inflate, kernel_size=(3,3), stride=1, padding=1,
                             bias=False)

        self.bn1 = nn.BatchNorm2d(16 * self.inflate)
        if FLAGS.activation is None or FLAGS.activation is 'relu':
            self.activation = nn.ReLU()
            FLAGS.activation='relu'
            bn_pos='pre_res'
        elif FLAGS.activation=='tanh':
            self.activation = activation
            bn_pos='post_res'

        self.maxpool = lambda x: x

        self.layer1 = ResidualBlockPTQ(block, self.inplanes, 16 * self.inflate, n, bn_pos=bn_pos,
                                       layer_list=self.layer_list, layer_name='resblock1')
        self.inplanes = 16 * self.inflate
        self.layer2 = ResidualBlockPTQ(block, self.inplanes, 32 * self.inflate, n, stride=2, bn_pos=bn_pos,
                                       layer_list=self.layer_list, layer_name='resblock2')
        self.inplanes = 32 * self.inflate
        self.layer3 = ResidualBlockPTQ(block, self.inplanes, 64 * self.inflate, n, stride=2, bn_pos=bn_pos,
                                       layer_list=self.layer_list, layer_name='resblock3')
        self.inplanes = 64 * self.inflate

        if FLAGS.activation=='tanh':
            self.layer4 = ResidualBlockPTQ(block, self.inplanes, 128 * self.inflate, n, stride=2, bn_pos=None,
                                       layer_list=self.layer_list, layer_name='resblock4')
        else:
            self.layer4 = ResidualBlockPTQ(block, self.inplanes, 128 * self.inflate, n, stride=2, bn_pos=bn_pos,
                          layer_list = self.layer_list, layer_name = 'resblock4')
            self.inplanes = 128 * self.inflate

        self.avgpool = nn.AvgPool2d(4)
        self.fc_end = PTQLinear(128 * self.inflate, num_classes)

        self.layer_list['conv1'] = self.conv1
        self.layer_list['bn1'] = self.bn1
        self.layer_list['avgpool'] = self.avgpool
        self.layer_list['fc_end'] = self.fc_end

        init_model(self)
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]

    def set_weight_min_max(self):
        for name, m in self.named_modules():
            if 'linear' in name.lower() or 'conv' in name.lower():
                m.set_quantizer_limits()


def resnet_quantized_ptq_float_bn(**kwargs):
    return ResNet_cifar10_ptq(num_classes=10, inflate=FLAGS.inflate)
