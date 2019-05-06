import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
from quantize import quantize
import numpy as np


def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       import pdb; pdb.set_trace()
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output

def Quantize(tensor,quant_mode='det',  params=None, numBits=8):
    tensor.clamp_(-2**(numBits-1),2**(numBits-1))
    if quant_mode=='det':
        tensor=tensor.mul(2**(numBits-1)).round().div(2**(numBits-1))
    else:
        tensor=tensor.mul(2**(numBits-1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2**(numBits-1))
        quant_fixed(tensor, params)
    return tensor

import torch.nn._functions as tnnf


class QuantizeLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias = True, n_bits=1,  n_bitwt = 1, minval = -1, maxval = 1):
        super(QuantizeLinear, self).__init__(in_features, out_features, bias)
        self.n_bits = n_bits
        self.n_bits_wt = n_bitwt
        self.maxval = maxval
        self.minval = minval

    def forward(self, input):

        if input.size(1) != 784:
            input.data=quantize(input.data, num_bits=self.n_bits, min_value=self.minval, max_value=self.maxval)

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        if not hasattr(self.weight, 'binary_pass'):
            self.weight.quantize_pass = True

        if self.weight.quantize_pass:
            self.weight.data=quantize(self.weight.org, num_bits=self.n_bits_wt, min_value=self.minval, max_value=self.maxval)

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                                      stride, padding, bias, n_bits=1, n_bitwt = 1, minval = -1, maxval = 1):
        super(QuantizeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.n_bits = n_bits
        self.n_bits_wt = n_bitwt
        self.maxval = maxval
        self.minval = minval

    def forward(self, input):
        if input.size(1) != 3:
            input.data = quantize(input.data, num_bits=self.n_bits, min_value=self.minval, max_value=self.maxval)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        if not hasattr(self.weight, 'binary_pass'):
            self.weight.binary_pass = True

        if self.weight.binary_pass:
            self.weight.data=quantize(self.weight.org, num_bits=self.n_bits_wt, min_value=self.minval, max_value=self.maxval)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
