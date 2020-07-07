import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pdb

FLAGS = tf.app.flags.FLAGS
USING_RELU = FLAGS.activation=='relu'


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, noise=None, eta=.0, out_half=False, STE=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, eta, noise, num_chunks, out_half, STE)

class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits, min_value=None, max_value=None, eta=0., noise=None, num_chunks=None,
                out_half=False, STE=False):

        num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)

        # pdb.set_trace()
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            # min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())

        if max_value is None:
            # max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C

        ctx.noise = noise

        if quantize:
            ctx.num_bits = num_bits
            ctx.min_value = min_value
            ctx.max_value = max_value

            q_weight = input.clone()

            qmin = 0.
            qmax = 2. ** num_bits - 1.

            scale = (max_value - min_value) / (qmax - qmin)

            scale = max(scale, 1e-8)

            if FLAGS.enforce_zero:
                # TODO: Maybe delete this since this case is never used???
                initial_zero_point = qmin - min_value / scale
                zero_point = 0.

                # make zero exactly represented
                # TODO: Figure out how on earth this works
                if initial_zero_point < qmin:
                    zero_point = qmin
                elif initial_zero_point > qmax:
                    zero_point = qmax
                else:
                    zero_point = initial_zero_point

                zero_point = int(zero_point)

                q_weight = (q_weight / scale) + zero_point

            else:
                q_weight = (q_weight - min_value) / scale + qmin

            q_weight.clamp_(qmin, qmax).round_()  # quantize
            ctx.min_value = min_value
            ctx.max_value = max_value

            # TODO: figure out how this works
            if FLAGS.enforce_zero:
                q_weight.add_(-zero_point).mul_(scale)  # dequantize
            else:
                q_weight.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

            if out_half and num_bits <= 16:
                q_weight = q_weight.half()

        else:
            # If layer is not quantized, we still need to compute
            # some type of min-max statistics on the weight kernel for noising
            q_weight = input

        ctx.STE=STE

        return q_weight

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.STE:
            grad_input = grad_output
            return grad_input, None, None, None, None, None, None, None, None
        else:
            return None, None, None, None, None, None, None, None, None


class PTQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, biprecision=False):

        self.has_bias = bias

        super(PTQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                        dilation, groups, bias)

        self.stride = stride
        self.padding = padding
        self.qmin_wt = None
        self.qmax_wt = None
        self.qbias = None

    def forward(self, input, is_quantized=False, num_bits_act=8, num_bits_wt=8, STE=False):
        if is_quantized:
            if self.qmin_wt or self.qmin_wt is None:
                self.set_quantizer_limits()

        #todo: min/max value
        #todo: need to set the levels correctly

        self.qweight = quantize(self.weight, num_bits=num_bits_wt, min_value=-.3, max_value=.3, STE=STE)
        # self.qweight = quantize(self.weight, num_bits=num_bits_wt, STE=STE)

        if self.has_bias:
            #todo; need to set minimum/maximum correctly

            # self.qbias = quantize(self.bias, num_bits=num_bits_wt, min_value=-.5, max_value=.5)
            self.qbias = self.bias
            # self.bias.pert = self.qbias - self.bias

        self.weight.pert = self.qweight - self.weight

        if is_quantized:
            # qinput = quantize(input, num_bits=num_bits_act, min_value=-.3, max_value=.3, STE=STE)
            qinput = quantize(input, num_bits=num_bits_act, STE=STE)
            output = F.conv2d(qinput, self.qweight, self.qbias, self.stride, self.padding, self.dilation, self.groups)

        else:
            output = F.conv2d(input=input, weight=self.weight, bias=self.bias, stride=self.stride,
                              padding=self.padding, groups=self.groups)
        return output


    def set_quantizer_limits(self, qmin=None, qmax=None):
        if qmax is not None:
            self.qmax_wt = np.percentile(self.weight.detach().cpu(), 99.5)
        else:
            self.qmax_wt = qmax
        if qmin is not None:
            self.qmin_wt = np.percentile(self.weight.detach().cpu(), .5)
        else:
            self.qmin_wt = qmin

class PTQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        self.has_bias = bias
        super(PTQLinear, self).__init__(in_features, out_features, bias)

        #TODO: Add the flaggs into this
        self.qmin_wt = None
        self.qmax_wt = None
        self.qbias = None


    def forward(self, input, is_quantized=False, num_bits_act=8, num_bits_wt=8, STE=False):
        if is_quantized:
            if self.qmin_wt or self.qmin_wt is None:
                self.set_quantizer_limits()

        if self.has_bias:
            # self.qbias = quantize(self.bias, num_bits=num_bits_wt, min_value=-.5, max_value=.5)
            self.qbias = self.bias
            # self.bias.pert = self.qbias - self.bias

        self.qweight = quantize(self.weight, num_bits=num_bits_wt, min_value=-.3, max_value=.3, STE=STE)
        # self.qweight = quantize(self.weight, num_bits=num_bits_wt, STE=STE)
        self.weight.pert = self.qweight - self.weight



        if is_quantized:
            # qinput = quantize(input, num_bits=num_bits_act, min_value=-.3, max_value=.3, STE=STE)
            qinput = quantize(input, num_bits=num_bits_act, STE=STE)

            output = F.linear(qinput, self.qweight, self.qbias)
        else:
            # print(self.weight.shape)
            output = F.linear(input, self.weight, self.bias)

        return output


    def set_quantizer_limits(self, qmin=None, qmax=None):
        if qmax is not None:
            self.qmax_wt = np.percentile(self.weight.detach().cpu(), 99.5)
        else:
            self.qmax_wt = qmax
        if qmin is not None:
            self.qmin_wt = np.percentile(self.weight.detach().cpu(), .5)
        else:
            self.qmin_wt = qmin

