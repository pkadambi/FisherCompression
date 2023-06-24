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
            # ctx.min_value = y.min(-1)[0].mean(-1)  # C
            min_value = y.min() # C
            # min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        # else:
        #     ctx.min_value = min_value

        if max_value is None:
            # max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max()  # C
            # ctx.max_value = y.max(-1)[0].mean(-1)  # C
        # else:
        #     ctx.max_value = max_value

        ctx.noise = noise

        if quantize:
            ctx.num_bits = num_bits
            ctx.min_value = min_value
            ctx.max_value = max_value

            q_weight = input.clone()

            qmin = 0.
            qmax = 2. ** num_bits - 1.

            scale = (ctx.max_value - ctx.min_value) / (qmax - qmin)

            scale = max(scale, 1e-8)

            if FLAGS.enforce_zero:
                # TODO: Maybe delete this since this case is never used???
                initial_zero_point = qmin - ctx.min_value / scale
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
                q_weight = (q_weight - ctx.min_value) / scale + qmin

            q_weight.clamp_(qmin, qmax).round_()  # quantize
            # ctx.min_value = min_value
            # ctx.max_value = max_value

            # TODO: figure out how this works
            if FLAGS.enforce_zero:
                q_weight.add_(-zero_point).mul_(scale)  # dequantize
            else:
                q_weight.add_(-qmin).mul_(scale).add_(ctx.min_value)  # dequantize

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

class Quantize_EMA(nn.Module):
    """

    This class quantizes an input to a layer by keeping a running mean of the min and max value

    """

    #TODO: add an argument here where you can select between using an absolute minimum and a percentile

    #TODO: This slows down training by a lot, figure out why
    def __init__(self, momentum=0.1):
        super(Quantize_EMA, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum

    def forward(self, input, num_bits, STE=False):
        # print(self.training)
        if self.training:
            _input = input.view(input.shape[0], -1)
            min_value = _input.min()
            max_value = _input.max()

            self.running_min.mul_(self.momentum).add_(min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
            # print('here')
        else:
            min_value = self.running_min
            max_value = self.running_max

        # return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)
        return quantize(input, num_bits=num_bits, min_value=float(self.running_min), max_value=float(self.running_max),
                        num_chunks=16, STE=STE)


class PTQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, biprecision=False):
        super(PTQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                        dilation, groups, bias)
        self.biprecision = biprecision
        self.has_bias = bias
        self.qmin_wt = None
        self.qmax_wt = None
        self.qbias = None
        self.CALIBRATION=False
        self.quantize_input_ema = Quantize_EMA()

        # todo: min/max value
        # todo: need to set the levels correctly



    def forward(self, input, is_quantized=False, num_bits_act=8, num_bits_wt=8, STE=False):
        if is_quantized:
            if self.qmin_wt or self.qmin_wt is None:
                self.set_quantizer_limits()

        # self.qweight = quantize(self.weight, num_bits=num_bits_wt, min_value=-.3, max_value=.3, STE=STE)
        self.qweight = quantize(self.weight, num_bits=num_bits_wt, min_value=self.qmin_wt,
                                max_value=self.qmax_wt, STE=STE)
        # self.qweight = quantize(self.weight, num_bits=num_bits_wt, STE=STE)

        if self.has_bias:
            #todo; need to set minimum/maximum correctly

            # self.qbias = quantize(self.bias, num_bits=num_bits_wt, min_value=-.5, max_value=.5)
            self.qbias = self.bias
            # self.bias.pert = self.qbias - self.bias

        self.weight.pert = self.qweight - self.weight

        if self.CALIBRATION:
            _qinput = self.quantize_input_ema(input, num_bits=num_bits_act, STE=STE)

        if is_quantized:
            # qinput = quantize(input, num_bits=num_bits_act, min_value=-.3, max_value=.3, STE=STE)
            if self.CALIBRATION:
                qinput = _qinput
            else:
                qinput = quantize(input, num_bits=num_bits_act, STE=STE)
            # qinput = input
            output = F.conv2d(input=qinput, weight=self.qweight, bias=self.qbias, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=self.groups)
        else:
            output = F.conv2d(input=input, weight=self.weight, bias=self.bias, stride=self.stride,
                              padding=self.padding, dilation=self.dilation, groups=self.groups)
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

    def enable_calibration(self):
        self.CALIBRATION = True

    def disable_calibration(self):
        self.CALIBRATION = False


class PTQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(PTQLinear, self).__init__(in_features, out_features, bias)
        self.has_bias = bias

        #TODO: Add the flaggs into this
        self.qmin_wt = None
        self.qmax_wt = None
        self.qbias = None
        self.CALIBRATION=False
        self.quantize_input_ema = Quantize_EMA()

    def forward(self, input, is_quantized=False, num_bits_act=8, num_bits_wt=8, STE=False):
        if is_quantized:
            if self.qmin_wt or self.qmin_wt is None:
                self.set_quantizer_limits()

        if self.has_bias:
            # self.qbias = quantize(self.bias, num_bits=num_bits_wt, min_value=-.5, max_value=.5)
            self.qbias = self.bias
            # self.bias.pert = self.qbias - self.bias

        # self.qweight = quantize(self.weight, num_bits=num_bits_wt, min_value=-.3, max_value=.3, STE=STE)
        self.qweight = quantize(self.weight, num_bits=num_bits_wt, min_value=self.qmin_wt,
                                max_value=self.qmax_wt, STE=STE)
        # self.qweight = quantize(self.weight, num_bits=num_bits_wt, STE=STE)
        self.weight.pert = self.qweight - self.weight

        if self.CALIBRATION:
            _qinput = self.quantize_input_ema(input, num_bits=num_bits_act, STE=STE)

        if is_quantized:
            # qinput = quantize(input, num_bits=num_bits_act, min_value=-.3, max_value=.3, STE=STE)
            if self.CALIBRATION:
                qinput = _qinput

            else:
                qinput = quantize(input, num_bits=num_bits_act, STE=STE)
            # qinput = input

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

    def enable_calibration(self):
        self.CALIBRATION = True

    def disable_calibration(self):
        self.CALIBRATION = False