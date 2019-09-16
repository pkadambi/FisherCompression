import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS

def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


class UniformQuantize(InplaceFunction):
# class UniformQuantize(nn.Module):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, eta = .01,
                stochastic=False, enforce_true_zero=False, num_chunks=None, out_half=False):

        num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)

        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            # min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())

        if max_value is None:
            # max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C

        if quantize:

            ctx.num_bits = num_bits
            ctx.min_value = min_value
            ctx.max_value = max_value
            ctx.stochastic = stochastic

            output = input.clone()

            qmin = 0.
            qmax = 2. ** num_bits - 1.
            #import pdb; pdb.set_trace()
            scale = (max_value - min_value) / (qmax - qmin)

            scale = max(scale, 1e-8)

            if enforce_true_zero:
                initial_zero_point = qmin - min_value / scale
                zero_point = 0.
                # make zero exactly represented
                #TODO: Figure out how on earth this works
                if initial_zero_point < qmin:
                    zero_point = qmin
                elif initial_zero_point > qmax:
                    zero_point = qmax
                else:
                    zero_point = initial_zero_point
                zero_point = int(zero_point)
                output.div_(scale).add_(zero_point)
            else:
                output.add_(-min_value).div_(scale).add_(qmin)

            output.clamp_(qmin, qmax).round_()  # quantize

        else:
            # If layer is not quantized, we still need to compute
            # some type of min-max statistics on the weight kernel for noising
            ctx.min_value = min_value
            ctx.max_value = max_value

        if ctx.stochastic:
            #TODO: implement pcm noise model (add flag to choose between them)

            noise_std = eta * (max_value - min_value)

            noise = output.new(output.shape).normal_(mean=0, std=noise_std)

            output.add_(noise)

        #TODO: figure out how this works
        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)  # dequantize
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

        if out_half and num_bits <= 16:
            output = output.half()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        # return grad_input
        return grad_input, None, None, None, None, None, None



def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)

    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)

    return out1 + out2 - out1.detach()


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)
    # quant = UniformQuantize()
    # return quant(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)



class QuantMeasure(nn.Module):
    """

    This class quantizes an input to a layer by keeping a running mean of the min and max value

    """


    #TODO: add an argument here where you can select between using an absolute minimum and a percentile

    #TODO: This slows down training by a lot, figure out why
    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(
                input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(
                input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(
                min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(
                max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max

        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, is_quantized,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.is_quantized = is_quantized

        self.quantize_input = QuantMeasure(self.num_bits)

        self.biprecision = biprecision

    #TODO: add inputs eta, noise model
    def forward(self, input):


        #HERE: specifcy a way to calculate minimum and maximum for weight

        if self.is_quantized:
            qinput = self.quantize_input(input)
            qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()))
            self.qweight = qweight.clone()

            if self.bias is not None:
                qbias = quantize(self.bias, num_bits=self.num_bits_weight)

            else:
                qbias = None

            if not self.biprecision or self.num_bits_grad is None:
                output = F.conv2d(qinput, qweight, qbias, self.stride,
                                  self.padding, self.dilation, self.groups)
            else:
                output = conv2d_biprec(qinput, qweight, qbias, self.stride,
                                       self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)

        else:

            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, is_quantized, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.biprecision = biprecision

        self.is_quantized = is_quantized
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):

        qinput = self.quantize_input(input)

        if self.is_quantized:
            qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()))
            # print(qweight)
            if self.bias is not None:
                qbias = quantize(self.bias, num_bits=self.num_bits_weight)
            else:
                qbias = None

            if not self.biprecision:
                output = F.linear(qinput, qweight, qbias)

            else:
                output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)

        else:

            output = F.linear(input, self.weight, self.bias)

        return output
