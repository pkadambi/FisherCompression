
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
# class UniformQuantize(torch.autograd.Function):
# class UniformQuantize(nn.Module):

    # @classmethod

    @staticmethod
    # def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, eta = .01,
    #             noise=None, num_chunks=None, out_half=False):
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None, eta=.01,
                    noise=None, num_chunks=None, out_half=False):

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

##
        ctx.noise = noise

        if quantize:

            ctx.num_bits = num_bits
            ctx.min_value = min_value
            ctx.max_value = max_value
            # print(input)
            # exit()
            q_weight = input.clone()

            qmin = 0.
            qmax = 2. ** num_bits - 1.
            #import pdb; pdb.set_trace()
            scale = (max_value - min_value) / (qmax - qmin)

            scale = max(scale, 1e-8)

            if FLAGS.enforce_zero:
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

                # output.div_(scale).add_(zero_point)
                q_weight = (q_weight / scale) + zero_point

            else:
                # output.add_(-min_value).div_(scale).add_(qmin)
                q_weight = (q_weight - min_value) / scale + qmin

            q_weight.clamp_(qmin, qmax).round_()  # quantize
            ctx.min_value = min_value
            ctx.max_value = max_value

            #TODO: figure out how this works
            if FLAGS.enforce_zero:
                q_weight.add_(-zero_point).mul_(scale)  # dequantize
            else:
                q_weight.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

            if out_half and num_bits <= 16:
                q_weight = q_weight.half()

        else:
            # If layer is not quantized, we still need to compute
            # some type of min-max statistics on the weight kernel for noising
            q_weight=input


        # if FLAGS.regularization is not None:
        #     pert = input - q_weight
        #     ctx.save_for_backward(pert)


        return q_weight

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator

        # if FLAGS.regularization=='l2':
        #     pert = ctx.saved_tensors[0]
        #     grad_input = grad_output + FLAGS.gamma * FLAGS.diag_load_const * 2 * pert
        # elif FLAGS.regularization=='fisher':
        #     pert = ctx.saved_tensors[0]
        #
        #     # grad_input = grad_output + torch.clamp(FLAGS.gamma *
        #     #                                        2 * (grad_output * grad_output *
        #     #                                        pert + FLAGS.diag_load_const * pert),
        #     #                                        -.1, .1)
        #
        #     grad_input = grad_output + FLAGS.gamma * 2 * (grad_output * grad_output * pert + FLAGS.diag_load_const * pert)
        #
        #
        # else:
        #     grad_input = grad_output
        grad_input = grad_output

        # return grad_input
        return grad_input, None, None, None, None, None, None, None, None


class NoiseInjection(torch.autograd.Function):
    '''
        Perform noise injection with straight-through estimator
    '''

    @staticmethod
    def forward(ctx, input, eta, max_value, min_value):

        # TODO: change this to it's own layer

        output = input.clone()

        if ctx.noise == 'NVM':
            noise_std = eta * (max_value - min_value)

            noise = output.new(input.shape).normal_(mean=0, std=noise_std)

            output.add_(noise)

        elif ctx.noise == 'PCM':
            # TODO: correct implementation of PCM noise model (send email to paul about how to do this)
            noise_std = eta * (max_value - min_value)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        #straight through estimator
        grad_input = grad_output.clone()
        return grad_input

def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
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


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, noise=None, eta=.0, out_half=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, eta, noise, num_chunks, out_half)
    # quant = UniformQuantize()
    # return quant(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)

def tensor_clamp(input_tensor, min_value_tensor, max_value_tensor):
    '''
    This function re-implements the tensor.clamp() since tensor.clamp(min,max)
    requires that min and max are numbers (not tensors)

    :return:
    '''

    return torch.max(torch.min(input_tensor, max_value_tensor), min_value_tensor)



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

        if self.training and FLAGS.regularization is None:
        # if self.training:
            # std = torch.std(input.detach().view(-1))
            # mean = torch.mean(input.detach().view(-1))

            # min_value = mean - 3 * std
            # max_value = mean + 3 * std

            # n_elems = torch.numel(input)
            #
            # ind_min = int(n_elems * .001)
            # ind_max = n_elems - ind_min

            # if self.num_bits<=2:

                # min_value = -.5
                # max_value = .5

            # else:

                # std = torch.std(input.detach().view(-1))
                # mean = torch.mean(input.detach()
                # .view(-1))

                # min_value = mean - 3 * std
                # max_value = mean + 3 * std

            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean()

                # min_value = torch.kthvalue(input.view(-1), ind_min)[0]
                # max_value = torch.kthvalue(input.view(-1), ind_max)[0]

            self.running_min.mul_(self.momentum).add_(min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max

        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, is_quantized,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits_act=8,
                 num_bits_weight=8, biprecision=False, noise=None):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.num_bits_act = num_bits_act
        self.num_bits_weight = num_bits_weight
        self.is_quantized = is_quantized


        if self.num_bits_act<32 and is_quantized:
            self.quantize_input = QuantMeasure(self.num_bits_act)

        self.biprecision = biprecision
        self.noise = noise

        if FLAGS.q_min is not None:
            self.min_value = torch.tensor(FLAGS.q_min, device='cuda')
        else:
            self.min_value = self.weight.min()

        if FLAGS.q_max is not None:
            self.max_value = torch.tensor(FLAGS.q_max, device='cuda')
        else:
            self.max_value = self.weight.max()

    #TODO: add inputs eta, noise model (eta kept in formward
    def forward(self, input, eta=0.):

        #Eta is kept as an input to the forward() function since eta_train=\=eta_inf sometimes

        if self.is_quantized:


            if self.num_bits_act < 32:
                qinput = self.quantize_input(input)
            else:
                qinput = input


            if FLAGS.q_min is None and FLAGS.regularization is None:
                self.min_value = self.weight.min()
                # print(self.min_value)

            if FLAGS.q_max is None and FLAGS.regularization is None:
                self.max_value = self.weight.max()


            # self.weight.data.clamp(self.min_value.detach().cpu().numpy(), self.max_value.detach().cpu().numpy())
            # self.weight.data = tensor_clamp(self.weight, self.min_value, self.max_value)

            self.qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.min_value),
                               max_value=float(self.max_value), noise=self.noise, eta=eta)



            #TODO: ADD NOISING FUNCTION HERE


            if self.bias is not None:
                self.qbias = quantize(self.bias, num_bits=self.num_bits_weight)

            else:
                self.qbias = None



            output = F.conv2d(qinput, self.qweight, self.qbias, self.stride,
                                  self.padding, self.dilation, self.groups)

        else:

            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output


    def set_min_max(self):
        self.min_value = self.weight.min()
        self.max_value = self.weight.max()

class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, is_quantized, bias=True, num_bits_act=8,
                 num_bits_weight=8, noise=None):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.num_bits_act = num_bits_act
        self.num_bits_weight = num_bits_weight
        self.is_quantized = is_quantized

        if self.num_bits_act<32 and is_quantized:
            self.quantize_input = QuantMeasure(self.num_bits_act)

        self.noise = noise

        if FLAGS.q_min is not None:
            self.min_value = torch.tensor(FLAGS.q_min, device='cuda')
        else:
            self.min_value = self.weight.min()

        if FLAGS.q_max is not None:
            self.max_value = torch.tensor(FLAGS.q_max, device='cuda')
        else:
            self.max_value = self.weight.max()

    def forward(self, input, eta=0.):

        if self.is_quantized:
            if self.num_bits_act < 32:
                qinput = self.quantize_input(input)
            else:
                qinput = input


            if FLAGS.q_min is None and FLAGS.regularization is None:
                self.min_value = self.weight.min()

            if FLAGS.q_max is None and FLAGS.regularization is None:
                self.max_value = self.weight.max()

            # self.weight.data.clamp(self.min_value, self.max_value)
            # self.weight.data = tensor_clamp(self.weight, self.min_value, self.max_value)

            self.qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.min_value),
                               max_value=float(self.max_value), noise=self.noise, eta=eta)

            #TODO: choose whether to quantize bias
            if self.bias is not None:
                self.qbias = quantize(self.bias, num_bits=self.num_bits_weight)
            else:
                self.qbias = None


            output = F.linear(qinput, self.qweight, self.qbias)

        else:

            output = F.linear(input, self.weight, self.bias)

        return output


    def set_min_max(self):
        self.min_value = self.weight.min()
        self.max_value = self.weight.max()
