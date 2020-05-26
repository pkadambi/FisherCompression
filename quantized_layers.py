
import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import math
import pdb

FLAGS = tf.app.flags.FLAGS
USING_RELU = FLAGS.activation=='relu'
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
    def forward(ctx, input, num_bits=8, min_value=None, max_value=None, num_chunks=None, is_quantized=False):

        if is_quantized:

            ctx.num_bits = num_bits
            ctx.min_value = min_value
            ctx.max_value = max_value

            # q_weight = input.clone()
            q_weight = input

            qmin = 0.
            qmax = 2. ** num_bits - 1.
            #import pdb; pdb.set_trace()
            scale = (ctx.max_value - ctx.min_value) / (qmax - qmin)

            scale = max(scale, 1e-8) #avoids qunatizing all bits to one level

            q_weight = (q_weight - ctx.min_value) / scale + qmin

            q_weight.clamp_(qmin, qmax).round_()  # quantize


            q_weight.add_(-qmin).mul_(scale).add_(ctx.min_value)  # dequantize

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

        grad_input = grad_output

        return grad_input, None, None, None, None, None, None, None, None


# class NoiseInjection(torch.autograd.Function):
class NoiseInjection(InplaceFunction):
    '''
        Perform noise injection with straight-through estimator
    '''
    @staticmethod
    def forward(ctx, input, eta, max_value, min_value):

        # output = input.clone()
        output = input

        noise_std = eta * (max_value - min_value)

        noise = output.new(input.shape).normal_(mean=0, std=noise_std)

        output.add_(noise)

        # elif ctx.noise == 'PCM':
        # TODO: correct implementation of PCM noise model (send email to paul about how to do this)
        # noise_std = eta * (max_value - min_value)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        #straight through estimator
        grad_input = grad_output.clone()
        return grad_input, None, None, None

def noise_weight(input, eta, min_value, max_value):
    return NoiseInjection().apply(input, eta, min_value, max_value)

def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, noise=None, eta=.0, out_half=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, eta, num_chunks, out_half)
    # quant = UniformQuantize()
    # return quant(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)

def tensor_clamp(input_tensor, min_value_tensor, max_value_tensor):
    '''
    This function re-implements the tensor.clamp() since tensor.clamp(min,max)
    requires that min and max are numbers (not tensors)

    :return:
    '''

    return torch.max(torch.min(input_tensor, max_value_tensor), min_value_tensor)



class Quantize(nn.Module):
    """

    This class quantizes an input to a layer by keeping a running mean of the min and max value

    """

    #TODO: add an argument here where you can select between using an absolute minimum and a percentile

    #TODO: This slows down training by a lot, figure out why
    def __init__(self, num_bits=8, momentum=0.1):
        super(Quantize, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum

    def forward(self, input, num_bits, quantize):

        if FLAGS.regularization is None:

            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean()

            self.running_min.mul_(self.momentum).add_(min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))

        else:
            min_value = self.running_min
            max_value = self.running_max

        if quantize:
            #TODO: binary case
            return quantize(input, num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)
        else:
            return input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, is_quantized,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits_act=8,
                 num_bits_weight=8, biprecision=False, noise=None, quant_input=True):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.num_bits_act = num_bits_act
        self.num_bits_weight = num_bits_weight
        self.is_quantized = is_quantized

        self.quant_inp = quant_input

        # if self.num_bits_act<32 and is_quantized and self.quant_inp:

        self.biprecision = biprecision
        self.noise = noise

        with torch.no_grad():
            self.q_min = None
            if FLAGS.q_min is not None:
                # self.min_value = torch.tensor(FLAGS.q_min, device='cuda')
                self.register_buffer('running_min', FLAGS.q_min)
                self.q_min = FLAGS.q_min
            else:
                self.register_buffer('running_min', self.weight.min())


            self.q_max = None
            if FLAGS.q_max is not None:
                self.register_buffer('running_max', FLAGS.q_max)
                self.q_max = FLAGS.q_max
            else:
                self.register_buffer('running_max', self.weight.min())


        # if FLAGS.q_max is None and FLAGS.n_bits_wt<=2:
        #     self.max_value=1
        #
        # if FLAGS.q_min is None and FLAGS.n_bits_wt<=2:
        #     self.min_value=-1

        self.d_theta = None

    #TODO: add inputs eta, noise model (eta kept in formward
    def forward(self, input, eta=0.):

        if self.is_quantized:

            #TODO: incorporate the running min style here (w/momentum)
            #  see if it's better than straight min/max
            if self.q_min is None and FLAGS.regularization is None and self.training:
                # self.running_min.add_(self.weight.min() - self.running_min)
                self.running_min = self.weight.min()


            if self.q_max is None and FLAGS.regularization is None and self.training:
                # self.running_max.add_(self.weight.max() - self.running_max)
                self.running_max = self.weight.max()



            if self.quant_inp:
                if self.num_bits_act<4 and USING_RELU:
                    qinput =  quantize(input, num_bits=self.num_bits_act,
                                       min_value=0,
                                       max_value=2.*float(self.running_max), noise=self.noise, eta=eta)
                elif self.num_bits_act < 32:
                    qinput = self.quantize_input(input)
                # if self.num_bits_act < 32:
                #     qinput = self.quantize_input(input)
                else:
                    qinput = input
            else:
                qinput = input

            # self.weight.data.clamp(self.min_value.detach().cpu().numpy(), self.max_value.detach().cpu().numpy())
            # self.weight.data = tensor_clamp(self.weight, self.min_value, self.max_value)

            if FLAGS.regularization:
                # pdb.set_trace()
                self.weight.data = tensor_clamp(self.weight.data, self.running_min, self.running_max)

            self.qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                                    min_value=float(self.running_min),
                                    max_value=float(self.running_max), noise=self.noise, eta=eta)

            if FLAGS.loss_surf_eval_d_qtheta and self.d_theta is not None:
                self.qweight = self.qweight + self.d_theta


            if self.bias is not None:
                self.qbias = quantize(self.bias, num_bits=self.num_bits_weight)
            else:
                self.qbias = None

            if eta>0:
                self.qweight_n = noise_weight(input=self.qweight, eta=torch.tensor(eta), max_value=self.running_min, min_value=self.running_max)
                output = F.conv2d(qinput, self.qweight_n, self.qbias, self.stride,
                                  self.padding, self.dilation, self.groups)
            else:

                output = F.conv2d(qinput, self.qweight, self.qbias, self.stride,
                                  self.padding, self.dilation, self.groups)

            if FLAGS.q_min is not None:
                self.weight.clamp(FLAGS.q_min, FLAGS.q_max)


        else:

            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return output

    #TODO: remove this function since now using a buffer for the code (it's still here incase code breaks)
    def set_min_max(self):
        with torch.no_grad():
            self.weight.min_value = self.weight.min()
            self.weight.max_value = self.weight.max()

            self.weight.min_value = self.weight.min_value.cuda()
            self.weight.max_value = self.weight.max_value.cuda()

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

        with torch.no_grad():
            self.q_min = None
            if FLAGS.q_min is not None:
                # self.min_value = torch.tensor(FLAGS.q_min, device='cuda')
                self.register_buffer('running_min', FLAGS.q_min)
                self.q_min = FLAGS.q_min
            else:
                self.register_buffer('running_min', self.weight.min())


            self.q_max = None
            if FLAGS.q_max is not None:
                self.register_buffer('running_max', FLAGS.q_max)
                self.q_max = FLAGS.q_max
            else:
                self.register_buffer('running_max', self.weight.min())
        # else:
        #     self.weight.max_value = self.weight.max()
        #     self.q_max = None

        # if FLAGS.q_max is None and FLAGS.n_bits_wt<=2:
        #     self.max_value=1
        #
        # if FLAGS.q_min is None and FLAGS.n_bits_wt<=2:
        #     self.min_value=-1
        self.d_theta = None
    def forward(self, input, eta=0.):

        if self.is_quantized:

            # if self.weight.min_value is None:
            #     self.weight.min_value = self.weight.min()
            #     self.q_min = None

            # if self.weight.max_value is None:
            #     self.weight.max_value = self.weight.max()
            #     self.q_max = None

            if self.num_bits_act<4 and USING_RELU:
                qinput = quantize(input, num_bits=self.num_bits_act,
                                  min_value=0,
                                  max_value=2.*float(self.running_max), noise=self.noise, eta=eta)
            elif self.num_bits_act < 32:
                qinput = self.quantize_input(input)
            # if self.num_bits_act < 32:
            #     qinput = self.quantize_input(input)
            else:
                qinput = input

            #Code for learning min/max
            if self.q_min is None and FLAGS.regularization is None and self.training:
                self.running_min = self.weight.min()

            if self.q_max is None and FLAGS.regularization is None and self.training:
                self.running_max = self.weight.max()


            if FLAGS.regularization:
                # self.weight.data.tensor_clamp(self.min_value, self.max_value)
                self.weight.data = tensor_clamp(self.weight.data, self.running_min, self.running_max)

            self.qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=float(self.running_min),
                               max_value=float(self.running_max), noise=self.noise, eta=eta)

            if FLAGS.loss_surf_eval_d_qtheta and self.d_theta is not None:
                self.qweight = self.qweight + self.d_theta

            if self.bias is not None:
                self.qbias = quantize(self.bias, num_bits=self.num_bits_weight)
            else:
                self.qbias = None

            if FLAGS.q_min is not None:
                self.weight.clamp(FLAGS.q_min, FLAGS.q_max)

            if eta>0:
                self.qweight_n = noise_weight(input=self.qweight, eta=torch.tensor(eta), max_value=self.running_min, min_value=self.running_max)
                output = F.linear(qinput, self.qweight_n, self.qbias)
            else:

                output = F.linear(qinput, self.qweight, self.qbias)

        else:

            output = F.linear(input, self.weight, self.bias)

        return output

    #TODO: remove this function since now using a buffer for the code (it's still here incase code breaks)
    def set_min_max(self):
        # self.weight.max_value = self.weight.min()
        # self.weight.min_value = self.weight.min()

        with torch.no_grad():

            self.weight.min_value = self.weight.min()
            self.weight.max_value = self.weight.max()

            self.weight.min_value = self.weight.min_value.cuda()
            self.weight.max_value = self.weight.max_value.cuda()
