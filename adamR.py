import math
import torch
from torch.optim.optimizer import Optimizer
import tensorflow as tf
import matplotlib.pyplot as plt
FLAGS = tf.app.flags.FLAGS
gamma_target = FLAGS.gamma
gamma = gamma_target
diag_load = FLAGS.diag_load_const
import pdb
from debug_helper_funcs import *
# print(gamma_target)
# exit()
class AdamR(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamR, self).__setstate__(state)

    def step(self, regularizer = None, closure=None, return_reg_val=False, gamma=gamma, pause=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                # grad = p.grad.data.clamp_(-.1,.1)
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # step_size = group['lr'] / bias_correction1

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # if pause:
                #     print()
                #     pdb.set_trace()

                FIM_estimate = exp_avg_sq + diag_load

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # if p has a perturbation assigned to it, it is regularized


                #TODO: Add regularization here
                if hasattr(p, 'pert') and gamma==gamma_target:
                    # if pause:
                    #     pdb.set_trace()
                    if regularizer=='l2':
                        p.data.add_(gamma * group['lr'], p.pert)
                    elif regularizer=='fisher' or regularizer=='gradual_fisher':

                        reg_grad = p.pert * FIM_estimate
                        p.data.add_(gamma * group['lr'], reg_grad)


                    elif regularizer == 'inv_fisher':
                        # pdb.set_trace()
                        inv_FIM = 1 / (FIM_estimate)
                        inv_FIM = inv_FIM * diag_load


                        # inv_FIM = 1/FIM_estimate

                        reg_grad = inv_FIM * p.pert

                        p.data.add_(gamma * group['lr'], reg_grad)
                        # diagonal loading for fisher regularizer

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
