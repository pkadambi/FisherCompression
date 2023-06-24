import torch
from torch.optim.optimizer import Optimizer, required
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
gamma_target = FLAGS.gamma
gamma = gamma_target
diag_load = FLAGS.diag_load_const

class SGDR(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}
        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDR, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, regularizer = None, closure=None, return_reg_val=False, gamma = FLAGS.gamma):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # import pdb
            # pdb.set_trace()
            if return_reg_val:
                reg_val=torch.tensor(0.).cuda()

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # d_p = p.grad.data.clamp_(-.1,.1)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                #TODO: reformat such that fisher calc isn't under the momentum area
                if momentum != 0:

                    param_state = self.state[p]

                    #estimating fisher information
                    if 'exp_avg_sq' not in param_state:
                        param_state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg_sq = param_state['exp_avg_sq']
                    exp_avg_sq.mul_(.999).addcmul_(1 - .999, d_p, d_p)

                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                FIM_estimate = exp_avg_sq + diag_load

                if hasattr(p, 'pert') and gamma==gamma_target:
                    # if pause:
                    #     pdb.set_trace()
                    if regularizer=='l2':
                        p.data.add_(gamma * group['lr'], -p.pert)
                    elif regularizer=='fisher' or regularizer=='gradual_fisher':

                        reg_grad = -p.pert * FIM_estimate
                        p.data.add_(gamma * group['lr'], reg_grad)


                    elif regularizer == 'inv_fisher':
                        # pdb.set_trace()
                        inv_FIM = 1 / (FIM_estimate)
                        inv_FIM = inv_FIM * diag_load


                        # inv_FIM = 1/FIM_estimate

                        reg_grad = inv_FIM * (-p.pert)

                        p.data.add_(gamma * group['lr'], reg_grad)
                        # diagonal loading for fisher regularizer
                p.data.add_(-group['lr'], d_p)

                # if hasattr(p, 'pert') and gamma>0.:
                #
                #     if regularizer=='l2':
                #         p.data.add_(-gamma * group['lr'], p.pert)
                #
                #     elif 'fisher' in regularizer:
                #         # import pdb
                #         # pdb.set_trace()
                #         # reg_grad = p.pert * (exp_avg_sq/torch.max(exp_avg_sq))
                #
                #         if FLAGS.layerwise_fisher:
                #             p.fisher = p.fisher / torch.max(p.fisher)
                #         reg_grad = p.pert * p.fisher
                #         reg_val = reg_val + torch.sum(p.pert * reg_grad)
                #
                #         # reg_grad = reg_grad.clamp(-.1, .1)
                #
                #         p.data.add_(-gamma * group['lr'], reg_grad)
                #
                #         #diagonal loading for fisher regularizer
                #         p.data.add_(-gamma * diag_load * group['lr'], p.pert)
                #
                #     elif regularizer == 'inv_fisher':
                #         # FIM = exp_avg_sq
                #         # # FIM = p.grad.data * p.grad.data
                #         #
                #         # inv_FIM = 1 / (FIM + 1e-7)
                #         # inv_FIM = inv_FIM * 1e-7
                #         reg_grad = p.inv_FIM * p.pert
                #         reg_val = reg_val + torch.sum(p.pert * reg_grad)
                #
                #         p.data.add_(-gamma * group['lr'],  reg_grad)
                #
                #         # diagonal loading for fisher regularizer
                #         # p.data.add_(-gamma * diag_load * group['lr'], p.pert)

        # return loss

        if return_reg_val:
            return loss, reg_val
        else:
            return loss

