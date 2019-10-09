import torch
from torch.optim.optimizer import Optimizer, required
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
gamma = FLAGS.gamma
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
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required,betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.):
        if lr is not required and lr < 0.0:
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

        super(SGDR, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDR, self).__setstate__(state)

    def step(self, regularizer, closure=None, returns=None):
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

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data



                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.clone(d_p).detach()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']
                # print(beta1)
                # print(beta2)

                state['step'] += 1

                # Compute momentum here (exp_avg)
                exp_avg.mul_(beta1).add_(1 - beta1, d_p)

                # Compute fisher information here
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, d_p, d_p)

                # d_p=exp_avg

                if weight_decay != 0:
                    # d_p.add_(weight_decay, p.data)
                    p.data.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], exp_avg)

                if hasattr(p, 'pert') and regularizer is not None:
                    if regularizer=='l2':
                        p.data.add_(-group['lr'], p.pert)

                    elif regularizer=='fisher':
                        reg_grad = p.pert * exp_avg_sq
                        p.data.add_(-gamma * group['lr'], reg_grad)
                        #diagonal loading for fisher regularizer
                        p.data.add_(-gamma * diag_load * group['lr'], p.pert)

                    elif regularizer == 'inv_fisher':
                        FIM = exp_avg_sq

                        inv_FIM = 1 / (FIM + 1e-7)
                        inv_FIM = inv_FIM * 1e-7
                        reg_grad = inv_FIM * p.pert

                        p.data.add_(-gamma * group['lr'], p.pert)
                        # diagonal loading for fisher regularizer
                        p.data.add_(-gamma * diag_load * group['lr'], p.pert)


        return loss
