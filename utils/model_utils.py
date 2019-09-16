import torch
import torchvision
import torch.nn as nn
# import sklearn.cluster.k_means
from sklearn.cluster import k_means
import numpy as np
import logging.config
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
def quantizer_levels_from_wts(model, n_levels):
    """
    This function takes in a model and tries to find the levels of the quantizer by performing k-means on the weights
    :param model: trained pytorch model
    :param n_levels: number of levels in the quantizer
    :return: k-means scipy object that has the levels of the quantizer
    """
    params = []
    for p in model.parameters():
        p_arr = p.cpu().numpy()
        params.append(p_arr.ravel())

    params = np.hstack(params)

    return k_means(params.reshape(-1, 1), n_levels)

from collections import namedtuple
QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().float()
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}
def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer

def test_model(data_loader, model, criterion, printing=True, eta=None):

    #switch to eval mode
    print('Evaluating Model...')
    model.eval()


    n_test = 0.
    n_correct = 0.
    loss = 0.

    for iter, (inputs, target) in enumerate(data_loader):
        n_batch = inputs.size()[0]

        inputs = inputs.cuda()
        target = target.cuda()

        if eta is not None:
            output = model(inputs, eta=eta)
        else:
            output = model(inputs)

        loss += criterion(output, target).item()
        acc_ = accuracy(output, target)

        n_correct += accuracy(output, target).item() * n_batch/100.
        n_test += n_batch

    test_accuracy= 100*n_correct/n_test
    test_loss = 128*loss/n_test

    if printing:
        print('Test Accuracy %.3f'% test_accuracy)
        print('Test Loss %.3f'% test_loss)

    #Revert model to training mode before exiting
    model.train()

    return test_loss, test_accuracy


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def xavier_initialize_weights(network):
    if type(network)==nn.Linear or type(network)==nn.Conv2d:
        nn.init.xavier_uniform_(network.weight)

def normal_initialize_biases(network):
    if type(network)==nn.Linear or type(network)==nn.Conv2d:
        nn.init.normal_(network.bias, std=.1)

def constant_initialize_bias(network):
    if type(network)==nn.Linear or type(network)==nn.Conv2d:
        nn.init.constant_(network.bias, val=.001)