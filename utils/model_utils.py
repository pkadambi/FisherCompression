#TODO: merge this with utils.py from the lyme project and somehow decide how to split what you need from the lyme project
#TODO: create an ML Utils package that has what you need
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
import torch.nn.functional as F
from torch.autograd import Variable


def loss_fn_kd(student_logits, teacher_logits, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs from student and teacher
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """


    teacher_soft_logits = F.softmax(teacher_logits / T, dim=1)

    teacher_soft_logits = teacher_soft_logits.float()
    student_soft_logits = F.log_softmax(student_logits/T, dim=1)


    #For KL(p||q), p is the teacher distribution (the target distribution), and
    KD_loss = nn.KLDivLoss(reduction='batchmean')(student_soft_logits, teacher_soft_logits)
    KD_loss = (T ** 2) * KD_loss
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)

    return KD_loss

def loss_fn_smooth_labels(model_logits, target_smooth_labels):

    model_probs = F.log_softmax(model_logits)
    smooth_label_loss = nn.KLDivLoss(reduction='batchmean')(model_probs, target_smooth_labels)

    return smooth_label_loss


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

def update_lr(epoch, optimizer, scheduler=None, decay_method='cosine'):

    if decay_method=='cosine':
            scheduler.step()

    elif decay_method=='step':

        if epoch<60:
            for g in optimizer.param_groups:
                g['lr'] = 0.1

        elif epoch<120:
            for g in optimizer.param_groups:
                g['lr'] = 0.02

        elif epoch<160:
            for g in optimizer.param_groups:
                g['lr'] = 0.004
        elif epoch < 200:
            for g in optimizer.param_groups:
                g['lr'] = 0.0008

import pdb
def test_model_ptq(data_loader, model, criterion, printing=True, topk=1, is_quantized=False,
                   n_bits_wt=8, n_bits_act=8):

    if printing:
        print('Evaluating Model...')

    model.eval()

    n_test = 0.
    n_correct = 0.
    loss = 0.
    # pdb.set_trace()
    for iter, (inputs, target) in enumerate(data_loader):
        n_batch = inputs.size()[0]
        # print(iter)
        inputs = inputs.cuda().float()
        target = target.cuda()        # pdb.set_trace()
        output = model(inputs, is_quantized=is_quantized, n_bits_act=n_bits_act, n_bits_wt=n_bits_wt)

        loss += criterion(output, target).item()
        # acc_ = accuracy(output, target)

        n_correct += accuracy(output, target, topk=(topk,)).item() * n_batch / 100.
        n_test += n_batch


    test_accuracy = 100 * n_correct / n_test
    test_loss = 128 * loss / n_test

    if printing:
        print('Test Accuracy %.3f' % test_accuracy)
        print('Test Loss %.3f' % test_loss)

    # Revert model to training mode before exiting
    model.train()
    # pdb.set_trace()

    return test_loss, test_accuracy

def model_output_distribution(data_loader, model, eta=None, temperature=1):
    model.eval()
    from torch.distributions import Categorical

    # print('Mean Shannon Entropy: ' )
    # print('Median Shannon Entropy: ')
    # print('')


    shannon_entropies = []
    n_test = 0.
    n_correct = 0.
    loss = 0.

    sfmx = torch.nn.Softmax()


    for iter, (inputs, target) in enumerate(data_loader):
        n_batch = inputs.size()[0]

        inputs = inputs.cuda()
        target = target.cuda()

        if eta is not None:
            output = model(inputs, eta=eta)
        else:
            output = model(inputs)


        temperature_probs = sfmx(output.detach().cpu() / temperature)
        # pdb.set_trace()
        # loss += loss_criterion(output, target).item()
        _shannonent = Categorical(probs = temperature_probs).entropy()
        # pdb.set_trace()
        shannon_entropies+=_shannonent
        n_correct += accuracy(output, target).item() * n_batch / 100.
        n_test += n_batch

    test_accuracy = 100 * n_correct / n_test
    test_loss = 128 * loss / n_test

    print('Test Accuracy %.3f' % test_accuracy)
    print('Test Loss %.3f' % test_loss)

    return np.array(shannon_entropies)





def test_model(data_loader, model, criterion, printing=True, eta=None, teacher_model=None, topk=1):

    #switch to eval mode
    if printing:
        print('Evaluating Model...')
    model.eval()


    n_test = 0.
    n_correct = 0.
    loss = 0.
    kl_loss = 0.
    for iter, (inputs, target) in enumerate(data_loader):
        n_batch = inputs.size()[0]
        # print(iter)
        inputs = inputs.cuda()
        target = target.cuda()

        if eta is not None:
            output = model(inputs, eta=eta)
        else:
            output = model(inputs)

        loss += criterion(output, target).item()
        # acc_ = accuracy(output, target)

        n_correct += accuracy(output, target, topk=(topk,)).item() * n_batch/100.
        n_test += n_batch

        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
                kl_loss += loss_fn_kd(output, teacher_output, T=1)
                # print(kl_loss)
            # print(kl_loss.size())
            # exit()

    test_accuracy= 100*n_correct/n_test
    test_loss = 128*loss/n_test
    kl_loss = 128*kl_loss/n_test



    if printing:
        print('Test Accuracy %.3f'% test_accuracy)
        print('Test Loss %.3f'% test_loss)

    #Revert model to training mode before exiting
    model.train()

    if teacher_model is None:
        return test_loss, test_accuracy
    else:
        return test_loss, test_accuracy, kl_loss


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

def to_one_hot(labels, n_classes, lowest_class_num=0):
    '''
    :param labels: should be size (n_samples, 1), values in the array should be integers or integer valued floats (will be cast)
    :param lowest_class_num: the value of the lowest class label (ie whether first class is labeled 0 or 1)
    :return: one hot encoded array size (n_samples, n_classes)
    '''
    n_samples = np.shape(labels)[0]
    # print(n_samples)
    # print(n_classes)
    one_hot_array = np.zeros([n_samples, n_classes])

    # if the lowest class number is
    labels = labels - lowest_class_num
    labels.astype('int')
    one_hot_array[np.arange(n_samples), labels.T.astype(int)] = 1
    return one_hot_array

def xavier_initialize_weights(network):
    if type(network)==nn.Linear or type(network)==nn.Conv2d:
        nn.init.xavier_uniform_(network.weight)

def normal_initialize_biases(network):
    if type(network)==nn.Linear or type(network)==nn.Conv2d:
        nn.init.normal_(network.bias, std=.1)

def constant_initialize_bias(network):
    if type(network)==nn.Linear or type(network)==nn.Conv2d:
        nn.init.constant_(network.bias, val=.001)