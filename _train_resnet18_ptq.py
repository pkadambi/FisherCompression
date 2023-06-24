import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os
import math
import pdb
from debug_helper_funcs import *

tf.app.flags.DEFINE_string('source_dataset', 'cifar10', 'can be any lenet trainable dataset')
tf.app.flags.DEFINE_string('target_dataset', 'notmnist', 'can be any of the datasets')
tf.app.flags.DEFINE_integer('n_classes', 10, 'num_classes')

tf.app.flags.DEFINE_integer('record_interval', 100, 'iterations between printing to the console')
tf.app.flags.DEFINE_integer('n_runs', 5, 'number of runs')
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode')

# quantization flags
tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add: None, NVM, PCM')
tf.app.flags.DEFINE_float('q_min', None, 'maximum qunatizer value')
tf.app.flags.DEFINE_float('q_max', None, 'minimum of quantizer value')
tf.app.flags.DEFINE_string('regularization', None,
                           'type of regularization to use `l2,` `fisher` `distillation` or `inv_fisher`')
tf.app.flags.DEFINE_boolean('enforce_zero', False, 'Whether or not to enforce zero level in quantizer')
tf.app.flags.DEFINE_boolean('is_quantized', True, 'quantized or not?')
tf.app.flags.DEFINE_integer('n_bits_act', 4, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 4, 'number of bits wt')
tf.app.flags.DEFINE_boolean('ste', False, 'whether or not to use STE while doing quantized regularization')

# regularization flags
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation multiplier')
tf.app.flags.DEFINE_float('temperature', 4.0, 'distillation temperature')
tf.app.flags.DEFINE_float('gamma', 1e-4, 'regularizer gamma')
tf.app.flags.DEFINE_float('diag_load_const', 1e-1, 'diagonal loading constant')
tf.app.flags.DEFINE_boolean('layerwise_fisher', True, 'diagonal loading constant')
tf.app.flags.DEFINE_string('fisher_method', 'adam','which method to use when computing fisher')

# model flags
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/cifar10/Resnet18/fp_updated/Run0/resnet', 'path to FP model for loading for distillation')
tf.app.flags.DEFINE_string('fpmodel_path', './ptq_models/resnet18/', 'directory to save the fp model to ')
tf.app.flags.DEFINE_string('savepath', None, 'directory to save to')
tf.app.flags.DEFINE_string('loadpath', './SavedModels/LenetPTQ/8ba_8bw/', 'directory to load from')
tf.app.flags.DEFINE_string('activation', 'relu', '`tanh` or `relu`')
tf.app.flags.DEFINE_integer('inflate', None, 'inflating factor for resnet (may need bigger factor if 1-b weights')

# Optimization flags
tf.app.flags.DEFINE_float('lr', .1, 'learning_rate')
tf.app.flags.DEFINE_float('lr_end', 2e-4, 'learning rate at end of cosine decay')
tf.app.flags.DEFINE_float('weight_decay', 2e-4, 'weight decay value')
tf.app.flags.DEFINE_string('lr_decay_type', 'cosine', '`step` or `cosine`, defaults to cosine')
tf.app.flags.DEFINE_integer('n_epochs', 300, 'num epochs')
tf.app.flags.DEFINE_integer('n_regularize_epochs', 100, 'num epochs')
tf.app.flags.DEFINE_string('optimizer', 'sgdr', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_integer('quantization_epochs', 15, 'number of regularization epochs for post-training quantization')
tf.app.flags.DEFINE_integer('fine_tune_epochs', 25, 'number of regularization epochs for post-training quantization')
tf.app.flags.DEFINE_integer('batch_size', 128, 'the batch size')

import time as time
import numpy as np
import torch.backends.cudnn as cudnn
from preprocess import get_transform
from data import get_dataset
from models.resnet_ptq import ResNet_cifar10_ptq
import matplotlib.pyplot as polt
from utils.model_utils import *

from utils.dataset_utils import *
from utils.visualization_utils import *
from torch.autograd import Variable
import torch.optim as optim
from sgdR import SGDR
from adamR import AdamR
from ptq_utils import *
from cot_loss import *
from models.resnet_quantized import ResNet_cifar10

'''
Parse flags
'''
if FLAGS.activation == 'tanh':
    activation = nn.Hardtanh()
elif FLAGS.activation is None:
    activation = nn.ReLU()
    FLAGS.activation = 'relu'
activation = FLAGS.activation
FLAGS = tf.app.flags.FLAGS
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act
finetune_epochs = FLAGS.fine_tune_epochs
n_workers = 4
regularizer = FLAGS.regularization
distil = FLAGS.regularization == 'distillation'
NUM_CLASSES = FLAGS.n_classes

cudnn.benchmark = True
# torch.cuda.set_device(0)
batch_size = FLAGS.batch_size
n_runs = FLAGS.n_runs
n_epochs = FLAGS.n_epochs
record_interval = FLAGS.record_interval
xentropy_criterion = nn.CrossEntropyLoss()
n_reg_epochs = FLAGS.n_regularize_epochs


if FLAGS.activation == 'tanh':
    activation = nn.Hardtanh()
elif FLAGS.activation is None:
    activation = nn.ReLU()
    FLAGS.activation = 'relu'

FP_SAVEPATH = os.path.join(FLAGS.fpmodel_path, '%s' % FLAGS.source_dataset)

'''
Define train loader - for the specified dataset
'''
print('Loading Source Dataset: %s' % (FLAGS.source_dataset))
source_train_data = get_dataset(name=FLAGS.source_dataset, split='train',
                                transform=get_transform(name=FLAGS.source_dataset, augment=True))
source_test_data = get_dataset(name=FLAGS.source_dataset, split='test',
                               transform=get_transform(name=FLAGS.source_dataset, augment=False))
source_train_loader = torch.utils.data.DataLoader(source_train_data, batch_size=FLAGS.batch_size, shuffle=True,
                                                  num_workers=n_workers, pin_memory=True)
source_test_loader = torch.utils.data.DataLoader(source_test_data, batch_size=FLAGS.batch_size, shuffle=True,
                                                 num_workers=n_workers, pin_memory=True)

'''
Teacher Model
'''
model = ResNet_cifar10(is_quantized=FLAGS.is_quantized, inflate=FLAGS.inflate, activation=activation,
                               num_classes=NUM_CLASSES)
checkpoint = torch.load(FLAGS.fp_loadpath)
print('Restoring teacher model from:\t' + FLAGS.fp_loadpath)

teacher_model = ResNet_cifar10(is_quantized=False, num_classes=NUM_CLASSES)
teacher_model.cuda()
teacher_model.load_state_dict(checkpoint['model_state_dict'])

for param in teacher_model.parameters():
    param.requires_grad = False

test_loss, test_acc = test_model(source_test_loader, teacher_model, xentropy_criterion, printing=False)
msg = '\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc)
print(msg)
# if FLAGS.target_dataset == FLAGS.source_dataset:
#     print('Source and target dataset are same!')
#
#     target_train_data = source_train_data
#     target_test_data = source_test_data
#
#     # target_train_loader = source_train_loader
#     # target_test_loader = source_test_loader
# else:
#     '''
#     Define dataloaders for source task
#     '''
#     target_train_data = get_dataset(name=FLAGS.target_dataset, split='train',
#                              transform=get_transform(name=FLAGS.target_dataset, augment=True))
#     target_test_data = get_dataset(name=FLAGS.target_dataset, split='test',
#                             transform=get_transform(name=FLAGS.target_dataset, augment=False))
# target_train_loader = torch.utils.data.DataLoader(target_train_data, batch_size=FLAGS.batch_size, shuffle=True,
#                                            num_workers=n_workers, pin_memory=True)
# target_test_loader = torch.utils.data.DataLoader(target_test_data, batch_size=FLAGS.batch_size, shuffle=True,
#                                            num_workers=n_workers, pin_memory=True)


# if FLAGS.loadpath is not None:
#     loadpath = os.path.join(FLAGS.loadpath, 'Run%d' % (k), 'resnet')
#     # loadpath = FLAGS.loadpath
#     print('Restoring model to train from:\t' + loadpath)
#     checkpoint = torch.load(loadpath)
#
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     optimizer.param_groups[0]['lr'] = FLAGS.lr
#     test_loss, test_acc = test_model(target_test_data, model, xentropy_criterion, printing=False)
#     print('\nRestored Model Accuracy: \t %.3f' % (test_acc))
'''

Train num runs worth of fp32 models

'''


def get_calibrated_activation_minmax(model):
    for name, layer in list(model.named_modules()):
        if 'ema' in name:
            print(name)
            print(layer.running_min)
            print(layer.running_max)


def enable_model_calibration(model):
    for name, layer in list(model.named_modules()):
        if 'conv' in name or 'linear' in name or 'fc' in name:
            if 'ema' not in name:
                layer.enable_calibration()


def disable_model_calibration(model):
    for name, layer in list(model.named_modules()):
        if 'conv' in name or 'linear' in name or 'fc' in name:
            if 'ema' not in name:
                layer.disable_calibration()


def run_calibration_epochs(n_calib_epochs, model, train_data_loader, loss_fn):
    enable_model_calibration(model)
    model.train()
    # model.training=True
    with torch.no_grad():
        for epoch in range(n_calib_epochs):
            print('\n************** STARTED CALIBRATION EPOCH %d/%d **************' % \
                  (int(epoch + 1), int(n_calib_epochs)))
            for iter, batch_data in enumerate(train_data_loader):
                inputs = batch_data[0]
                targets = batch_data[1]

                inputs = inputs.cuda()
                targets = targets.cuda()
                # pdb.set_trace()
                output = model(inputs)
                loss = loss_fn(output, targets)
                lossval = loss.item()
                # optimizer.zero_grad()

                train_acc = accuracy(output, targets).item()

                if iter % 25 == 0:
                    msg = 'Step [%d]| Loss [%.4f]| Train Acc [%.3f]' % (iter, lossval, train_acc)
                    # msg += '\n'.th
                    print(msg)


# pdb.set_trace()
for run_number in range(FLAGS.n_runs):

    '''
    Instantiate model and optimizer
    '''
    model = ResNet_cifar10_ptq(inflate=FLAGS.inflate, activation=activation)
    # pdb.set_trace()

    model.cuda()
    # pdb.set_trace()
    if FLAGS.optimizer == 'adam':
        optimizer = optim.SGD(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    elif FLAGS.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)

    elif FLAGS.optimizer == 'sgdr':
        # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        optimizer = SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)

    elif FLAGS.optimizer == 'adamr':
        # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        optimizer = AdamR(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)

    if FLAGS.lr_decay_type == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)
    else:
        lr_scheduler = None
    logstr = '\n**************************************\n'
    logstr += '\n********** TRAINING STARTED **********\n'
    logstr += '\n**************************************\n'
    print(logstr)

    '''
    File directory setup
    '''
    fp32_modelsdir = os.path.join(FP_SAVEPATH, 'Run%d' % run_number)
    if not os.path.exists(fp32_modelsdir):
        os.makedirs(fp32_modelsdir, exist_ok=True)
    fp32_model_savepath = os.path.join(fp32_modelsdir, 'resnet')
    logfile_path = os.path.join(fp32_modelsdir, 'logfile.txt')
    resultsfile_path = os.path.join(fp32_modelsdir, 'resultsfile.txt')

    logfile = open(logfile_path, 'a')
    resultsfile = open(resultsfile_path, 'a')
    '''
    Train model if not exist, load model if exists
    '''
    if not os.path.isfile(fp32_model_savepath):

        train_fp32_model_renset18(model=model,
                                  train_loader=source_train_loader,
                                  test_loader=source_test_loader,
                                  optimizer=optimizer,
                                  lr_scheduler=lr_scheduler,
                                  n_epochs=FLAGS.n_epochs,
                                  loss_fn=xentropy_criterion,
                                  savepath=fp32_model_savepath,
                                  logfile_handle=logfile)
    else:
        checkpoint = torch.load(fp32_model_savepath)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = FLAGS.lr
    '''

    Return FP32 Accuracy

    '''
    test_loss, source_test_acc = test_model_ptq(source_test_loader, model, xentropy_criterion, printing=False)
    msg = '************* Train Accuracy FP32 on Source Dataset: %s*************\n' % FLAGS.source_dataset
    msg += 'TrainedMODEL WITH FP32 ACC | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, source_test_acc)
    msg += '************* END *************\n'

    print()
    pdb.set_trace()

    '''

    Apply ptq - get the baseline

    '''
    # pdb.set_trace()
    test_loss, source_quantized_test_acc = test_model_ptq(source_test_loader, model, xentropy_criterion, printing=False,
                                                          is_quantized=True, n_bits_wt=FLAGS.n_bits_wt,
                                                          n_bits_act=FLAGS.n_bits_act)

    msg += '\t\t\t TARGET BIT WIDTH %dbA/%dbW' % (FLAGS.n_bits_act, FLAGS.n_bits_wt)
    msg += '************* Trained Accuracy (QUANTIZED) on source dataset: %s*************\n' % FLAGS.source_dataset

    msg += 'Trained QUANTIZED ACC| Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % \
           (test_loss, source_quantized_test_acc)
    msg += '************* END *************\n'
    print(msg)

    # run_calibration_epochs(3, model, source_train_loader, xentropy_criterion)
    # test_loss, source_quantized_test_acc = test_model_ptq(source_test_loader, model, xentropy_criterion, printing=False,
    #                                                       is_quantized=True, n_bits_wt=FLAGS.n_bits_wt,
    #                                                       n_bits_act=FLAGS.n_bits_act)
    #
    # msg += '\t\t\t TARGET BIT WIDTH %dbA/%dbW' % (FLAGS.n_bits_act, FLAGS.n_bits_wt)
    # msg += '************* Trained Accuracy (QUANTIZED) on source dataset: %s*************\n' % FLAGS.source_dataset
    #
    # msg += 'Trained QUANTIZED ACC| Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % \
    #        (test_loss, source_quantized_test_acc)
    # msg += '************* END *************\n'
    # print(msg)

    max_orig_fisher = torch.tensor(0.).cuda()
    max_inv_fisher = torch.tensor(0.).cuda()
    for group in optimizer.param_groups:
        for p in group['params']:
            # pdb.set_trace()
            if hasattr(p, 'pert'):
                if FLAGS.fisher_method == 'adam':

                    p.fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                    p.orig_fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                    max_orig_fisher = torch.max(max_orig_fisher, torch.max(p.orig_fisher))

    for group in optimizer.param_groups:
        for p in group['params']:
            if hasattr(p, 'pert'):
                # orig_fisher.append(p.fisher.view(-1).cpu().numpy()/max_orig_fisher.cpu().numpy())
                # orig_inv_fish.append(p.inv_FIM.view(-1).cpu().numpy()/max_inv_fisher.cpu().numpy())
                # pdb.set_trace()
                if FLAGS.layerwise_fisher:
                    p.fisher = p.fisher / torch.max(p.fisher)
                    # p.inv_FIM = p.inv_FIM / torch.max(p.inv_FIM)
                else:
                    p.fisher = p.fisher / max_orig_fisher

                p.inv_FIM = 1 / (p.fisher + 1e-7)
                p.inv_FIM[p.inv_FIM>1e4] = 1e4
                p.inv_FIM = p.inv_FIM/1e4
                inv_f = p.inv_FIM.view(-1).cpu().numpy() * max_orig_fisher.cpu().numpy()

    resultsfile.write(msg)
    resultsfile.flush()

    logfile.close()
    resultsfile.close()
    print()


    model.train()
    gamma_ = FLAGS.gamma
    msg = '*******************************************************'
    msg += '\t\t TEST ACCURACY @ TRAINING START %.3f' % (source_quantized_test_acc)
    msg += '*******************************************************'
    print(msg)
    i=0
    # pdb.set_trace()
    for epoch in range(n_reg_epochs):
        epoch_msg = '\n********************************\n\tEpoch %d' % epoch
        print(epoch_msg)

        if regularizer is not None:
            gamma_ = FLAGS.gamma
            epoch_msg = '\n********************************\n\tEpoch %d| Reg Mult [%.2e]' % (epoch, gamma_)

        else:
            gamma_=0.

        for iter, (inputs, targets) in enumerate(source_train_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            if distil:
                output = model(inputs, is_quantized=True, STE=FLAGS.ste,
                               n_bits_act=FLAGS.n_bits_act, n_bits_wt=FLAGS.n_bits_wt)

                loss = xentropy_criterion(output, targets)

                teacher_output = Variable(teacher_model(inputs), requires_grad=False)

                kd_loss = FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)

                loss = loss + kd_loss
                # pdb.set_trace()
            else:
                output = model(inputs, is_quantized=True, STE=FLAGS.ste,
                               n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)

                loss = xentropy_criterion(output, targets)

            lossval = loss.item()
            train_acc = accuracy(output, targets).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(regularizer=regularizer, gamma=gamma_)

            if iter % FLAGS.record_interval == 0 or i == 0:
                msg = 'Step [%d]| Loss [%.4f]| Acc [%.3f]' % (i, lossval, train_acc)
                print(msg)
            i += 1

        # pdb.set_trace()
        if epoch % 5 == 0:
            # test_loss, test_acc = test_model(source_test_loader, model, xentropy_criterion, printing=False)
            test_loss, test_acc = test_model_ptq(source_test_loader, model, xentropy_criterion, printing=False,
                                                    is_quantized=True, n_bits_wt=FLAGS.n_bits_wt,
                                                    n_bits_act=FLAGS.n_bits_act)

            msg='*******************************************************'
            msg+='\t\t TEST ACCURACY @ EPOCH %d : %.3f' % (epoch, test_acc)
            msg+='*******************************************************'
            print(msg)





