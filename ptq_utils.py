'''
#TODO: repurpose this file as some sort of post training quantization file
'''
import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os
import math
import pdb
from debug_helper_funcs import *
import time as time
import numpy as np
import torch.backends.cudnn as cudnn
from preprocess import get_transform
from data import get_dataset
import models
import matplotlib.pyplot as polt
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualization_utils import *
from torch.autograd import Variable
import torch.optim as optim
from adamR import AdamR

FLAGS = tf.app.flags.FLAGS
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act

def train_fp32_model_lenet(model, train_loader, test_loader, optimizer, n_epochs, loss_fn, savepath):
    '''

    In this training loop, an fp32 lenet model is trained
    There are different functions for resnet, lenet, vgg, etc because there are different things that need
    to be recorded

    :param model:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param n_epochs:
    :param loss_criterion:
    :param savepath:
    :return:
    '''
    i=0
    for epoch in range(n_epochs):
        epoch_msg = '\n********************************\n\tEpoch %d' % epoch
        gamma_=0.

        print(epoch_msg)
        model.train()
        start = time.time()

        for iter, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.cuda().float()
            targets = targets.cuda()



            output = model(inputs, is_quantized=False)
            loss = loss_fn(output, targets)
            lossval = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(regularizer=None, gamma=gamma_)

            train_acc = accuracy(output, targets).item()

            if iter % FLAGS.record_interval==0 or i==0:
                msg = 'Step [%d]| Loss [%.4f]| Train Acc [%.3f]' % (i, lossval, train_acc)
                print(msg)
            i+=1

        if epoch % 10 == 0:
            test_loss, test_acc = test_model(test_loader, model, loss_fn, printing=False)
            print('*******************************************************')
            print('\t\t TEST ACCURACY @ EPOCH%d : %.3f' % (epoch, test_acc))
            print('*******************************************************')

    test_loss, test_acc = test_model_ptq(test_loader, model, loss_fn, printing=False, is_quantized=False)

    msg = '****************** FINAL ACCURACY ******************\n'
    msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
    msg += '****************** END ******************\n'
    print(msg)

    test_loss, test_acc = test_model_ptq(test_loader, model, loss_fn, printing=False, is_quantized=True,
                       n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
    msg = '****************** FINAL ACCURACY ******************\n'
    msg += 'TRAINING END | Test Loss Quantized (%d/%d) [%.3f]| Test Acc Quantized [%.3f]\n' % \
           (n_bits_wt, n_bits_act, test_loss, test_acc)
    msg += '****************** END ******************\n'
    print(msg)

    if savepath is not None:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn}, savepath)
        print('SAVING TO: \t' + savepath)




def train_fp32_model_renset18(model, train_loader, test_loader, optimizer, n_epochs, loss_fn, savepath,
                              lr_scheduler, logfile_handle):
    '''

    this function has been tested on cifar10

    :param model:
    :param train_loader:
    :param test_loader:
    :param optimizer:
    :param n_epochs:
    :param loss_criterion:
    :param savepath:
    :return:
    '''
    i=0
    for epoch in range(n_epochs):
        epoch_msg = '\n********************************\n\tEpoch %d' % epoch
        if epoch>150 and FLAGS.n_bits_wt<=2:
            for group in optimizer.param_groups:
                group['weight_decay']=0.
        print(epoch_msg)

        for iter, batch_data in enumerate(train_loader):
            inputs = batch_data[0]
            targets = batch_data[1]

            inputs = inputs.cuda()
            targets = targets.cuda()
            # pdb.set_trace()
            output = model(inputs)
            loss = loss_fn(output, targets)
            lossval = loss.item()
            train_acc = accuracy(output, targets).item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step(regularizer=None, gamma=0.)

            if iter % FLAGS.record_interval == 0 or i == 0:
                msg = 'Step [%d]| Loss [%.4f]| Train Acc [%.3f]' % (i, lossval, train_acc)
                msg+='\n'
                logfile_handle.write(msg)
                logfile_handle.flush()
                print(msg)

            i += 1

        if epoch % 25 == 0:
            test_loss, test_acc = test_model(test_loader, model, loss_fn, printing=False)
            msg='*******************************************************'
            msg+='\t\t TEST ACCURACY @ EPOCH%d : %.3f' % (epoch, test_acc)
            msg+='*******************************************************'
            logfile_handle.write(msg)
            logfile_handle.flush()
            print(msg)
        lr_scheduler.step()
    test_loss, test_acc = test_model_ptq(test_loader, model, loss_fn, printing=False, is_quantized=False)

    msg = '****************** FINAL ACCURACY ******************\n'
    msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
    msg += '****************** END ******************\n'
    print(msg)
    logfile_handle.write(msg)
    logfile_handle.flush()
    test_loss, test_acc = test_model_ptq(test_loader, model, loss_fn, printing=False, is_quantized=True,
                       n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
    msg = '****************** FINAL ACCURACY ******************\n'
    msg += 'TRAINING END | Test Loss Quantized (%d/%d) [%.3f]| Test Acc Quantized [%.3f]\n' % \
           (n_bits_wt, n_bits_act, test_loss, test_acc)
    msg += '****************** END ******************\n'
    print(msg)
    logfile_handle.write(msg)
    logfile_handle.flush()
    if savepath is not None:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn}, savepath)
        print('SAVING TO: \t' + savepath)




