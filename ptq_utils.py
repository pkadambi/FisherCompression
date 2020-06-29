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

def train_fp32_model(model, train_loader, test_loader, optimizer, n_epochs, loss_criterion, savepath):

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
            optimizer.step(regularizer=None, gamma=gamma_)
            loss = loss_criterion(output, targets)
            lossval = loss.item()

            optimizer.zero_grad()
            loss.backward()
            train_acc = accuracy(output, targets).item()

            if iter % FLAGS.record_interval==0 or i==0:
                msg = 'Step [%d]| Loss [%.4f]| Acc [%.3f]' % (i, lossval, train_acc)
                print(msg)
            i+=1

    test_loss, test_acc = test_model_ptq(test_loader, model, loss_criterion, printing=False, is_quantized=False)

    msg = '****************** FINAL ACCURACY ******************\n'
    msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
    msg += '****************** END ******************\n'
    print(msg)

    test_loss, test_acc = test_model_ptq(test_loader, model, loss_criterion, printing=False, is_quantized=True,
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
            'loss': loss_criterion}, savepath)
        print('SAVING TO: \t' + savepath)


