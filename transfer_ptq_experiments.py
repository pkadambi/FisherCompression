import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os
import math
import pdb
from debug_helper_funcs import *


tf.app.flags.DEFINE_string('source_dataset', 'fashionmnist' ,'can be any lenet trainable dataset')
tf.app.flags.DEFINE_string('target_dataset', 'notmnist', 'can be any of the datasets')
tf.app.flags.DEFINE_integer('batch_size', 128, 'the batch size')
tf.app.flags.DEFINE_integer('n_epochs', 25, 'number of epochs')
tf.app.flags.DEFINE_integer('record_interval', 100, 'iterations between printing to the console')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning_rate')
tf.app.flags.DEFINE_boolean('ste', False, 'whether or not to use STE while doing quantized regularization')
tf.app.flags.DEFINE_string('activation', 'relu','`tanh` or `relu`')
tf.app.flags.DEFINE_boolean('is_quantized', True, 'quantized or not?')
tf.app.flags.DEFINE_integer('n_bits_act', 4, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 4, 'number of bits wt')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add: None, NVM, PCM')
tf.app.flags.DEFINE_float('q_min', None, 'maximum qunatizer value')
tf.app.flags.DEFINE_float('q_max', None, 'minimum of quantizer value')
tf.app.flags.DEFINE_integer('n_runs', 8, 'number of runs')
tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2,` `fisher` `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'Whether or not to enforce zero level in quantizer')
tf.app.flags.DEFINE_float('gamma', 1e-4, 'regularizer gamma')
tf.app.flags.DEFINE_float('diag_load_const', 1e-1, 'diagonal loading constant')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save to')
tf.app.flags.DEFINE_string('loadpath', './SavedModels/LenetPTQ/8ba_8bw/', 'directory to load from')

tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode')
tf.app.flags.DEFINE_string('optimizer', 'adamr', 'optimizer to use `sgd` or `adam`')

tf.app.flags.DEFINE_string('fp_loadpath', './tmp_models/lenet5_fashionmnist/lenet', 'path to FP for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'distillation temperature')
tf.app.flags.DEFINE_integer('fine_tune_epochs', 25,'number of regularization epochs for post-training quantization')
tf.app.flags.DEFINE_integer('quantization_epochs', 15,'number of regularization epochs for post-training quantization')


#histplot_from_tensor(model.fc2.weight, bins=750)
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
from ptq_utils import *

'''
Parse flags
'''
FLAGS = tf.app.flags.FLAGS
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act
finetune_epochs = FLAGS.fine_tune_epochs
n_workers = 4
criterion = nn.CrossEntropyLoss()
distil = FLAGS.regularization=='distillation'


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
# pdb.set_trace()
if FLAGS.target_dataset == FLAGS.source_dataset:
    print('Source and target dataset are same!')

    target_train_data = source_train_data
    target_test_data = source_test_data

    target_train_loader = source_train_loader
    target_test_loader = source_test_loader
else:
    '''
    Define dataloaders for source task
    '''
    target_train_data = get_dataset(name=FLAGS.target_dataset, split='train',
                             transform=get_transform(name=FLAGS.target_dataset, augment=True))
    target_test_data = get_dataset(name=FLAGS.target_dataset, split='test',
                            # transform=None)
                            transform=get_transform(name=FLAGS.target_dataset, augment=False))
    target_train_loader = torch.utils.data.DataLoader(target_train_data, batch_size=FLAGS.batch_size, shuffle=True,
                                               num_workers=n_workers, pin_memory=True)
    target_test_loader = torch.utils.data.DataLoader(target_test_data, batch_size=FLAGS.batch_size, shuffle=True,
                                               num_workers=n_workers, pin_memory=True)
# pdb.set_trace()
n_runs = FLAGS.n_runs


quantized_transferred_acc = np.zeros([n_runs])
transferred_acc = np.zeros([n_runs])

regularized_quantized_acc= np.zeros([n_runs])
regularized_acc= np.zeros([n_runs])

for run_number in range(n_runs):
    '''
    Define model
    '''
    model = models.Lenet5PTQ(is_quantized=FLAGS.is_quantized)
    model.cuda()
    if FLAGS.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'adamr':
        optimizer = AdamR(model.parameters(), lr=FLAGS.lr)
    n_epochs = FLAGS.n_epochs
    n_quantize_epochs = FLAGS.quantization_epochs
    regularizer = FLAGS.regularization

    '''
    Define optimizer 
    '''
    if FLAGS.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'adamr':
        optimizer = AdamR(model.parameters(), lr=FLAGS.lr)

    model_path_ = './tmp_models/Run%d/lenet5_%s' % (run_number, FLAGS.source_dataset)
    model_path = model_path_ + '/lenet'
    criterion = nn.CrossEntropyLoss()

    if not os.path.isfile(model_path):
        os.makedirs(model_path_, exist_ok=True)
        train_fp32_model(model=model, train_loader=source_train_loader, test_loader=source_test_loader,
                         optimizer=optimizer, n_epochs=n_epochs, loss_criterion=criterion,
                         savepath=model_path)
    else:
        checkpoint = torch.load(model_path)
        # pdb.set_trace()
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]['lr'] = FLAGS.lr

        test_loss, source_test_acc = test_model_ptq(source_test_loader, model, criterion, printing=False)
        msg = '************* FINAL ACCURACY ON SOURCE DATASET*************\n'
        msg += 'LOADED MODEL WITH FP32 ACC | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, source_test_acc )
        msg += '************* END *************\n'
        print(msg)

        test_loss, source_quantized_test_acc = test_model_ptq(source_test_loader, model, criterion, printing=False, is_quantized=True,
                           n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
        msg = '************* FINAL ACCURACY ON SOURCE DATASET*************\n'
        msg += 'MODEL QUANTIZED ACC| Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % \
               (test_loss, source_quantized_test_acc )
        msg += '************* END *************\n'
        print(msg)


    if FLAGS.source_dataset == FLAGS.target_dataset:

        print('No fine tuning of layers required! Source and target dataset are the same!')
    else:
        '''
        Fine-tune last layer on the target task - no need to do this if the target task is the same as the soruce task
        '''
        target_model_path_ = './tmp_models/Run%d/lenet5_source_%s_target_%s' % \
                      (run_number, FLAGS.source_dataset,FLAGS.target_dataset)
        target_model_path = target_model_path_ + '/lenet'

        criterion = nn.CrossEntropyLoss()

        test_loss, target_test_acc_prefinetune = test_model_ptq(target_test_loader, model, criterion, printing=False)
        msg = '************* ACCURACY ON TARGET DATASET %s BEFORE FINE TUNING*************\n' % (FLAGS.target_dataset)
        msg += 'LOADED MODEL WITH FP32 ACC | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, target_test_acc_prefinetune)
        msg += '************* END *************\n'
        print(msg)
        layers_to_freeze = ['conv1', 'conv2']
        model.freeze_unfreeze_layers(layers_to_freeze, mode='freeze')

        if not os.path.exists(target_model_path_):
            os.makedirs(target_model_path_, exist_ok=True)

            train_fp32_model(model=model, train_loader=target_train_loader, test_loader=target_test_loader,
                             optimizer=optimizer, n_epochs=finetune_epochs, loss_criterion=criterion,
                             savepath=target_model_path)
        else:
            checkpoint = torch.load(target_model_path)
            # pdb.set_trace()
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer.param_groups[0]['lr'] = FLAGS.lr

        # i=0
        # model.train()
        # print('BEGAN FINE TUNINING MODEL')
        # for epoch in range(finetune_epochs):
        #     epoch_msg = '\n********************************\n\tEpoch %d' % epoch
        #     print(epoch_msg)
        #     for iter, (inputs, targets) in enumerate(target_train_loader):
        #         inputs = inputs.cuda()
        #         targets = targets.cuda()
        #
        #         output = model(inputs, is_quantized=False)
        #         loss = criterion(output, targets)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step(regularizer=regularizer, gamma=0.)
        #
        #         lossval = loss.item()
        #         train_acc = accuracy(output, targets).item()
        #         if iter % FLAGS.record_interval == 0 or i==0:
        #             msg = 'Step [%d]| Loss [%.4f]| Acc [%.3f]' % (i, lossval, train_acc)
        #             print(msg)
        #         i+=1

        # model.freeze_unfreeze_layers(layers_to_freeze, mode='unfreeze')
        # pdb.set_trace()
        test_loss, target_test_acc = test_model_ptq(target_test_loader, model, criterion, printing=False)
        msg = '************* FINAL ACCURACY ON TARGET DATASET %s*************\n' % (FLAGS.target_dataset)
        msg += 'LOADED MODEL WITH FP32 ACC | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, target_test_acc)
        msg += '************* END *************\n'
        print(msg)

        test_loss, quantized_target_test_acc = test_model_ptq(target_test_loader, model, criterion, printing=False, is_quantized=True, n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
        msg = '************* FINAL ACCURACY ON TARGET DATASET %s*************\n' % (FLAGS.target_dataset)
        msg += 'MODEL QUANTIZED ACC| Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % \
               (test_loss, quantized_target_test_acc)
        msg += '************* END *************\n'
        print(msg)


        #3b
        #l2
        #fisher
        #invfisher

        #4b
        #invfisher
        #


    model.train()
    gamma_ = FLAGS.gamma

    '''
    Load teacher model if distillation
    '''
    if distil:
        checkpoint = torch.load(target_model_path)

        teacher_model = models.Lenet5PTQ(is_quantized=False)
        teacher_model.cuda()
        teacher_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        for param in teacher_model.parameters():
            param.requires_grad = False

        test_loss, test_acc = test_model_ptq(target_test_loader, teacher_model, criterion, printing=False)
        msg=''
        msg += '\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc)
        print(msg)

        teacher_model.eval()

    i=0
    for epoch in range(n_quantize_epochs):
        epoch_msg = '\n********************************\n\tEpoch %d' % epoch
        print(epoch_msg)

        if regularizer is not None:
            gamma_ = FLAGS.gamma
            epoch_msg = '\n********************************\n\tEpoch %d| Reg Mult [%.2e]' % (epoch, gamma_)

        else:
            gamma_=0.

        for iter, (inputs, targets) in enumerate(target_train_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            if distil:
                output = model(inputs, is_quantized=True, STE=FLAGS.ste,
                               n_bits_act=FLAGS.n_bits_act, n_bits_wt=FLAGS.n_bits_wt)

                loss = criterion(output, targets)

                teacher_output = Variable(teacher_model(inputs), requires_grad=False)

                kd_loss = FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)

                loss = loss + kd_loss
                # pdb.set_trace()
            else:
                output = model(inputs, is_quantized=True, STE=FLAGS.ste,
                               n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)

                loss = criterion(output, targets)

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

    test_loss, regularized_target_test_acc = test_model_ptq(target_test_loader, model, criterion, printing=False)
    msg = '************* FINAL ACCURACY TARGET TASK *************\n'
    msg += 'LOADED MODEL WITH FP32 ACC | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss,
                                                                                 regularized_target_test_acc)
    msg += '************* END *************\n'
    print(msg)
    test_loss, quantized_regularized_test_acc = test_model_ptq(target_test_loader, model, criterion, printing=False, is_quantized=True,
                                         n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
    msg = '************* FINAL ACCURACY TARGET TASK *************\n'
    msg += 'MODEL QUANTIZED ACC| Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % \
           (test_loss, quantized_regularized_test_acc )
    msg += '************* END *************\n'
    print(msg)

    print('RESULTS---------------------------------------------------------------------------------------------------------------------')
    print('\nOriginal (Source) Accuracy: %.3f' % source_test_acc)
    print('\nTransferred (Target) Accuracy: %.3f' % target_test_acc)
    print('\Regularized (Target) Accuracy: %.3f' % regularized_target_test_acc)

    print('\nORIGINAL Quantized (Target) Accuracy: %.3f' % quantized_target_test_acc)
    print('REGULARIZED Quantized (Target) Accuracy: %.3f' % quantized_regularized_test_acc)
    print('--------------------------------------------------------- NEXT RUN: %d---------------------------------------------------------' % run_number)
    quantized_transferred_acc[run_number] = quantized_target_test_acc
    transferred_acc[run_number] = target_test_acc

    regularized_quantized_acc[run_number]= quantized_regularized_test_acc
    regularized_acc[run_number]= regularized_target_test_acc
    # pdb.set_trace()

print('Regularization %s' % FLAGS.regularization)
print('Bits wt %d' % FLAGS.n_bits_wt)
print('Bits act %d' % FLAGS.n_bits_act)
print('\n\nTransferred (Target) Accuracy: %.3f +/- %.2f' % (np.mean(transferred_acc), np.std(transferred_acc)))
print('Quantized Transferred (Target) Accuracy: %.3f +/- %.2f' % (np.mean(quantized_transferred_acc), np.std(quantized_transferred_acc)))

print('\n\nRegularized(Target) Accuracy: %.3f +/- %.2f' % (np.mean(regularized_acc), np.std(regularized_acc)))
print('Quantized Transferred (Target) Accuracy: %.3f +/- %.2f' % (np.mean(regularized_quantized_acc), np.std(regularized_quantized_acc)))
pdb.set_trace()

