import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import os
import math
import pdb
from debug_helper_funcs import *


tf.app.flags.DEFINE_string('dataset', 'fashionmnist' ,'either mnist or fashionmnist')
tf.app.flags.DEFINE_integer('batch_size', 128, 'the batch size')
tf.app.flags.DEFINE_integer('n_epochs', 25, 'number of epochs')
tf.app.flags.DEFINE_integer('record_interval', 100, 'iterations between printing to the console')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning_rate')

tf.app.flags.DEFINE_string('activation', 'relu','`tanh` or `relu`')
tf.app.flags.DEFINE_boolean('is_quantized', True, 'quantized or not?')
tf.app.flags.DEFINE_integer('n_bits_act', 8, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 8, 'number of bits wt')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add: None, NVM, PCM')
tf.app.flags.DEFINE_float('q_min', None, 'maximum qunatizer value')
tf.app.flags.DEFINE_float('q_max', None, 'minimum of quantizer value')
tf.app.flags.DEFINE_integer('n_runs', 7, 'number of runs')
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
tf.app.flags.DEFINE_integer('n_reg_epochs', 10,'number of regularization epochs for post-training quantization')

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

FLAGS = tf.app.flags.FLAGS

n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act
n_workers = 4

criterion = nn.CrossEntropyLoss()

train_data = get_dataset(name=FLAGS.dataset, split='train',
                         transform=get_transform(name=FLAGS.dataset, augment=True))
test_data = get_dataset(name=FLAGS.dataset, split='test',
                        transform=get_transform(name=FLAGS.dataset, augment=False))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)

distil = FLAGS.regularization=='distillation'
if distil:
    checkpoint = torch.load(FLAGS.fp_loadpath)

    teacher_model = models.Lenet5PTQ(is_quantized=False)
    teacher_model.cuda()
    teacher_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    for param in teacher_model.parameters():
        param.requires_grad = False

    test_loss, test_acc = test_model_ptq(test_loader, teacher_model, criterion, printing=False)
    msg=''
    msg += '\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc)
    print(msg)

teacher_model.eval()

model = models.Lenet5PTQ(is_quantized=FLAGS.is_quantized)
model.cuda()
if FLAGS.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
elif FLAGS.optimizer == 'adamr':
    optimizer = AdamR(model.parameters(), lr=FLAGS.lr)

n_epochs = FLAGS.n_epochs
n_reg_epochs = FLAGS.n_reg_epochs

i=0

regularizer = FLAGS.regularization

def regularizer_multiplier(curr_epoch, n_epochs, offset=10):


    midpt = int((n_epochs)/2)
    return 1/(1+np.exp(-0.5*(curr_epoch - midpt)))


model_path = './tmp_models/lenet5_fashionmnist/lenet'

if not os.path.exists(model_path):
    pause = False
    for epoch in range(n_epochs):
        if regularizer is not None:
            mult = regularizer_multiplier(epoch, n_epochs+10)
            gamma_ = FLAGS.gamma * mult
            epoch_msg = '\n********************************\n\tEpoch %d| Reg Mult [%.2e]' % (epoch, gamma_)

            print(mult)
        else:
            epoch_msg = '\n********************************\n\tEpoch %d' % epoch
            gamma_=0.

        print(epoch_msg)
        model.train()
        start = time.time()

        for iter, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.cuda()
            targets = targets.cuda()



            if distil:
                output = model(inputs, is_quantized=True)

                loss = criterion(output, targets)

                teacher_output = Variable(teacher_model(inputs), requires_grad=False)
                # teacher_output = teacher_output.long()
                # teach_train_acc = accuracy(teacher_output, targets).item()
                # print('Teacher Acc: '+ str(teach_train_acc ))
                kd_loss = FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)

                loss = loss + kd_loss
                pdb.set_trace()
            else:
                output = model(inputs, is_quantized=False)

                loss = criterion(output, targets)


            optimizer.zero_grad()
            loss.backward()

            if epoch>20:
                pause=False
            optimizer.step(regularizer = regularizer, gamma=gamma_, pause=pause)

            lossval = loss.item()
            train_acc = accuracy(output, targets).item()

            if iter % FLAGS.record_interval==0 or i==0:
                if distil:
                    kd_l = kd_loss.item()
                    msg = 'Step [%d]| Loss [%.4f]| KD Loss [%.4f]| Acc [%.3f]' % (i, lossval, kd_l , train_acc)
                else:
                    msg = 'Step [%d]| Loss [%.4f]| Acc [%.3f]' % (i, lossval, train_acc)
                print(msg)
            i+=1

    test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=False,
                       n_bits_wt=8, n_bits_act=8)

    msg = '************* FINAL ACCURACY *************\n'
    msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
    msg += '************* END *************\n'
    print(msg)

    test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=True,
                       n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
    msg = '************* FINAL ACCURACY *************\n'
    msg += 'TRAINING END | Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % (test_loss, test_acc)
    msg += '************* END *************\n'
    print(msg)

    msg = '************* FINAL ACCURACY *************\n'
    msg += 'POST TRAINING QUANTIZATION START'
    msg += '************* END *************\n'

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, model_path)
    print('SAVING TO: \t' + model_path)

else:

    checkpoint = torch.load(model_path)
    # pdb.set_trace()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer.param_groups[0]['lr'] = FLAGS.lr

    test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False)
    msg = '************* FINAL ACCURACY *************\n'
    msg += 'LOADED MODEL WITH FP32 ACC | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
    msg += '************* END *************\n'
    print(msg)

    test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=True,
                       n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
    msg = '************* FINAL ACCURACY *************\n'
    msg += 'MODEL QUANTIZED ACC| Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % (test_loss, test_acc)
    msg += '************* END *************\n'
    print(msg)
pause = False
for epoch in range(n_reg_epochs):
    if regularizer is not None:
        mult = regularizer_multiplier(epoch, n_epochs+10)
        gamma_ = FLAGS.gamma * mult
        epoch_msg = '\n********************************\n\tEpoch %d| Reg Mult [%.2e]' % (epoch, gamma_)

        print(mult)
    else:
        epoch_msg = '\n********************************\n\tEpoch %d' % epoch
        gamma_=0.
    print(epoch_msg)
    gamma_ = FLAGS.gamma

    model.train()
    start = time.time()

    for iter, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.cuda()
        targets = targets.cuda()

        if distil:
            output = model(inputs, n_bits_act=FLAGS.n_bits_act, n_bits_wt=FLAGS.n_bits_wt)

            loss = criterion(output, targets)

            teacher_output = Variable(teacher_model(inputs), requires_grad=False)
            # teacher_output = teacher_output.long()
            # teach_train_acc = accuracy(teacher_output, targets).item()
            # print('Teacher Acc: '+ str(teach_train_acc ))
            kd_loss = FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)

            loss = loss + kd_loss
            # pdb.set_trace()
        else:
            output = model(inputs, is_quantized=False)

            loss = criterion(output, targets)


        optimizer.zero_grad()
        loss.backward()

        if epoch==n_reg_epochs-1:
            pause=True
        optimizer.step(regularizer = regularizer, gamma=gamma_, pause=pause)

        lossval = loss.item()
        train_acc = accuracy(output, targets).item()

        if iter % FLAGS.record_interval == 0 or i == 0:
            if distil:
                kd_l = kd_loss.item()
                msg = 'Step [%d]| Loss [%.4f]| KD Loss [%.4f]| Acc [%.3f]' % (i, lossval, kd_l, train_acc)
            else:
                msg = 'Step [%d]| Loss [%.4f]| Acc [%.3f]' % (i, lossval, train_acc)
            print(msg)
        i+=1

test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=False,
                   n_bits_wt=8, n_bits_act=8)

for layer in model.layers:
    print(layer.qmin_wt)
    print(layer.qmax_wt)

msg = '************* FINAL ACCURACY *************\n'
msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
msg += '************* END *************\n'
print(msg)

test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=True,
                   n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
msg = '************* FINAL ACCURACY *************\n'
msg += 'TRAINING END | Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % (test_loss, test_acc)
msg += '************* END *************\n'
print(msg)

def plot_weight_histograms(model, log=False):
    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(model.conv1.weight.detach().cpu().numpy().ravel(), bins=50, log=log)
    plt.title('Conv1 pdf')

    plt.subplot(2, 2, 2)
    plt.hist(model.conv2.weight.detach().cpu().numpy().ravel(), bins=250, log=log)
    plt.title('Conv2 pdf')

    plt.subplot(2, 2, 3)
    plt.hist(model.fc1.weight.detach().cpu().numpy().ravel(), bins=1500, log=log)
    plt.title('FC1 pdf')

    plt.subplot(2, 2, 4)
    plt.hist(model.fc2.weight.detach().cpu().numpy().ravel(), bins=1000, log=log)
    plt.title('FC2 pdf')


'''

Plot histogram of fisher vs 

'''

plot_weight_histograms(model, log=False)
plt.show()

exit()

'''
----------------------------------------------------------------------------------------------------------------------------------------
'''








PERFORM_RUN=False
if PERFORM_RUN:

    bit_widths = [8, 7, 6, 5, 4, 3, 2]
    accuracies = np.zeros([FLAGS.n_runs, len(bit_widths)])
    for k in range(FLAGS.n_runs):

        model = models.Lenet5PTQ(is_quantized=FLAGS.is_quantized)
        model.cuda()
        if FLAGS.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
        elif FLAGS.optimizer == 'adamr':
            optimizer = AdamR(model.parameters(), lr=FLAGS.lr)

        n_epochs = FLAGS.n_epochs

        i=0
        for epoch in range(n_epochs):
            print('\n********************************\n\tEpoch %d' % epoch)
            model.train()
            start = time.time()

            for iter, (inputs, targets) in enumerate(train_loader):

                inputs = inputs.cuda()
                targets = targets.cuda()

                output = model(inputs)
                loss = criterion(output, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lossval = loss.item()
                train_acc = accuracy(output, targets).item()

                if iter % FLAGS.record_interval==0 or i==0:
                    msg = 'Step [%d]| Loss [%.4f]| Acc [%.3f]' % (i, lossval, train_acc)
                    print(msg)
                i+=1

        test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=False,
                           n_bits_wt=8, n_bits_act=8)

        msg = '************* FINAL ACCURACY *************\n'
        msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
        msg += '************* END *************\n'
        print(msg)

        for j, bit_width in enumerate(bit_widths):
            test_loss, test_acc = test_model_ptq(test_loader, model, criterion, printing=False, is_quantized=True,
                               n_bits_wt=bit_width, n_bits_act=bit_width)
                               # n_bits_wt=FLAGS.n_bits_wt, n_bits_act=FLAGS.n_bits_act)
            msg = '************* FINAL ACCURACY *************\n'
            msg += 'TRAINING END | Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % (test_loss, test_acc)
            msg += '************* END *************\n'
            print(msg)
            accuracies[k, j] = test_acc

    for j, bit_width in enumerate(bit_widths):
        print('************************************')
        print('Bit Width')
        print(bit_width)
        print('Accuracy')
        print(np.mean(accuracies[:, j]))
        print('Std')
        print(np.std(accuracies[:, j]))

    np.savetxt('./results/accuracies_lenet5ptq_naive.txt', accuracies, fmt='%.3f')

#Apply quantization to parameters

#Set is quantized flag to true
