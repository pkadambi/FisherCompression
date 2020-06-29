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
tf.app.flags.DEFINE_integer('n_bits_act', 4, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 4, 'number of bits wt')

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
from ptq_utils import *

'''
Parse flags
'''
FLAGS = tf.app.flags.FLAGS
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act
n_workers = 4
criterion = nn.CrossEntropyLoss()
distil = FLAGS.regularization=='distillation'


'''
Define train loader - for the specified dataset
'''
train_data = get_dataset(name=FLAGS.dataset, split='train',
                         transform=get_transform(name=FLAGS.dataset, augment=True))
test_data = get_dataset(name=FLAGS.dataset, split='test',
                        transform=get_transform(name=FLAGS.dataset, augment=False))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)

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
n_reg_epochs = FLAGS.n_reg_epochs
regularizer = FLAGS.regularization

'''
Define optimizer 
'''
if FLAGS.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
elif FLAGS.optimizer == 'adamr':
    optimizer = AdamR(model.parameters(), lr=FLAGS.lr)

model_path_ = './tmp_models/lenet5_%s' % FLAGS.dataset
model_path = model_path_ + '/lenet'
criterion = nn.CrossEntropyLoss()

if not os.path.isfile(model_path):
    os.makedirs(model_path_, exist_ok=True)
    train_fp32_model(model=model, train_loader=train_loader, test_loader=test_loader,
                     optimizer=optimizer, n_epochs=n_epochs, loss_criterion=criterion,
                     savepath=model_path)
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

'''
Load teacher model (if distillation)
'''
train_data = get_dataset(name=FLAGS.dataset, split='train',
                         transform=get_transform(name='notmnist', augment=True))
test_data = get_dataset(name=FLAGS.dataset, split='test',
                        transform=get_transform(name='notmnist', augment=False))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)

'''
Define dataloaders for notmnist
'''



'''
Fine-tune last layer on the target task (notmnist)
'''

i=0
for iter, (inputs, targets) in enumerate(train_loader):
    pass






