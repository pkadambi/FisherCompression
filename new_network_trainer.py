from data import get_dataset
import json
import torch.nn as nn
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import matplotlib.pyplot as plt
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualization_utils import *
from torch.autograd import Variable
import torch.optim as optim
import tensorflow as tf
import time as time
import numpy as np
import os
import pdb
import scipy.stats as stats


'''

n epochs = 300
cosine decay rate
weight decay

'''

#MUST KEEP THIS AS THE FIRST FLAG
tf.app.flags.DEFINE_string( 'dataset', 'fashionmnist', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_string('noise_scale',None,'`inv_fisher` or `fisher`')

tf.app.flags.DEFINE_string('activation', 'relu','`tanh` or `relu`')
tf.app.flags.DEFINE_string('lr_decay_type', 'cosine', '`step` or `cosine`, defaults to cosine')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 30, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')
tf.app.flags.DEFINE_float('weight_decay', 0., 'weight decay value')

tf.app.flags.DEFINE_integer('inflate', None,'inflating factor for resnet (may need bigger factor if 1-b weights')
tf.app.flags.DEFINE_float('lr', .001, 'learning rate')

tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 4, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 4, 'number of bits weight')
tf.app.flags.DEFINE_float('eta', .0, 'noise eta')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, 'minimum quant value')
tf.app.flags.DEFINE_float('q_max', None, 'maximum quant value')
tf.app.flags.DEFINE_integer('n_runs', 5, 'number of times to train network')
tf.app.flags.DEFINE_integer('n_classes', 10, 'n classes')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')
# tf.app.flags.DEFINE_string('regularization', 'distillation', 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_float('gamma', 0.0001, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 0.005, 'diagonal loading constant')

# tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_boolean('lr_decay', True, 'Whether or not to decay learning rate')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')

#TODO: find where this is actually used in the code
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')

#Distillation Params
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/cifar10/Resnet18/fp_buffer/Run3/resnet', 'path to FP model for loading for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation regularizer multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'temperature for distillation')

tf.app.flags.DEFINE_boolean('logging', True,'whether to enable writing to a logfile')
tf.app.flags.DEFINE_float('lr_end', 2e-4, 'learning rate at end of cosine decay')
tf.app.flags.DEFINE_boolean('constant_fisher', True,'whether to keep fisher/inv_fisher constant from when the checkpoint')

tf.app.flags.DEFINE_string('fisher_method', 'adam','which method to use when computing fisher')
tf.app.flags.DEFINE_boolean('layerwise_fisher', True,'whether or not to use layerwise fisher')

tf.app.flags.DEFINE_boolean('eval', False,'if this flag is enabled, the code doesnt write anyhting, it just loads from `FLAGS.loadpath` and evaluates test acc once')
tf.app.flags.DEFINE_boolean('loss_surf_eval_d_qtheta', default=False, help='whether we are in loss surface generation mode')

tf.app.flags.DEFINE_string('regularizer', '', help='can have `SLS` OR `ULS`, `Fisher` OR `MSQE`, `distillation`')

'''

NOTE: the following imports must be after flags declarations since these files query the flags 

'''

from sgdR import SGDR
from adamR import AdamR
from models.resnet_quantized import ResNet_cifar10
from models.resnet_lowp import ResNet_cifar10_lowp
from models.resnet_binary import ResNet_cifar10_binary
import models
from cot_loss import *


FLAGS = tf.app.flags.FLAGS
flag_dict = FLAGS.flag_values_dict()
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act

batch_size = FLAGS.batch_size
n_workers = 4
dataset = FLAGS.dataset
n_epochs = FLAGS.n_epochs
record_interval = FLAGS.record_interval
regularizer = FLAGS.regularizer

#Torch gpu stuff
cudnn.benchmark = True
torch.cuda.set_device(0)


def regularizer_multiplier(curr_epoch, n_epochs):

    midpt = int(n_epochs/2)

    return 1/(1+np.exp(-0.5*(curr_epoch - midpt)))

#TODO: add dataset label smoothing here


#1. Weizhi's method

#2. Distillation ( we already have that)

#3. Not label smoothing but add complement entropy loss

#4. Naive label smoothing - via the method in the medium article


DISTIL = False
SLS = False
MSQE = False
FISHER = False
COT=False

if 'distillation' in regularizer.lower():
    DISTIL = True

if 'SLS' in regularizer.lower():
    SLS = True

if 'MSQE' in regularizer.lower():
    MSQE = True

if 'fisher' in regularizer.lower():
    FISHER = True

if 'complement' in regularizer.lower() or 'cot' in regularizer.lower():
    COT = True

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()


    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target))/sample_num

        return loss


criterion = nn.CrossEntropyLoss()
criterion_full_labels = SoftCrossEntropy()
cot_loss = ComplementEntropy()

# torch.scatter()

train_data = get_dataset(name = dataset, split = 'train', transform=get_transform(name=dataset, augment=True))
test_data = get_dataset(name = dataset, split = 'test', transform=get_transform(name=dataset, augment=False))

# train_data = torch.nn.functional.one_hot(targets)


if not os.path.exists('./fashionmnist_clusters.txt'):
    npdata = np.vstack([tr[0].numpy().reshape(1,-1) for tr in train_data])
    labels = np.array([tr[1] for tr in train_data])

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    pca = PCA(n_components=256)
    pca.fit(npdata)
    reduc_data = pca.transform(npdata)

    kmeans = KMeans(n_clusters=64, random_state=0)
    kmeans.fit(reduc_data)
    np.savetxt('./fashionmnist_clusters.txt', kmeans.labels_, fmt='%d')

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                          num_workers=n_workers, pin_memory=True)

#instantiate and train the model
model = models.Lenet5(is_quantized=False)
model.cuda()

if FLAGS.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

    if COT:
        cot_optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

elif FLAGS.optimizer == 'adamr':
    # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
    optimizer = AdamR(model.parameters(), lr=FLAGS.lr)
    if COT:
        cot_optimizer = AdamR(model.parameters(), lr=FLAGS.lr)
i=0
cot_ = []
for epoch in range(n_epochs):
    model.train()
    for iter, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model(inputs)
        # targets_oh = torch.nn.functional.one_hot(targets)
        cot = cot_loss(output, targets)

        mult = regularizer_multiplier(epoch, n_epochs+10)
        loss =  criterion(output, targets) #+ 1. *  cot #+  cot_loss(output, targets)

        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

        lossval = loss.item()

        # for group in optimizer.param_groups:
        #     for p in group['params']:
        #         if hasattr(p, 'pert'):
        #             if FLAGS.fisher_method == 'adam':
        #                 p.fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
        #             elif FLAGS.fisher_method == 'g2':
        #                 p.fisher = p.grad * p.grad * FLAGS.batch_size
        #
        #             p.inv_FIM = 1 / (p.fisher + 1e-7)
        #             p.inv_FIM = p.inv_FIM * 1e-7
        #
        #
        # perts = []
        # if FLAGS.is_quantized:
        #     for l, (name, layer) in enumerate(model.named_modules()):
        #         if 'conv' in name or 'fc' in name:
        #             with torch.no_grad():
        #                 if hasattr(layer, 'qweight'):
        #                     pertw = layer.weight - layer.qweight
        #                     layer.weight.pert = pertw
        #                     perts.append(pertw)

        cot_.append(cot.item())

        if COT:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = 0. * cot_loss(output, targets)
            cot_optimizer.zero_grad()
            loss.backward()
            cot_optimizer.step()
            cot_lossval = loss.item()
            cot_.append(cot_lossval)

        train_acc = accuracy(output, targets).item()

        if i % record_interval == 0 or i == 0:
            msg = 'Step [%d] | Loss [%.4f] | Acc [%.3f]|' % (i, lossval, train_acc)
            if COT:
                msg += '| COT Loss [%.3f]' % cot_lossval

            print(msg)
        i+=1

    if epoch % 1 == 0 or epoch == n_epochs - 1:
        msg = '\n*** TESTING ***\n'
        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=0.)


        msg += 'End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| | LR [%.5f]\n' % (
            epoch, test_loss, test_acc, optimizer.param_groups[0]['lr'])
        print('Multiplier: '+ str(mult))

        if COT:
            msg += '| COT Loss [%.3f]' % cot_lossval
        print(msg)
print('\n*** TESTING ***\n')
# test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=0.)
# print('End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| LR [%.3f]' %
#       (epoch, test_loss, test_acc, optimizer.param_groups[0]['lr']))

test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=0.)

msg = '************* FINAL ACCURACY *************\n'
msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
msg += '************* END *************\n'
print(msg)

model.is_quantized=True
test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=0.)
msg = '************* FINAL ACCURACY *************\n'
msg += 'TRAINING END | Test Loss Quantized [%.3f]| Test Acc Quantized [%.3f]\n' % (test_loss, test_acc)
msg += '************* END *************\n'
print(msg)



# import torch.nn.functional as F
# T=4
# clusters = np.readtxt('./fashionmnist_clusters.txt', fmt='%d')
# n_clusters =
# cluster_entropy_baseline =
# for i in range(60000):
#
# teacher_soft_logits = F.softmax(teacher_logits / T, dim=1)
#
#
# pdb.set_trace()
# # if COT:
# plt.plot(cot_)
# plt.grid()
# plt.show()
# #test the model
#


