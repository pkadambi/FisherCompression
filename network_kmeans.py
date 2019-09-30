from data import get_dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualization_utils import *
from scipy.cluster.vq import kmeans
import torch.optim as optim
import tensorflow as tf
import time as time
import numpy as np
import os

'''

This script loads a model and then performs k-means on the weights of a pre-trained model

The idea is that this model will be able to 

'''


tf.app.flags.DEFINE_string( 'dataset', 'fashionmnist', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 100, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')

tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 8, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 8, 'number of bits weight')
tf.app.flags.DEFINE_float('eta', .0, 'noise eta')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, 'minimum quant value')
tf.app.flags.DEFINE_float('q_max', None, 'maximum quant value')
tf.app.flags.DEFINE_integer('n_runs', 1, 'number of times to train network')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2` or `fisher`')
tf.app.flags.DEFINE_float('gamma', 0.01, 'gamma value')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')


loadpath = './SavedModels/Lenet/FPa_FPw'

loadpath = os.path.join(loadpath, 'Run%d' % (3))
loadpath = loadpath + '/lenet'

model = models.Lenet5()

checkpoint = torch.load(loadpath)
model.load_state_dict(checkpoint['model_state_dict'])
params = []
params = np.array([])

for i, p in enumerate(model.parameters()):
    wt = p.detach().cpu().numpy()
    params = np.append(params, wt.ravel())

params = np.expand_dims(params, axis=1)
params = params[0::2]
centroids = kmeans(params, 5)
centroids = centroids[0]

print(centroids)
print(min(centroids))
print(max(centroids))



# 4 bits weight [-0.2836685]
# [0.261567934]
# use .25


# 2 bits weight [-0.18037436]
# [0.1658216]
#use .15


# 1 bit binary [-0.13865472]
# [0.11845241]
#use .11
