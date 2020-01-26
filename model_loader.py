import os
from models.resnet_quantized import ResNet_cifar10
from models.resnet_lowp import ResNet_cifar10_lowp
from models.resnet_binary import ResNet_cifar10_binary
import torch

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def load():

    #step1 - instantiate model
    if FLAGS.dataset is 'cifar10':
        NUM_CLASSES=10
    elif FLAGS.dataset is 'cifar100':
        NUM_CLASSES=100
    else:
        NUM_CLASSES=10

    if FLAGS.n_bits_wt<=2:
        model = ResNet_cifar10_lowp(is_quantized=FLAGS.is_quantized, inflate=FLAGS.inflate, activation=FLAGS.activation,
                                    num_classes=NUM_CLASSES)
    else:
        model = ResNet_cifar10(is_quantized=FLAGS.is_quantized, inflate=FLAGS.inflate, activation=FLAGS.activation,
                               num_classes=NUM_CLASSES)

    #step2 - restore model

    if FLAGS.loadpath is not None:
        loadpath = os.path.join(FLAGS.loadpath, 'resnet')
        print('Restoring model to train from:\t' + loadpath)
        checkpoint = torch.load(loadpath)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        exit('ERROR NO LOADPATH PROVIDED!!!')

    model.cuda()
    return model
