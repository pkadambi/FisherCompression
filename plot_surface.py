"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
# import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
# import dataloader

import pdb
import torch.backends.cudnn as cudnn
import tensorflow as tf
from utils.dataset_utils import *
from data import get_dataset
from utils.model_utils import *
from preprocess import get_transform


'''

n epochs = 300
cosine decay rate
weight decay

'''
#MUST KEEP THIS AS THE FIRST FLAG
tf.app.flags.DEFINE_string( 'dataset', 'cifar10', 'choose - mnist, fashionmnist, cifar10, cifar100')

tf.app.flags.DEFINE_string('noise_scale',None,'`inv_fisher` or `fisher`')

tf.app.flags.DEFINE_string('activation', None,'`tanh` or `relu`')
tf.app.flags.DEFINE_string('lr_decay_type', 'cosine', '`step` or `cosine`, defaults to cosine')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 300, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')
tf.app.flags.DEFINE_float('weight_decay', 2e-4, 'weight decay value')
# tf.app.flags.DEFINE_float('weight_decay', 100, 'weight decay value')
tf.app.flags.DEFINE_integer('inflate', None,'inflating factor for resnet (may need bigger factor if 1-b weights')

tf.app.flags.DEFINE_float('lr', .1, 'learning rate')

tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 4, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 4, 'number of bits weight')
tf.app.flags.DEFINE_float('eta', .0, 'noise eta')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, 'minimum quant value')
tf.app.flags.DEFINE_float('q_max', None, 'maximum quant value')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')
# tf.app.flags.DEFINE_string('regularization', 'distillation', 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_float('gamma', 0.01, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 0.005, 'diagonal loading constant')

# tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_string('optimizer', 'sgdr', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_boolean('lr_decay', True, 'Whether or not to decay learning rate')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
# tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')
tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw_buffer/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_buffer/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/fisher_buffer/Run1/', 'directory to load model from')

# tf.app.flags.DEFINE_string('savepath', './tmp', 'directory to save model to')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/distillation', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw_sgdr/', 'directory to load model from')


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

tf.app.flags.DEFINE_boolean('eval', False,'if this flag is enabled, the code doesnt write anyhting, it just loads from `FLAGS.savepath` and evaluates test acc once')

tf.app.flags.DEFINE_string('model', default='resnet', help='model name')


# direction parameters
tf.app.flags.DEFINE_string('dir_file', default='',help='specify the name of direction file, or the path to an eisting direction file')
tf.app.flags.DEFINE_string('dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')

tf.app.flags.DEFINE_string('x', default='-1.:1.:50', help='A string with format xmin:x_max:xnum')
tf.app.flags.DEFINE_string('y', default='-1.:1.:50', help='A string with format ymin:ymax:ynum')

# tf.app.flags.DEFINE_string('x', default='-0.7:0.7:35', help='A string with format xmin:x_max:xnum')
# tf.app.flags.DEFINE_string('y', default='-0.7:0.7:35', help='A string with format ymin:ymax:ynum')

# tf.app.flags.DEFINE_string('x', default='-.1:.1:10', help='A string with format xmin:x_max:xnum')
# tf.app.flags.DEFINE_string('y', default='-.1:.1:10', help='A string with format ymin:ymax:ynum')

tf.app.flags.DEFINE_string('xnorm', default='filter', help='direction normalization: filter | layer | weight')
tf.app.flags.DEFINE_string('ynorm', default='filter', help='direction normalization: filter | layer | weight')
tf.app.flags.DEFINE_string('xignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
tf.app.flags.DEFINE_string('yignore', default='biasbn', help='ignore bias and BN parameters: biasbn')
tf.app.flags.DEFINE_boolean('same_dir', default=False, help='use the same random direction for both x-axis and y-axis')
tf.app.flags.DEFINE_integer('idx', default=0, help='the index for the repeatness experiment')
tf.app.flags.DEFINE_string('surf_file', default='', help='customize the name of surface file, could be an existing file.')


tf.app.flags.DEFINE_string('model_file', default='', help='path to the trained model file')
tf.app.flags.DEFINE_string('model_file2', default='', help='path to the trained model file') #here just so that the ported code doesnt break
tf.app.flags.DEFINE_string('model_file3', default='', help='path to the trained model file') #here just so that the ported code doesnt break

#TODO:remove since this has been taken over by --loadpath flag

#TODO: use this flag to choose between distillation loss and normal xent loss
# parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

tf.app.flags.DEFINE_string('proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
tf.app.flags.DEFINE_float('loss_max', default=5., help='Maximum value to show in 1D plot')
tf.app.flags.DEFINE_float('vmax', default=10., help='Maximum value to map')
tf.app.flags.DEFINE_float('vmin', default=0.1, help='Miminum value to map')
tf.app.flags.DEFINE_float('vlevel', default=0.5, help='plot contours every vlevel')
tf.app.flags.DEFINE_boolean('show', default=True, help='show plotted figures')
tf.app.flags.DEFINE_boolean('log', default=False, help='use log scale for loss values')
tf.app.flags.DEFINE_boolean('plot', default=True, help='plot figures after computation')

tf.app.flags.DEFINE_float('xmin', default=0., help='min x value in grid')
tf.app.flags.DEFINE_float('xmax', default=0., help='max x value in grid')
tf.app.flags.DEFINE_float('xnum', default=0., help='Number of x in grid')

tf.app.flags.DEFINE_float('ymin', default=0., help='min y value in grid')
tf.app.flags.DEFINE_float('ymax', default=0., help='min y value in grid')
tf.app.flags.DEFINE_float('ynum', default=0., help='Number of y in grid')

#based on the flag below, the layers in the fwd pass will add the d_theta perturbation to the quantized weight
tf.app.flags.DEFINE_boolean('loss_surf_eval_d_qtheta', default=True, help='whether we are in loss surface generation mode')

tf.app.flags.DEFINE_string('loss_surf_type', default='test', help='whether to create test or train loss surface')

'''

NOTE: the following imports must be after flags declarations since these files query the flags 

'''

import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
FLAGS = tf.app.flags.FLAGS


#Torch gpu stuff
cudnn.benchmark = True
torch.cuda.set_device(0)

batch_size = FLAGS.batch_size
n_workers = 4
dataset = FLAGS.dataset
n_epochs = FLAGS.n_epochs
record_interval = FLAGS.record_interval

FLAGS.model_file = FLAGS.loadpath+FLAGS.model

criterion = nn.CrossEntropyLoss()




def name_surface_file(args, dir_file):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # # dataloder parameters
    # if args.raw_data: # without data normalization
    #     surf_file += '_rawdata'
    # if args.data_split > 1:
    #     surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, state, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies = [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    # pdb.set_trace()
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    # if args.loss_name == 'mse':
    #     criterion = nn.MSELoss()

    net.cuda()
    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # evaluation.test_model(net, criterion, testloader, printing=True)

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            # pdb.set_trace()
            net_plotter.set_weights(net, w, d, coord)
        elif args.dir_type == 'states':
            net_plotter.set_states(net, state, d, coord)

        # Record the time to compute the loss value
        loss_start = time.time()
        # loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda)
        loss, acc = test_model(dataloader, net, criterion, printing=False)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        # pdb.set_trace()

        #following lines bypassed since mpi.reduce_max just returns the array itself if comm is none
        # losses     = mpi.reduce_max(losses)
        # accuracies = mpi.reduce_max(accuracies)

        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

        #reset weights
        net_plotter.set_weights(net, w, d, np.array([0.,0.]))
    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    # for i in range(max(inds_nums) - len(inds)):
    #     losses = mpi.reduce_max(losses)
    #     accuracies = mpi.reduce_max(accuracies)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':



    torch.manual_seed(123)

    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    comm, rank, nproc = None, 0, 1

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------

    try:
        # pdb.set_trace()
        a1 = [float(a) for a in FLAGS.x.split(':')]
        FLAGS.xmin = a1[0]
        FLAGS.xmax = a1[1]
        FLAGS.xnum = a1[2]
        FLAGS.ymin, FLAGS.ymax, FLAGS.ynum = (None, None, None)

        if FLAGS.y:
            a1 = [float(a) for a in FLAGS.y.split(':')]
            FLAGS.ymin = a1[0]
            FLAGS.ymax = a1[1]
            FLAGS.ynum = a1[2]
            assert FLAGS.ymin and FLAGS.ymax and FLAGS.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------

    net = model_loader.load()

    #TODO: fix the two lines below

    w = net_plotter.get_weights(net) # initial parameters
    wts = net_plotter.get_weights(net) # initial parameters

    # [print(_.size()) for _ in wts]
    # pdb.set_trace()


    state = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    # if FLAGS.ngpu > 1:
    #     # data parallel with multiple GPUs on a single node
    #     net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(FLAGS) # name the direction file
    # pdb.set_trace()
    if rank == 0:
        net_plotter.setup_direction(FLAGS, dir_file, net)

    surf_file = name_surface_file(FLAGS, dir_file)
    if rank == 0:
        setup_surface_file(FLAGS, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    time.sleep(.05)

    # load directions
    d = net_plotter.load_directions(dir_file)
    print(len(d[0]))
    # exit()
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    # if rank == 0 and FLAGS.dataset == 'cifar10':
    #     torchvision.datasets.CIFAR10(root=FLAGS.dataset + '/data', train=True, download=True)

    # mpi.barrier(comm)
    time.sleep(.05)
    # TODO: add distillation choice
    criterion = nn.CrossEntropyLoss()

    #TODO: replace with correct data loading protocol
    train_data = get_dataset(name=dataset, split='train', transform=get_transform(name=dataset, augment=True))
    test_data = get_dataset(name=dataset, split='test', transform=get_transform(name=dataset, augment=False))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                              num_workers=n_workers, pin_memory=True)

    test_loss, test_acc = test_model(testloader, net, criterion, printing=False)
    msg = '\nRestored Model Accuracy: \t %.3f' % (test_acc)
    print(msg)

    # evaluation.test_model(net, criterion , testloader, printing=True)

    # exit()
    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, state, d, trainloader, 'train_loss', 'train_acc', comm, rank, FLAGS)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, FLAGS)
    print('here')
    #--------------------------------------------------------------------------
    # Plot figures
    #--------------------------------------------------------------------------
    if FLAGS.plot and rank == 0:
        if FLAGS.y and FLAGS.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, FLAGS.proj_file, 'train_loss', FLAGS.show)
        elif FLAGS.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', FLAGS.vmin, FLAGS.vmax, FLAGS.vlevel, FLAGS.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, FLAGS.xmin, FLAGS.xmax, FLAGS.loss_max, FLAGS.log, FLAGS.show)
