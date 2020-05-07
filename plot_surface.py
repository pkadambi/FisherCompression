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
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw_buffer/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_teq1/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_teq2/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_teq3/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_buffer/Run0/', 'directory to load model from')
# tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/fisher_buffer/Run1/', 'directory to load model from')
tf.app.flags.DEFINE_string('loadpath', './SavedModels/cifar10/Resnet18/4ba_4bw/msqe_buffer/Run0/', 'directory to load model from')

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

# tf.app.flags.DEFINE_string('x', default='-1.:1.:50', help='A string with format xmin:x_max:xnum')
# tf.app.flags.DEFINE_string('y', default='-1.:1.:50', help='A string with format ymin:ymax:ynum')

# tf.app.flags.DEFINE_string('x', default='-0.7:0.7:35', help='A string with format xmin:x_max:xnum')
# tf.app.flags.DEFINE_string('y', default='-0.7:0.7:35', help='A string with format ymin:ymax:ynum')

tf.app.flags.DEFINE_string('x', default='-0.75:0.75:30', help='A string with format xmin:x_max:xnum')
tf.app.flags.DEFINE_string('y', default='-0.75:0.75:30', help='A string with format ymin:ymax:ynum')

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
tf.app.flags.DEFINE_boolean('plot', default=False, help='plot figures after computation')

tf.app.flags.DEFINE_float('xmin', default=0., help='min x value in grid')
tf.app.flags.DEFINE_float('xmax', default=0., help='max x value in grid')
tf.app.flags.DEFINE_float('xnum', default=0., help='Number of x in grid')

tf.app.flags.DEFINE_float('ymin', default=0., help='min y value in grid')
tf.app.flags.DEFINE_float('ymax', default=0., help='min y value in grid')
tf.app.flags.DEFINE_float('ynum', default=0., help='Number of y in grid')

#based on the flag below, the layers in the fwd pass will add the d_theta perturbation to the quantized weight
tf.app.flags.DEFINE_boolean('loss_surf_eval_d_qtheta', default=True, help='whether we are in loss surface generation mode')

tf.app.flags.DEFINE_string('loss_surf_type', default='test', help='whether to create test or train loss surface')
# tf.app.flags.DEFINE_string('loss_surf_type', default='train', help='whether to create test or train loss surface')

tf.app.flags.DEFINE_string('loss_landscapes_directory', default=None, help='if specified, include subfolder /models/ with the models to calc loss fn for, also specify n_directions')
# './SavedModels/cifar10/Resnet18/4ba_4bw/loss_landscape_results'
tf.app.flags.DEFINE_integer('n_directions', default=5, help='number of directions to evaluate')


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


def save_xyz_surf_info(surf_file, results_dir, results_file, surf_name='train_loss'):
    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])

    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    np.savetxt(results_dir + 'X.txt', X, '%.6f', delimiter=',')
    np.savetxt(results_dir + 'Y.txt', Y, '%.6f', delimiter=',')
    np.savetxt(results_file         , Z, '%.6f', delimiter=',')

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

    # TODO: add distillation choice
    criterion = nn.CrossEntropyLoss()

    train_data = get_dataset(name=dataset, split='train', transform=get_transform(name=dataset, augment=True))
    test_data = get_dataset(name=dataset, split='test', transform=get_transform(name=dataset, augment=False))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                              num_workers=n_workers, pin_memory=True)

    if FLAGS.loss_landscapes_directory is None:
        '''
        If the direction directory is none, just load the model from FLAGS.loadpath and continue with that
        '''
        #--------------------------------------------------------------------------
        # Load models and extract parameters
        #--------------------------------------------------------------------------

        net = model_loader.load(FLAGS.loadpath)

        #TODO: fix the two lines below

        w = net_plotter.get_weights(net) # initial parameters
        # wts = net_plotter.get_weights(net) # initial parameters

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


        test_loss, test_acc = test_model(testloader, net, criterion, printing=False)
        msg = '\nRestored Model Accuracy: \t %.3f' % (test_acc)
        # print(msg)
        # print(dir_file)
        # print(surf_file)

        # evaluation.test_model(net, criterion , testloader, printing=True)

        # exit()
        #--------------------------------------------------------------------------
        # Start the computation
        #--------------------------------------------------------------------------

        if FLAGS.loss_surf_type=='train':
            crunch(surf_file, net, w, state, d, trainloader, 'train_loss', 'train_acc', comm, rank, FLAGS)
        elif FLAGS.loss_surf_type == 'train':
            crunch(surf_file, net, w, state, d, testloader, 'test_loss', 'test_acc', comm, rank, FLAGS)
        else:
            exit('ERROR: Incorrect loss surf type specified, must be `train` or `test`')

        #--------------------------------------------------------------------------
        # Plot figures
        #--------------------------------------------------------------------------


        # print(surf_file)
        # exit()
        if FLAGS.plot and rank == 0:
            if FLAGS.y and FLAGS.proj_file:
                plot_2D.plot_contour_trajectory(surf_file, dir_file, FLAGS.proj_file, 'train_loss', FLAGS.show)
            elif FLAGS.y:
                plot_2D.plot_2d_contour(surf_file, 'train_loss', FLAGS.vmin, FLAGS.vmax, FLAGS.vlevel, FLAGS.show)
            else:
                plot_1D.plot_1d_loss_err(surf_file, FLAGS.xmin, FLAGS.xmax, FLAGS.loss_max, FLAGS.log, FLAGS.show)

    else:
        working_dir = FLAGS.loss_landscapes_directory
        grid_config_str = 'grid_x_%s_y%s' % (FLAGS.x, FLAGS.y) + '_%s' % FLAGS.loss_surf_type
        directions_dir = os.path.join(working_dir, 'directions')
        models_dir = os.path.join(working_dir, 'models')
        results_folder = os.path.join(working_dir, 'results/'+grid_config_str)

        if FLAGS.loss_surf_type == 'test':
            surf_name = 'test_loss'
        elif FLAGS.loss_surf_type == 'train':
            surf_name = 'train_loss'
        else:
            exit('ERROR: loss surf type must be either `test` or `train`')

        #FUNCT 1: Setup directories
        for i in range(FLAGS.n_directions):

            #TODO: Step 1: check if direction file/directory exists, if it doesnt create it (This is common across all models for this run)
            #basis directory is independent of the grid density
            direction_folder = os.path.join(directions_dir, '%d/' % i)
            # exit()

            #name the direction file (based on ynum, etc)

            #TODO: add argument to function for supplied dir_file_directory
            dir_file = net_plotter.name_direction_file(FLAGS, direction_directory=direction_folder)  # name the direction file


            if rank == 0:
                if os.path.exists(dir_file):
                    net_plotter.setup_direction(FLAGS, dir_file)
                else:
                    os.makedirs(direction_folder, exist_ok=True)
                    #set up network here
                    ste_model_dir = os.path.join(models_dir, 'ste/resnet')
                    net = model_loader.load(loadpath = ste_model_dir)

                    net_plotter.setup_direction(FLAGS, dir_file, net)

            # print(dir_file)
            # continue

            results_dir = os.path.join(results_folder, '%d/' % i)



            d = net_plotter.load_directions(dir_file) #load direction file

            model_directories = os.listdir(models_dir)  # get all models in subdirectories

            for model_subfolder in model_directories:
                # print(model_subfolder)
                #TODO: correctly setup flags before instantiating network (ie regularization type, etc.)
                model_loadpath = os.path.join(models_dir, model_subfolder + '/%s' % FLAGS.model)

                if 'l2' in model_subfolder or 'msqe' in model_subfolder:
                    FLAGS.regularization = 'l2'
                    results_file = os.path.join(results_dir, 'msqe.txt')
                elif 'distillation' in model_subfolder:
                    FLAGS.regularization = 'distillation'
                    results_file = os.path.join(results_dir, model_subfolder+'.txt')
                elif 'fisher' in model_subfolder:
                    FLAGS.regularization = 'fisher'
                    results_file = os.path.join(results_dir, 'fisher.txt')

                elif 'inv_fisher' in model_subfolder:
                    FLAGS.regularization = 'inv_fisher'
                    results_file = os.path.join(results_dir, 'inv_fisher.txt')
                elif 'ste' in model_subfolder:
                    FLAGS.regularization = None
                    results_file = os.path.join(results_dir, 'STE.txt')

                # print(model_loadpath)

                #HACKY AF CODE: reimport model loader to apply the FLAGS changes
                import model_loader #TODO: is this import statement really needed?
                net = model_loader.load(model_loadpath)
                test_loss, test_acc = test_model(testloader, net, criterion, printing=False)
                print('\nRestored Model Accuracy: \t %.3f' % (test_acc))

                w = net_plotter.get_weights(net)  # initial parameters

                #TODO: (next after getting this script workinbg) also set temperature flag so that we can get the distillation loss landscape

                surf_dir = os.path.join(models_dir, os.path.join(model_subfolder, grid_config_str))
                surf_dir = os.path.join(surf_dir, '%d/' % i)

                #TODO: split crunching into it's own function/loop, this will help code maintainability going forward (wont be as nightmare like train_resnet18.py)
                # TODO: Step 2: check if surf file exists, if not create it (this should go in the model directory? or results dir)
                if not os.path.exists(surf_dir):
                    os.makedirs(surf_dir, exist_ok=True)

                surf_dir = net_plotter.name_direction_file(FLAGS, direction_directory=surf_dir)  # name the direction file
                surf_file = name_surface_file(FLAGS, surf_dir)
                setup_surface_file(FLAGS, surf_file, dir_file)
                # print(results_file)
                # continue


                # TODO: step 3a: check if results directory exists, if not, create it (this dirname should be based on the xnum/ynum/xmin/ymin/etc.)

                state = copy.deepcopy(net.state_dict())  # deepcopy since state_dict are references

                if 'test' in surf_name:
                    crunch(surf_file, net, w, state, d, testloader, surf_name, 'test_acc', comm, rank, FLAGS)

                if 'train' in surf_name:
                    crunch(surf_file, net, w, state, d, trainloader, 'train_loss', 'train_acc', comm, rank, FLAGS)


                os.makedirs(results_dir, exist_ok=True)


                # TODO: Step 3b: if results file not exist, crunch the numbers

                # TODO: Step 3c: save the Z.txt in the results directory
                save_xyz_surf_info(surf_file, results_dir, results_file, surf_name=surf_name)

            FLAGS.show=False

            if FLAGS.plot and rank == 0:
                if FLAGS.y and FLAGS.proj_file:
                    plot_2D.plot_contour_trajectory(surf_file, dir_file, FLAGS.proj_file, 'train_loss', FLAGS.show)
                elif FLAGS.y:
                    plot_2D.plot_2d_contour(surf_file, 'train_loss', FLAGS.vmin, FLAGS.vmax, FLAGS.vlevel, FLAGS.show)
                else:
                    plot_1D.plot_1d_loss_err(surf_file, FLAGS.xmin, FLAGS.xmax, FLAGS.loss_max, FLAGS.log, FLAGS.show)
