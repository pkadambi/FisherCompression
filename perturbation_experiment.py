'''


DETAILS OF THIS FILE:


In this experiment, we take a pretrained network on CIFAR10 (full precision) and perturb the weights of the network
with gaussian noise of identity covariance, gaussian noise with fisher covariance, and gaussian noise with inverse
fisher covariance


'''









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
tf.app.flags.DEFINE_string('noise_scale',None,'`inv_fisher` or `fisher`')
tf.app.flags.DEFINE_string( 'dataset', 'cifar10', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 300, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')
tf.app.flags.DEFINE_float('weight_decay', 2e-4, 'weight decay value')
# tf.app.flags.DEFINE_float('weight_decay', 100, 'weight decay value')
tf.app.flags.DEFINE_integer('inflate', None,'inflating factor for resnet (may need bigger factor if 1-b weights')

tf.app.flags.DEFINE_float('lr', .1, 'learning rate')

# tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_boolean('is_quantized', False, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 32, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 32, 'number of bits weight')
tf.app.flags.DEFINE_float('eta', .0, 'noise eta')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, 'minimum quant value')
tf.app.flags.DEFINE_float('q_max', None, 'maximum quant value')
tf.app.flags.DEFINE_integer('n_runs', 5, 'number of times to train network')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_float('gamma', 0.01, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 0.005, 'diagonal loading constant')

# tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_string('optimizer', 'sgdr', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_boolean('lr_decay', True, 'Whether or not to decay learning rate')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', './SavedModels/Resnet18/FP_perts', 'directory to load model from')


#TODO: find where this is actually used in the code
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')

#Distillation Params
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/Resnet18/FP_perts', 'path to FP model for loading for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation regularizer multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'temperature for distillation')

tf.app.flags.DEFINE_boolean('logging', False,'whether to enable writing to a logfile')
tf.app.flags.DEFINE_float('lr_end', 2e-4, 'learning rate at end of cosine decay')
tf.app.flags.DEFINE_boolean('constant_fisher', False,'whether to keep fisher/inv_fisher constant from when the checkpoint')

tf.app.flags.DEFINE_string('fisher_method', 'adam','which method to use when computing fisher')
tf.app.flags.DEFINE_boolean('layerwise_fisher', False,'whether or not to use layerwise fisher')
tf.app.flags.DEFINE_string('activation', None,'`tanh` or `relu`')


'''

NOTE: the following imports must be after flags declarations since these files query the flags 

'''
from sgdR import SGDR

from models.resnet_quantized import ResNet_cifar10
from models.resnet_binary import ResNet_cifar10_binary
FLAGS = tf.app.flags.FLAGS

flag_dict = FLAGS.flag_values_dict()

n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act


'''

Redefine any flags here if you want to run a sweep

'''


'''

Config file save information

'''

#Config string

fisher = FLAGS.regularization=='fisher'
l2 = FLAGS.regularization=='l2'
distillation = FLAGS.regularization=='distillation'
inv_fisher = FLAGS.regularization=='inv_fisher'


if FLAGS.regularization == 'fisher' or FLAGS.regularization=='l2' or FLAGS.regularization=='inf_fisher':
    reg_string = FLAGS.regularization

else:
    reg_string = ''


#Save path
if not FLAGS.debug and FLAGS.savepath is None:
    SAVEPATH = './SavedModels/Resnet18/%dba_%dbw' % (n_bits_act, n_bits_wt)
    if FLAGS.regularization is not None:
        SAVEPATH += '/' + FLAGS.regularization

elif not FLAGS.debug:
    SAVEPATH = FLAGS.savepath


etaval = FLAGS.eta

#Torch gpu stuff
cudnn.benchmark = True
torch.cuda.set_device(0)

batch_size = FLAGS.batch_size
n_workers = 6
dataset = FLAGS.dataset
n_epochs = FLAGS.n_epochs
record_interval = FLAGS.record_interval


criterion = nn.CrossEntropyLoss()

train_data = get_dataset(name = dataset, split = 'train', transform=get_transform(name=dataset, augment=True))
test_data = get_dataset(name = dataset, split = 'test', transform=get_transform(name=dataset, augment=False))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                          num_workers=n_workers, pin_memory=True)

n_runs = FLAGS.n_runs
test_accs=[]


for k in range(n_runs):
    i=0

    j=0

    while os.path.exists(os.path.join(SAVEPATH, 'Run%d' % j)):
        j+=1
    SAVEPATH_run = os.path.join(SAVEPATH, 'Run%d' % j)

    config_path = os.path.join(SAVEPATH_run, 'config_str.txt')
    os.makedirs(SAVEPATH_run, exist_ok=True)

    config_file = open(config_path, 'w+')


    SAVEPATH_run = os.path.join(SAVEPATH, 'Run%d' % j)


    dict_ind = list(flag_dict.keys()).index('dataset')
    keys = list(flag_dict.keys())
    values = list(flag_dict.values())
    for dict_ind_ in range(dict_ind, len(flag_dict)):
        config_file.write(keys[dict_ind_] + ':\t' + str(values[dict_ind_])+'\n')
        config_file.flush()


    if FLAGS.logging:
        logpath = SAVEPATH_run + '/logfile.txt'
        logfile = open(logpath, 'w+')

    model = ResNet_cifar10(is_quantized=FLAGS.is_quantized)
    # exit()
    model.cuda()
    # optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)

    if FLAGS.optimizer=='adam':
        optimizer = optim.SGD(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    elif FLAGS.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=2e-4)

    elif FLAGS.optimizer=='sgdr':
        # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        optimizer = SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)
        # CosineAnnealingLR()
    # print(model.conv1.quantize_input)
    # exit()
    logstr  = '\n**************************************\n'
    logstr += '\n********** TRAINING STARTED **********\n'
    logstr += '\n*************** RUN %d ***************\n' % k
    logstr += '\n**************************************\n'

    if FLAGS.loadpath is not None:
        loadpath = os.path.join(FLAGS.loadpath, 'Run%d' % (k), 'resnet')
        checkpoint = torch.load(loadpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        optimizer.param_groups[0]['lr'] = FLAGS.lr

        for l, (name, layer) in enumerate(model.named_modules()):
            if 'conv' in name or 'fc' in name:
                if hasattr(layer, 'weight'):
                    layer.set_min_max()

        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
        msg = '\nRestored Model Accuracy: \t %.3f' % (test_acc)
        logstr+=msg
        config_file.write(msg)
        # exit()

    #Load the exact same model as the 'teacher' model
    checkpoint = torch.load(loadpath)

    teacher_model = ResNet_cifar10(is_quantized=False, )
    teacher_model.cuda()
    teacher_model.load_state_dict(checkpoint['model_state_dict'])


    for param in teacher_model.parameters():
        param.requires_grad = False


    test_loss, test_acc = test_model(test_loader, teacher_model, criterion, printing=False, eta=etaval)
    msg += '\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc)
    logstr += msg
    config_file.write(logstr)

    # exit()
    print(logstr)
    if FLAGS.logging:
        logfile.write(logstr)
        logfile.flush()

    config_file.flush()


    loss_save = []
    msqe_save = []
    regularizer_save = []
    fim_trace_save = []
    fmsqe_save_g2 = []
    fmsqe_save_adam = []
    fmsqe_save_original_adam = []
    fim_trace_save_g2 = []
    fim_trace_save_adam = []

    pearson_fisher = []
    spearman_fisher = []

    pearson_regularizer = []
    spearman_regularizer = []

    orig_fisher = []
    orig_inv_fish = []


    for epoch in range(n_epochs):


        logstr='****** NEXT EPOCH ******\n'
        model.train()
        start = time.time()

        n_batches = 0

        '''
        
        We compute fisher information over the whole training set
        
        '''
        for iter, (inputs, targets) in enumerate(train_loader):


            inputs = inputs.cuda()
            targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)

            #adding distillation loss
            # if distillation:
            #     teacher_model.eval()
            #     teacher_output = Variable(teacher_model(inputs, eta=etaval), requires_grad=False)
                # teacher_output = teacher_output.long()
                # teach_train_acc = accuracy(teacher_output, targets).item()
                # print('Teacher Acc: '+ str(teach_train_acc ))
                # kd_loss = FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)
                # pdb.set_trace()
                # loss = loss + kd_loss


            # optimizer.zero_grad()
            loss.backward()
            n_batches+=1
            print(n_batches)


            for group in optimizer.param_groups:
                for p in group['params']:
                    if hasattr(p, 'pert'):
                        if FLAGS.fisher_method == 'adam':
                            p.fisher = p.fisher + optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                            p.orig_fisher = p.orig_fisher + optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                        elif FLAGS.fisher_method == 'g2':
                            p.fisher = p.fisher + p.grad * p.grad * FLAGS.batch_size


            for group in optimizer.param_groups:
                for p in group['params']:
                    if hasattr(p, 'pert'):
                        max_orig_fisher = torch.max(max_orig_fisher, torch.max(p.orig_fisher))

            #-----------------------------------------
            #fisher loss without messing up gradient flow
            perts = []
            for l, (name, layer) in enumerate(model.named_modules()):
                if 'conv' in name or 'fc' in name:

                    with torch.no_grad():
                        if hasattr(layer, 'is_quantized'):
                            layer.weight.pert = torch.zeros_like(layer.weight)
                            layer.weight.org = layer.weight.data.clone()
                            # print(name)

            max_orig_fisher = torch.tensor(0.).cuda()
            max_inv_fisher = torch.tensor(0.).cuda()

            # if epoch==0 and iter==0:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if hasattr(p, 'pert'):
                        if FLAGS.fisher_method=='adam':
                            p.fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                            p.orig_fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                            max_orig_fisher = torch.max(max_orig_fisher, torch.max(p.orig_fisher))

                        elif FLAGS.fisher_method=='g2':
                            p.fisher = p.grad * p.grad * FLAGS.batch_size

                        # max_inv_fisher = torch.max(max_inv_fisher, torch.max(p.inv_FIM))

            # print(max_orig_fisher)
            layernum=0

            orig_wts = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if hasattr(p, 'pert'):


                        if FLAGS.layerwise_fisher:
                            p.fisher = p.fisher / torch.max(p.fisher)
                            p.inv_FIM = 1e-7 * 1/(p.fisher + 1e-7)
                            p.inv_FIM = p.inv_FIM / torch.max(p.inv_FIM)
                            # p.inv_FIM = 1 - p.fisher + 1e-7

                        else:
                            # p.inv_FIM = 1 / (p.fisher + 5e-3) * 5e-3
                            # p.inv_FIM = p.inv_FIM * 1e-2

                            p.fisher = p.fisher / max_orig_fisher + 5e-3
                            p.inv_FIM = 1 / (p.fisher + 1e-6) * 1e-6
                            # p.inv_FIM = 1e-7 * 1/(p.fisher + 1e-7)

                            # p.inv_FIM = 1 - p.fisher + 1e-7
                            # p.inv_FIM = 1 - p.fisher + 1e-7

                        orig_wts.append(p.data.view(-1).detach().cpu().numpy())
                        orig_fisher.append(p.fisher.view(-1).cpu().numpy())

                        # p.inv_FIM = p.inv_FIM / max_inv_fisher
                        print('Computed Fisher for layer:\t' + str(layernum))
                        layernum+=1
                        # p.inv_FIM[p.inv_FIM>1e4] = 1e4
                        # inv_f = p.inv_FIM.view(-1).cpu().numpy() * max_orig_fisher.cpu().numpy()
                        inv_f = p.inv_FIM.view(-1).cpu().numpy()
                        orig_inv_fish.append(inv_f)

            # orig_wts = np.concatenate(orig_wts)
            # orig_fisher = np.concatenate(orig_fisher)
            # print(np.percentile(orig_fisher, 5))
            # print(np.percentile(orig_fisher, 10))
            # print(np.percentile(orig_fisher, 50))
            # print(np.percentile(orig_fisher, 95))
            #
            #
            # print()
            # exit()
            optimizer.zero_grad()
            N_MC_ITERS = 30
            N_noise_level = 40

            test_acc_matrix = np.zeros([N_MC_ITERS, N_noise_level])
            average_noise_magnitudes = np.zeros([N_MC_ITERS, N_noise_level])
            fisher_loss = np.zeros([N_MC_ITERS, N_noise_level])
            kl_loss = np.zeros([N_MC_ITERS, N_noise_level])

            kl_criterion = nn.KLDivLoss()

            for zz in range(N_noise_level):

                for kk in range(N_MC_ITERS):
                    #Step 2: calculate noise
                    # pdb.set_trace()
                    num_wts = 0
                    pert_mag = 0
                    fim = []
                    inv_fisher = []

                    fisher_loss_ = 0.

                    for p in model.parameters():
                        if hasattr(p, 'fisher'):
                            # pdb.set_trace()
                            # p.noise = .25 * (1.5/(N_noise_level-zz)) * torch.randn(size=p.data.size(), device='cuda')
                            p.noise = .15 * (1.5/(N_noise_level-zz)) * torch.randn(size=p.data.size(), device='cuda')
                            p.noise = 3. * (1.5/(N_noise_level-zz)) * torch.randn(size=p.data.size(), device='cuda')

                            #Step 2a: scale if needed
                            if FLAGS.noise_scale=='fisher':
                                p.noise = p.fisher * p.noise
                            if FLAGS.noise_scale == 'fisher_randperm':
                                p.noise = p.fisher * p.noise
                                orig_size = p.noise.size()
                                p.noise = p.noise.view(-1)
                                p.noise = p.noise[torch.randperm(list(p.noise.size())[0])]
                                p.noise = p.noise.view(orig_size)
                            elif FLAGS.noise_scale=='fisher_nullspace':
                                p.noise = (p.fisher < 1e-6).float() * p.noise
                            elif FLAGS.noise_scale=='inv_fisher':
                                p.noise = p.inv_FIM * p.noise

                            # print(p.fisher.size())
                            # print(p.noise.size())
                            fisher_loss_ += torch.sum(p.fisher * p.noise * p.noise)

                            #step 3: calculate perturbation average magnitude
                            num_wts += p.data.numel()
                            pert_mag += torch.sum(torch.abs(p.noise))

                            # noise_mag = torch.sum((1/p.data.numel()) * torch.abs(p.noise))


                            #step 4: noise weights
                            p.data = p.data + p.noise
                            fim.append(p.fisher.view(-1).cpu().numpy())
                            inv_fisher.append(p.inv_FIM.view(-1).cpu().numpy())


                    # plt.subplot(221)
                    # plt.title('Fisher Histogram')
                    # fishers = np.concatenate(fim)
                    # inv_fisher = np.concatenate(inv_fisher)
                    # print('Maximum Fisher Value')
                    # print(max(fishers))
                    # hist_, bins, _ = plt.hist(fishers, bins=1000)
                    # plt.yscale('log')
                    #
                    # plt.subplot(222)
                    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
                    # plt.title('Fisher Log Histogram')
                    # plt.hist(fishers, bins=logbins)
                    # plt.xscale('log')
                    # plt.yscale('log')
                    #
                    # plt.subplot(223)
                    # plt.title('Inv Fisher Histogram')
                    # hist_, bins, _ = plt.hist(inv_fisher, bins=500)
                    # plt.yscale('log')
                    # plt.ylabel('Frequency')
                    # plt.xlabel('Magnitude')
                    # plt.subplot(224)
                    # logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
                    # plt.title('Inv Fisher Log Histogram')
                    # plt.hist(inv_fisher, bins=logbins)
                    # plt.xscale('log')
                    # plt.yscale('log')
                    #
                    # plt.show()
                    # exit()


                    # #step 5: test
                    test_loss, test_acc, kl_loss_ = test_model(test_loader, model, criterion, printing=False,
                                                     eta=etaval, teacher_model=teacher_model)


                    average_noise_magnitudes[kk, zz] = pert_mag/num_wts
                    test_acc_matrix[kk,zz] = test_acc
                    kl_loss[kk,zz] = kl_loss_
                    fisher_loss[kk,zz] = fisher_loss_

                    # step 6: reload old weights
                    for p in model.parameters():
                        if hasattr(p, 'noise'):
                            p.data = p.org.clone()


                print('\n\nAverage Noise Level:\t' + str(np.mean(average_noise_magnitudes[:, zz], axis=0)))
                print('Avg Test Acc Noise Level:\t' + str(np.mean(test_acc_matrix[:, zz], axis=0)))
                print('Avg Fisher Loss:\t' + str(np.mean(fisher_loss[:, zz], axis=0)))
                print('Avg KL Div Loss:\t' + str(np.mean(kl_loss[:, zz], axis=0)))
                print('Avg fisher/kl_ratio:\t' + str(np.mean(fisher_loss[:, zz], axis=0)/np.mean(kl_loss[:, zz], axis=0)))


            np.savetxt('./noisy_accs.txt', test_acc_matrix)
            np.savetxt('./noise_magnitudes.txt', average_noise_magnitudes)
            np.savetxt('./fisher_loss.txt', fisher_loss)
            np.savetxt('./kl_loss.txt', kl_loss)
            exit()

