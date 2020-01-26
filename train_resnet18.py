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
tf.app.flags.DEFINE_integer('n_runs', 5, 'number of times to train network')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')
# tf.app.flags.DEFINE_string('regularization', 'distillation', 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_float('gamma', 0.01, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 0.005, 'diagonal loading constant')

# tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_string('optimizer', 'sgdr', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_boolean('lr_decay', True, 'Whether or not to decay learning rate')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')

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

tf.app.flags.DEFINE_boolean('loss_surf_eval_d_qtheta', default=False, help='whether we are in loss surface generation mode')


'''

NOTE: the following imports must be after flags declarations since these files query the flags 

'''

from sgdR import SGDR
from adamR import AdamR
from models.resnet_quantized import ResNet_cifar10
from models.resnet_lowp import ResNet_cifar10_lowp
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
distillation_fisher = FLAGS.regularization=='distillation_fisher'
inv_fisher = FLAGS.regularization=='inv_fisher'

if FLAGS.dataset=='cifar10':
    NUM_CLASSES=10
elif FLAGS.dataset=='cifar100':
    NUM_CLASSES=100



if FLAGS.regularization == 'fisher' or FLAGS.regularization=='l2' or FLAGS.regularization=='inf_fisher':
    reg_string = FLAGS.regularization
else:
    reg_string = ''


#Save path
if not FLAGS.debug and FLAGS.savepath is None:
    SAVEPATH = './SavedModels/%s/Resnet18/%dba_%dbw' % (FLAGS.dataset, n_bits_act, n_bits_wt)
    if FLAGS.regularization is not None:
        SAVEPATH += '/' + FLAGS.regularization
elif not FLAGS.debug:
    SAVEPATH = FLAGS.savepath


etaval = FLAGS.eta

#Torch gpu stuff
cudnn.benchmark = True
torch.cuda.set_device(0)

batch_size = FLAGS.batch_size
n_workers = 4
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
    print(SAVEPATH_run)
    # exit()
    config_path = os.path.join(SAVEPATH_run, 'config_str.txt')

    SAVEPATH_run = os.path.join(SAVEPATH, 'Run%d' % j)
    dict_ind = list(flag_dict.keys()).index('dataset')
    keys = list(flag_dict.keys())
    values = list(flag_dict.values())

    if FLAGS.logging and not FLAGS.eval:
        os.makedirs(SAVEPATH_run, exist_ok=True)

        logpath = SAVEPATH_run + '/logfile.txt'
        logfile = open(logpath, 'w+')
        config_file = open(config_path, 'w+')

        for dict_ind_ in range(dict_ind, len(flag_dict)):
            config_file.write(keys[dict_ind_] + ':\t' + str(values[dict_ind_])+'\n')
            config_file.flush()




    if FLAGS.activation == 'tanh':
        activation = nn.Hardtanh()
    elif FLAGS.activation is None:
        activation = nn.ReLU()
        FLAGS.activation = 'relu'

    if n_bits_wt<=2:
        # model = ResNet_cifar10_lowp(is_quantized=FLAGS.is_quantized, inflate=FLAGS.inflate)
        model = ResNet_cifar10_lowp(is_quantized=FLAGS.is_quantized, inflate=FLAGS.inflate, activation=activation,
                                    num_classes=NUM_CLASSES)
    else:
        model = ResNet_cifar10(is_quantized=FLAGS.is_quantized, inflate=FLAGS.inflate, activation=activation,
                               num_classes=NUM_CLASSES)

    model.cuda()
    # optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)

    if FLAGS.optimizer=='adam':
        optimizer = optim.SGD(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    elif FLAGS.optimizer=='sgd':
        optimizer = optim.SGD(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)

    elif FLAGS.optimizer=='sgdr':
        # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        optimizer = SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)

    elif FLAGS.optimizer == 'adamr':
        # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
        optimizer = AdamR(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)
        # CosineAnnealingLR()



    #TODO: abstract into utils file
    if FLAGS.lr_decay_type=='cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)
    else:
        lr_scheduler = None
    # print(model.conv1.quantize_input)
    # exit()
    logstr  = '\n**************************************\n'
    logstr += '\n********** TRAINING STARTED **********\n'
    logstr += '\n*************** RUN %d ***************\n' % k
    logstr += '\n**************************************\n'

    if FLAGS.loadpath is not None:
        loadpath = os.path.join(FLAGS.loadpath, 'Run%d' % (k), 'resnet')
        print('Restoring model to train from:\t' + loadpath)
        checkpoint = torch.load(loadpath)


        for l, (name, layer) in enumerate(model.named_modules()):
            # print(name)
            if FLAGS.q_min is None and FLAGS.q_max is None:
                if 'input' in name:
                    # print('\n'+name)

                    pass
                    # print('Min/Max:\t' + str(layer.running_min) + '\t' + str(layer.running_max))
                elif 'conv' in name or 'fc' in name:
                    # print('\n'+name)

                    if hasattr(layer, 'weight'):
                        if hasattr(layer.weight, 'min_value') is None:
                            print('NO weight.min_value saved before restore')
                        print(layer.weight.min())
                    break
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        optimizer.param_groups[0]['lr'] = FLAGS.lr
        # test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
        # print(test_acc)

        #TODO: rewrite this code for setting min/max (for visibility keep running_min/max instead of set_min_max())---------------------

        # for l, (name, layer) in enumerate(model.named_modules()):
        #     # print(name)
        #     if FLAGS.q_min is None and FLAGS.q_max is None:
        #         if 'input' in name:
        #             # print('\n'+name)
        #
        #             pass
        #             # print('Min/Max:\t' + str(layer.running_min) + '\t' + str(layer.running_max))
        #         elif 'conv' in name or 'fc' in name:
        #             # print('\n'+name)
        #
        #             if hasattr(layer, 'weight'):
        #                 print(layer.weight.min_value)
        #                 print(layer.weight.min())
        #
        #                 layer.set_min_max()
        #                 print('Min/Max:\t'+str(layer.weight.min_value)+'\t'+str(layer.weight.max_value))
        #                 exit()
        #----------------------------------------------------------------
        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
        msg = '\nRestored Model Accuracy: \t %.3f' % (test_acc)
        logstr+=msg
        # config_file.write(msg)
        # print(test_acc)
        # exit()

    if distillation or distillation_fisher:
        checkpoint = torch.load(FLAGS.fp_loadpath)
        print('Restoring teacher model from:\t' + FLAGS.fp_loadpath)

        teacher_model = ResNet_cifar10(is_quantized=False, num_classes=NUM_CLASSES)
        teacher_model.cuda()
        teacher_model.load_state_dict(checkpoint['model_state_dict'])


        for param in teacher_model.parameters():
            param.requires_grad = False


        test_loss, test_acc = test_model(test_loader, teacher_model, criterion, printing=False, eta=etaval)
        msg += '\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc)
        logstr += msg

        if FLAGS.eval:
            print(msg)
            exit('Finished evaluating model')

        config_file.write(logstr)
    # Accs: [93.72]

    print(logstr)
    # exit()

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


        if epoch>150 and FLAGS.n_bits_wt<=2:
            for group in optimizer.param_groups:
                for p in group['params']:
                    group['weight_decay']=0.

        logstr='****** NEXT EPOCH ******\n'
        model.train()
        start = time.time()

        for iter, (inputs, targets) in enumerate(train_loader):


            inputs = inputs.cuda()
            targets = targets.cuda()
            # print(targets)
            # exit()
            output = model(inputs)
            loss = criterion(output, targets)

            #adding distillation loss
            if distillation or distillation_fisher:
                teacher_model.eval()
                teacher_output = Variable(teacher_model(inputs, eta=etaval), requires_grad=False)
                # teacher_output = teacher_output.long()
                # teach_train_acc = accuracy(teacher_output, targets).item()
                # print('Teacher Acc: '+ str(teach_train_acc ))
                kd_loss = FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)

                loss = loss + kd_loss


            optimizer.zero_grad()
            loss.backward()

            #fisher loss without messing up gradient flow
            perts = []
            if FLAGS.is_quantized:
                for l, (name, layer) in enumerate(model.named_modules()):
                    if 'conv' in name or 'fc' in name:
                        with torch.no_grad():
                            if hasattr(layer, 'qweight'):

                                pertw = layer.weight - layer.qweight
                                layer.weight.pert = pertw
                                perts.append(pertw)

            # if FLAGS.regularization=='fisher' and epoch==0 and iter==0:
            if FLAGS.loadpath is not None:

                max_orig_fisher = torch.tensor(0.).cuda()
                max_inv_fisher = torch.tensor(0.).cuda()

                if epoch==0 and iter==0:
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
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if hasattr(p, 'pert'):

                                orig_fisher.append(p.fisher.view(-1).cpu().numpy()/max_orig_fisher.cpu().numpy())
                                # orig_inv_fish.append(p.inv_FIM.view(-1).cpu().numpy()/max_inv_fisher.cpu().numpy())

                                if FLAGS.layerwise_fisher:
                                    p.fisher = p.fisher / torch.max(p.fisher)
                                    # p.inv_FIM = p.inv_FIM / torch.max(p.inv_FIM)
                                else:
                                    p.fisher = p.fisher / max_orig_fisher

                                p.inv_FIM = 1 / (p.fisher + 1e-7)
                                p.inv_FIM[p.inv_FIM>1e4] = 1e4
                                p.inv_FIM = p.inv_FIM/1e4
                                inv_f = p.inv_FIM.view(-1).cpu().numpy() * max_orig_fisher.cpu().numpy()
                                orig_inv_fish.append(inv_f)

                elif distillation_fisher:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if hasattr(p, 'pert'):
                                if FLAGS.fisher_method=='adam':
                                    p.fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                                    max_fisher = torch.max(p.fisher)
                                elif FLAGS.fisher_method=='g2':
                                    p.fisher = p.grad * p.grad * FLAGS.batch_size


                                # max_inv_fisher = torch.max(max_inv_fisher, torch.max(p.inv_FIM))

                    # print(max_orig_fisher)
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if hasattr(p, 'pert'):

                                if FLAGS.layerwise_fisher:
                                    p.fisher = p.fisher / torch.max(p.fisher)
                                    # p.inv_FIM = p.inv_FIM / torch.max(p.inv_FIM)
                                else:
                                    p.fisher = p.fisher / max_orig_fisher

                                # p.inv_FIM = 1 / (p.fisher + 1e-7)
                                # p.inv_FIM[p.inv_FIM>1e4] = 1e4
                                # p.inv_FIM = p.inv_FIM/1e4
                                # inv_f = p.inv_FIM.view(-1).cpu().numpy() * max_orig_fisher.cpu().numpy()
                                # orig_inv_fish.append(inv_f)


                else:

                    if not FLAGS.constant_fisher:

                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if hasattr(p, 'pert'):
                                    if FLAGS.fisher_method == 'adam':
                                        p.fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size
                                    elif FLAGS.fisher_method == 'g2':
                                        p.fisher = p.grad * p.grad * FLAGS.batch_size

                                    p.inv_FIM = 1 / (p.fisher + 1e-7)
                                    p.inv_FIM = p.inv_FIM * 1e-7
            # else:
            #     if FLAGS.regularization=='fisher_training':


            # print(sum([np.sum(np.ones_like(p_[0].detach().cpu().numpy())) for p_ in ps]))

            gamma_ = FLAGS.gamma

            if distillation_fisher:
                if epoch<20:
                    gamma_=0.
                elif epoch<40:
                    gamma_=FLAGS.gamma * .5
                elif epoch<60:
                    gamma_=FLAGS.gamma * .75
                elif epoch<70:
                    gamma_ = FLAGS.gamma


            if FLAGS.optimizer=='sgdr':
                _, reg_val = optimizer.step(regularizer=FLAGS.regularization, return_reg_val=True, gamma = gamma_)

            elif FLAGS.regularization =='distillation':
                reg_val = kd_loss
            else:
                reg_val=torch.tensor(0.)

            train_acc = accuracy(output, targets).item()
            lossval = loss.item()

            # print(i)
            if i % record_interval==0 or i==0:

                fim_trace_g2 = 0
                fim_trace_adam = 0
                fmsqe_g2 = 0
                fmsqe_adam = 0
                orig_adam_fisher_msqe = 0

                corcoeffs_fisher = []
                corcoeffs_regularizer = []

                spearmans_fisher  = []
                spearmans_regularizer = []

                jj=0
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if hasattr(p, 'pert'):
                            pertvalue = p.pert.view(-1).cpu().numpy()

                            pert_sq = p.pert * p.pert

                            g2_fisher = p.grad * p.grad

                            fim_trace_g2 = fim_trace_g2 + torch.sum(g2_fisher)
                            fmsqe_g2 = fmsqe_g2 + torch.sum(g2_fisher * pert_sq) * FLAGS.batch_size

                            fim_trace_adam = fim_trace_adam + torch.sum(optimizer.state[p]['exp_avg_sq'])
                            fmsqe_adam = fmsqe_adam + torch.sum(optimizer.state[p]['exp_avg_sq'] * pert_sq) * FLAGS.batch_size

                            if FLAGS.regularization is not None:
                                orig_adam_fisher_msqe = orig_adam_fisher_msqe + torch.sum(p.orig_fisher * pert_sq)

                                corcoeffs_fisher.append(np.corrcoef(pertvalue, orig_fisher[jj])[1,0])
                                spearmans_fisher.append(stats.spearmanr(pertvalue, orig_fisher[jj])[0])

                                if 'fisher' in FLAGS.regularization=='inv_fisher':
                                    spearmans_regularizer.append(
                                        stats.spearmanr(pertvalue, p.inv_FIM.view(-1).cpu().numpy())[0])
                                    corcoeffs_regularizer.append(
                                        np.corrcoef(pertvalue, p.inv_FIM.view(-1).cpu().numpy())[0])
                                else:
                                    spearmans_regularizer.append(
                                        stats.spearmanr(pertvalue, p.fisher.view(-1).cpu().numpy())[0])
                                    corcoeffs_regularizer.append(
                                        np.corrcoef(pertvalue, p.fisher.view(-1).cpu().numpy())[0])

                            jj+=1



                # Step 1: compute metrics for saving
                msqe = sum([torch.sum(pert_ * pert_) for pert_ in perts])

                # msg = 'Step [%d] | Loss [%.4f] | Acc [%.3f]| MSQE [%.3f]| Trace [%.5f]' % (i, lossval, train_acc, msqe, fim_trace )
                if FLAGS.regularization is not None:
                    msg = 'Step [%d] | Loss [%.4f] | Acc [%.3f]| MSQE [%.3f]| FMSQE [%.4f]| REG [%.6f]| Trace [%.4f]| ORIG FMSQE [%.5f]| Orig Corr [%.2f]| Orig Spearman [%.2f]' % (
                        i, lossval, train_acc, msqe, fmsqe_adam, reg_val, fim_trace_adam, orig_adam_fisher_msqe, np.mean(corcoeffs_fisher), np.mean(spearmans_fisher))
                else:
                    msg = 'Step [%d] | Loss [%.4f] | Acc [%.3f]| MSQE [%.3f]| FMSQE [%.4f]| REG [%.6f]| Trace [%.4f]' % \
                          (i, lossval, train_acc, msqe, fmsqe_adam, reg_val, fim_trace_adam)
                print(msg)

                logstr += msg + '\n'

                if FLAGS.is_quantized:
                    fim_trace_save_g2.append(fim_trace_g2.item())
                    fim_trace_save_adam.append(fim_trace_adam.item())

                    fmsqe_save_adam.append(fmsqe_adam.item())
                    fmsqe_save_g2.append(fmsqe_g2.item())

                    if FLAGS.regularization is not None:
                        fmsqe_save_original_adam.append(orig_adam_fisher_msqe.item())
                        pearson_fisher.append(corcoeffs_fisher)
                        spearman_fisher.append(spearmans_fisher)

                        pearson_regularizer.append(corcoeffs_regularizer)
                        spearman_regularizer.append(spearmans_regularizer)

                    msqe_save.append(msqe.item())

                    regularizer_save.append(reg_val.item())
                    loss_save.append(lossval)

                #TODO: recode below for running_min instead of layer.weight.min_value

                # for l, (name, layer) in enumerate(model.named_modules()):
                #     # print(name)
                #     if FLAGS.q_min is None and FLAGS.q_max is None:
                #         if 'conv' in name or 'fc' in name:
                #             if hasattr(layer, 'weight'):
                #                 print('Min Value:\t' + str(layer.weight.min_value))
                #                 break
                #
                #             break
                #             print('Min/Max:\t' + str(layer.min_value) + '\t' + str(layer.max_value))


            i+=1

            # fim.append(FIM)

            # for fim_ in fim:
            #     inv_FIM = 1 / (fim_ + 1e-7)
            #     inv_FIM = inv_FIM * 1e-7
            #     inv_fim.append(inv_FIM)

            # fim = np.concatenate([np.ravel(fim_[0].cpu().numpy()) for fim_ in fim])
            # inv_fim = np.concatenate([np.ravel(inv_fim_[0].cpu().numpy()) for inv_fim_ in inv_fim])
            # corrected = fim+1e-6
            #
            # plt.figure()
            # plt.hist(inv_fim, log=False, bins=100)
            # plt.title('inv fisher')
            #
            # plt.figure()
            # plt.hist(fim/np.max(fim), log=False, bins=100)
            # plt.title('scaled fisher')
            #
            # plt.figure()
            # plt.hist(fim, log=False,bins=100)
            # plt.title('fisher')
            #
            # log_fim = np.log2(corrected)
            # plt.figure()
            # plt.hist(np.log(corrected), log=False,bins=100)
            # plt.title('log corrected fisher')
            #
            # plt.figure()
            # scaled = inv_fim * 1e-6
            # plt.hist(scaled, log=False, bins=100)
            # plt.title('scaled inv')
            #
            # plt.figure()
            # inv_log_fim = -log_fim
            # inv_log_fim = inv_log_fim - np.min(inv_log_fim )
            #
            # # inv_log_fim = (inv_log_fim-np.min(inv_log_fim))/(np.max(inv_log_fim))
            # plt.hist(inv_log_fim, log=False, bins=100)
            # plt.title('scaled log inv')
            #
            #
            # plt.figure()
            # plt.hist(corrected, log=False, bins=100)
            # plt.title('corrected fisher')
            #
            # plt.show()
            #
            # exit()

            # exit()
            # print(optimizer.state[p])
            # pdb.set_trace()
            # print(len(perts))
            # exit()


        end = time.time()
        elapsed = end - start
        model.eval()

        #Report test error every n epochs
        if epoch % 1 == 0 or epoch==n_epochs-1:
            msg = '\n*** TESTING ***\n'
            test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)

            if NUM_CLASSES>10:
                test_loss, test_acc_top5 = test_model(test_loader, model, criterion, printing=False, eta=etaval, topk=5)
                msg += 'End Epoch [%d]| Test Loss [%.3f]| Test Acc Top-1/Top-5 [%.3f | %.3f]| Ep Time [%.1f]  | LR [%.5f]\n' % (epoch, test_loss, test_acc, test_acc_top5, elapsed,  optimizer.param_groups[0]['lr'])
            else:
                msg += 'End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| Ep Time [%.1f]  | LR [%.5f]\n' % (epoch, test_loss, test_acc, elapsed,  optimizer.param_groups[0]['lr'])
            print(msg)
            logstr+=msg


        #save dict every epoch
        if epoch % 1 == 0:
            savedict = {'fim_trace_g2': fim_trace_save_g2, 'fim_trace_adam': fim_trace_save_adam,
                        'fmsqe_adam': fmsqe_save_adam, 'fmsqe_g2': fmsqe_save_g2,
                        'msqe': msqe_save, 'regularizer': regularizer_save,
                        'ce_loss': loss_save}

            if FLAGS.regularization is not None:
                savedict['fmqe_adam_orig'] = fmsqe_save_original_adam

                savedict['pearson_fisher'] = np.array(pearson_fisher).tolist()
                savedict['spearman_fisher'] = np.array(spearman_fisher).tolist()

                savedict['pearson_reg'] = np.array(pearson_regularizer).tolist()
                savedict['spearman_reg'] = np.array(spearman_regularizer).tolist()

            # print(savedict)
            with open(os.path.join(SAVEPATH_run, 'data_dict.txt'),'w+') as file:
                json.dump(savedict, file)
            file.close()


        #save model every 15 epochs
        model_path = os.path.join(SAVEPATH_run, 'resnet')

        # if epoch % 5 ==0:
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss}, model_path)
        #     print('SAVING TO: \t' + SAVEPATH_run)
        # print(model.conv1.quantize_input.running_min)
        # print(model.conv1.quantize_input.running_max)

        # print(model.conv1.min_value)
        # print(model.conv1.max_value)
        # print(np.ravel(model.conv1.qweight.detach().cpu().numpy()))
        # exit()
        msg = '\n*** EPOCH END ***\n\n\n'

        logstr += msg

        if FLAGS.logging:
            logfile.write(logstr)
            logfile.flush()

        if FLAGS.lr_decay:
            update_lr(epoch, optimizer, lr_scheduler, decay_method=FLAGS.lr_decay_type)

    print('\n*** TESTING ***\n')
    test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
    print('End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| Ep Time [%.1f]  | LR [%.3f]' % (
        epoch, test_loss, test_acc, elapsed, optimizer.param_groups[0]['lr']))
    test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
    test_accs.append(test_acc)

    msg = '************* FINAL ACCURACY *************\n'
    msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
    msg += '************* END *************\n'
    print(msg)

    if FLAGS.logging:
        logfile.write(msg)
        logfile.flush()

    # for l, (name, layer) in enumerate(model.named_modules()):
    #     # print(name)
    #     if FLAGS.q_min is None and FLAGS.q_max is None:
    #         if 'conv' in name or 'fc' in name:
    #             if hasattr(layer, 'weight'):
    #                 print('Min Value AFTER TEST:\t' + str(layer.min_value))
    #                 print('layer.weight.min():\t' + str(layer.weight.min()))
    #                 break
    model_path = os.path.join(SAVEPATH_run, 'resnet')
    config_file.write('\n FINAL Test Accuracy: %.3f' % test_acc)
    config_file.flush()
    config_file.close()

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, model_path)
    print('SAVING TO: \t' + SAVEPATH_run)


results_str = '\n******************************\n'
results_str += 'Accs: \t'+ str(test_accs) + '\n'
results_str += 'Avg accuracy: %.3f +\- %.4f' % (np.mean(test_accs), np.std(test_accs)) + '\n'
results_str += 'Num bits weight ' + str(n_bits_wt) + '\n'
results_str += 'Num bits activation ' + str(n_bits_act) + '\n'
print(results_str)



SAVEPATH_results = os.path.join(SAVEPATH, 'results.txt')
f = open(SAVEPATH_results, 'w+')
f.write(results_str)
f.flush()

dict_ind = list(flag_dict.keys()).index('dataset')
keys = list(flag_dict.keys())
values = list(flag_dict.values())
for dict_ind_ in range(dict_ind, len(flag_dict)):
    f.write(keys[dict_ind_] + ':\t' + str(values[dict_ind_])+'\n')
    f.flush()
f.close()
exit()

# sweep eta

etavals = np.array([.01, .1, .3, .5, .75, 1.25, 1.75, 2.5, 3.5, 10., 15, 25, 50])
n_etavals = len(etavals)
n_mc_iters = 20

test_accs = np.zeros([n_etavals, n_mc_iters])


for i, etaval_ in enumerate(etavals):

    for k in range(n_mc_iters):

        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval_)
        # print(test_acc)
        test_accs[i, k] = test_acc


import matplotlib.pyplot as plt


plt.title('Eta vs Acc for Baseline (not retrained)')
plt.grid(True)
plt.plot(etavals, np.mean(test_accs, axis=1), '-bo')
plt.xlabel('Eta')
plt.ylabel('Accuracy')
plt.show()








