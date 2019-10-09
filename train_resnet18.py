from data import get_dataset
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
'''

n epochs = 300
cosine decay rate
weight decay

'''

tf.app.flags.DEFINE_string( 'dataset', 'cifar10', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 200, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')
tf.app.flags.DEFINE_float('weight_decay', 2e-4, 'weight decay value')
# tf.app.flags.DEFINE_float('weight_decay', 100, 'weight decay value')

tf.app.flags.DEFINE_float('lr', .1, 'learning rate')

# tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
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

tf.app.flags.DEFINE_float('gamma', 0.005, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 0.005, 'diagonal loading constant')

# tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_string('optimizer', 'sgdr', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_boolean('lr_decay', True, 'Whether or not to decay learning rate')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')


#TODO: find where this is actually used in the code
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')

#Distillation Params
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/Resnet18/FP/Run0/resnet', 'path to FP model for loading for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation regularizer multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'temperature for distillation')

tf.app.flags.DEFINE_boolean('logging', False,'whether to enable writing to a logfile')

'''

NOTE: the following imports must be after flags declarations since these files query the flags 

'''
from sgdR import SGDR

from models.resnet_quantized import ResNet_cifar10
from models.resnet_binary import ResNet_cifar10_binary
FLAGS = tf.app.flags.FLAGS


n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act

'''

Redefine any flags here if you want to run a sweep

'''


'''

Config file save information

'''

#Config string
config_str = ''
config_str += 'Dataset:\t' + FLAGS.dataset + '\n'
config_str += 'Batch Size:\t' + str(FLAGS.batch_size) + '\n'
config_str += 'N Epochs:\t' + str(FLAGS.n_epochs) + '\n'
config_str += 'Learning Rate:\t' + str(FLAGS.lr) + '\n'


config_str += 'Is Quantized:\t' + str(FLAGS.is_quantized) + '\n'



fisher = FLAGS.regularization=='fisher'
l2 = FLAGS.regularization=='l2'
distillation = FLAGS.regularization=='distillation'
inv_fisher = FLAGS.regularization=='inv_fisher'

if FLAGS.is_quantized:
    config_str += 'Q Min:\t' + str(FLAGS.q_min) + '\n'
    config_str += 'Q Max:\t' + str(FLAGS.q_max) + '\n'
    config_str += 'N Bits Act:\t' + str(FLAGS.n_bits_act) + '\n'
    config_str += 'N Bits Wt:\t' + str(FLAGS.n_bits_wt) + '\n'


config_str += 'Regularizer: \t' + str(FLAGS.regularization) + '\n'

if FLAGS.regularization == 'fisher' or FLAGS.regularization=='l2' or FLAGS.regularization=='inf_fisher':
    config_str += 'Diag Load Const:\t' + str(FLAGS.diag_load_const) + '\n'
    config_str += 'Gamma:\t' + str(FLAGS.gamma) + '\n'
    reg_string = FLAGS.regularization

elif FLAGS.regularization=='distillation':
    distillation = True
    config_str+= 'Temperature:\t' + str(FLAGS.temperature) + '\n'
    config_str+= 'Alpha:\t' + str(FLAGS.alpha) + '\n'
    reg_string = FLAGS.regularization

else:
    reg_string = ''
    distillation=False
if FLAGS.noise_model is not None:
    config_str += '' + FLAGS.noise_model + '\n'


#Save path
if not FLAGS.debug and FLAGS.savepath is None:
    SAVEPATH = './SavedModels/Resnet18/%dba_%dbw' % (n_bits_act, n_bits_wt)
    if reg_string is not '':
        SAVEPATH += '_' + reg_string
    SAVEPATH += '/'
    config_str += 'Savepath:\t' + SAVEPATH + '\n'
    if FLAGS.regularization is not None:
        SAVEPATH += '/' + FLAGS.regularization

elif not FLAGS.debug:
    SAVEPATH = FLAGS.savepath

if FLAGS.loadpath is not None:
    config_str += 'Loadpath:\t' + FLAGS.loadpath + '\n'

if FLAGS.logging:
    logpath =SAVEPATH + '/logfile.txt'

    logfile = open(logpath, 'w+')

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
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=2e-4)
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
        logstr += '\nRestored Model Accuracy: \t %.3f' % (test_acc)
        config_str += 'Restored Model Test Acc:\t%.3f' % (test_acc) + '\n'
        # exit()


    if distillation:
        checkpoint = torch.load(FLAGS.fp_loadpath)

        teacher_model = ResNet_cifar10(is_quantized=False)
        teacher_model.cuda()
        teacher_model.load_state_dict(checkpoint['model_state_dict'])


        for param in teacher_model.parameters():
            param.requires_grad = False


        test_loss, test_acc = test_model(test_loader, teacher_model, criterion, printing=False, eta=etaval)
        logstr += '\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc)
        config_str += logstr

    if FLAGS.logging:
        print(logstr)
        logfile.write(logstr)
        logfile.flush()

    for epoch in range(n_epochs):


        logfile='\n\n****** NEXT EPOCH ******\n'
        model.train()
        start = time.time()
        for iter, (inputs, targets) in enumerate(train_loader):


            inputs = inputs.cuda()
            targets = targets.cuda()

            output = model(inputs)
            loss = criterion(output, targets)

            #adding distillation loss
            if distillation:
                teacher_model.eval()
                teacher_output = Variable(teacher_model(inputs, eta=etaval), requires_grad=False)
                # teacher_output = teacher_output.long()
                loss = loss + FLAGS.alpha * loss_fn_kd(output, teacher_output, FLAGS.temperature)


            optimizer.zero_grad()
            loss.backward()

            #fisher loss without messing up gradient flow
            perts = []
            fim=[]
            inv_fim=[]

            if FLAGS.is_quantized:
                for l, (name, layer) in enumerate(model.named_modules()):
                    if 'conv' in name or 'fc' in name:
                        # print('\n\nNEW LAYER')
                        # print(layer)
                        with torch.no_grad():
                            if hasattr(layer, 'qweight'):
                                # print(name)

                                pertw = layer.weight - layer.qweight
                                layer.weight.pert = pertw
                                FIM = layer.weight.grad * layer.weight.grad

                                if FLAGS.optimizer is not 'sgdr':
                                    fim.append(FIM)

                                perts.append(pertw)

                                if fisher:
                                    layer.weight.grad += FLAGS.gamma * 2 * (layer.weight.grad * layer.weight.grad * pertw +
                                                                            FLAGS.diag_load_const * pertw)
                                    # layer.weight.grad += FLAGS.gamma * 2 * (1/(layer.weight.grad * layer.weight.grad) * pertw )
                                elif inv_fisher:
                                    inv_FIM = 1/(FIM+1e-7)
                                    inv_FIM = inv_FIM * 1e-7
                                    inv_fim.append(inv_FIM)

                                    layer.weight.grad += torch.clamp(FLAGS.gamma * 2 * (inv_FIM * pertw),-.01,.01)
                                    # layer.weight.reg_grad = FLAGS.gamma * 2 * (inv_FIM *(pertw + FLAGS.diag_load_const * pertw))

                                elif l2:
                                    layer.weight.grad += FLAGS.gamma * 2 * FLAGS.diag_load_const * pertw


            if FLAGS.optimizer=='sgdr':
                optimizer.step(regularizer=FLAGS.regularization)

                # code to access fisher information from the optimizer
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if hasattr(p, 'pert'):
                            fim.append(optimizer.state[p]['exp_avg_sq'])

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
                # plt.hist(inv_fim, log=True, bins=50)
                # plt.title('inv fisher')
                #
                # plt.figure()
                # plt.hist(fim, log=True,bins=50)
                # plt.title('fisher')
                #
                # log_fim = np.log2(corrected)
                # plt.figure()
                # plt.hist(np.log(corrected), log=True,bins=50)
                # plt.title('log corrected fisher')
                #
                # plt.figure()
                # scaled = inv_fim * 1e-6
                # plt.hist(scaled, log=True, bins=50)
                # plt.title('scaled inv')
                #
                # plt.figure()
                # inv_log_fim = -log_fim
                # inv_log_fim = inv_log_fim - np.min(inv_log_fim )
                #
                # # inv_log_fim = (inv_log_fim-np.min(inv_log_fim))/(np.max(inv_log_fim))
                # plt.hist(inv_log_fim, log=True, bins=50)
                # plt.title('scaled log inv')
                #
                #
                # plt.figure()
                # plt.hist(corrected, log=True, bins=50)
                # plt.title('corrected fisher')
                #
                # plt.show()

                # exit()

                # exit()
                # print(optimizer.state[p])
                # pdb.set_trace()
                # print(len(perts))
                # exit()
            else:
                optimizer.step()


            train_acc = accuracy(output, targets).item()
            lossval = loss.item()

            # print(i)
            if i%record_interval==0 or i==0:
                msqe = sum([torch.sum(pert_ * pert_) for pert_ in perts])
                fim_trace = sum([torch.sum(fim_) for fim_ in fim])
                msg = 'Step [%d] | Loss [%.4f] | Acc [%.3f]| MSQE [%.3f]| Trace [%.5f]' % (i, lossval, train_acc, msqe, fim_trace )
                print(msg)
                logstr += msg

            i+=1

        end = time.time()
        elapsed = end - start
        model.eval()

        #Report test error every n epochs
        if epoch % 1 == 0:
            msg = '\n*** TESTING ***\n'
            test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
            msg += 'End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| Ep Time [%.1f]  | LR [%.5f]' % (epoch, test_loss, test_acc, elapsed,  optimizer.param_groups[0]['lr'])
            print(msg)
            logstr+=msg



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
            lr_scheduler.step(epoch)

    print('\n*** TESTING ***\n')
    test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
    print('End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| Ep Time [%.1f]  | LR [%.3f]' % (
    epoch, test_loss, test_acc, elapsed, optimizer.param_groups[0]['lr']))
    test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
    test_accs.append(test_acc)

    msg = '************* FINAL ACCURACY *************'
    msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]' % (test_loss, test_acc)
    msg += '************* END *************'
    print(msg)

    if FLAGS.logging:
        logfile.write(msg)
        logfile.flush()

    j=0
    while os.path.exists(os.path.join(SAVEPATH, 'Run%d' % j)):
        j+=1

    SAVEPATH_run = os.path.join(SAVEPATH, 'Run%d' % j)

    os.makedirs(SAVEPATH_run, exist_ok=True)
    config_path = os.path.join(SAVEPATH_run, 'config_str.txt')
    model_path = os.path.join(SAVEPATH_run, 'resnet')

    f = open(config_path, 'w+')


    f.write(config_str)
    f.flush()
    f.write('Test Accuracy: %.3f' % test_acc)
    f.flush()
    f.close()

    torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss}, model_path)

results_str = config_str + '\n******************************\n'
results_str += 'Accs: \t'+ str(test_accs) + '\n'
results_str += 'Avg accuracy: %.3f +\- %.4f' % (np.mean(test_accs), np.std(test_accs)) + '\n'
results_str += 'Num bits weight ' + str(n_bits_wt) + '\n'
results_str += 'Num bits activation ' + str(n_bits_act) + '\n'
print(results_str)

SAVEPATH_results = os.path.join(SAVEPATH, 'results.txt')
f = open(SAVEPATH_results, 'w+')
f.write(results_str )
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








