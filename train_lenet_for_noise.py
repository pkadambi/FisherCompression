import tensorflow as tf


tf.app.flags.DEFINE_string( 'dataset', 'fashionmnist', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 25, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')

tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 8, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 8, 'number of bits weight')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, 'minimum quant value')
tf.app.flags.DEFINE_float('q_max', None, 'maximum quant value')
tf.app.flags.DEFINE_integer('n_runs', 5, 'number of times to train network')
tf.app.flags.DEFINE_float('eta', .0, 'noise eta')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use `l2,` `fisher` `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_float('gamma', 0.0001, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 5e-5, 'diagonal loading constant')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', './SavedModels/Lenet/8ba_8bw/', 'directory to load model from')

#TODO: find where this is actually used in the code
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use `sgd` or `adam`')

#Distillation Params
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/Lenet/FP/Run0/lenet', 'path to FP model for loading for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation regularizer multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'temperature for distillation')

import time as time
import numpy as np
import os
from data import get_dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualization_utils import *
from torch.autograd import Variable
import torch.optim as optim
from adamR import AdamR


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

    SAVEPATH = './SavedModels/Lenet/%dba_%dbw' % (n_bits_act, n_bits_wt)
    if reg_string is not '':
        SAVEPATH += '_' + reg_string
    SAVEPATH += '/'
    config_str += 'Savepath:\t' + SAVEPATH + '\n'

    if FLAGS.regularization is not None:
        SAVEPATH = SAVEPATH + FLAGS.regularization

elif not FLAGS.debug:
    SAVEPATH = FLAGS.savepath

if FLAGS.loadpath is not None:
    config_str += 'Loadpath:\t' + FLAGS.loadpath + '\n'



etaval = FLAGS.eta
#89.84 88.59
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


# n_runs = FLAGS.n_runs
n_runs = 1
test_accs=[]

etas = list(np.linspace(.03, .05, 3))
# etas = [.12, .15, .2, .25]

N_MC_ITERS = 15
N_noise_level = len(etas)

test_acc_matrix = np.zeros([N_MC_ITERS, N_noise_level])

kl_criterion = nn.KLDivLoss()
SAVEPATH_ = SAVEPATH+''
for zz, eta_ in enumerate(etas):

    SAVEPATH = SAVEPATH_ + '_eta_%.3f'% eta_

    for k in range(n_runs):
        i=0
        model = models.Lenet5(is_quantized=FLAGS.is_quantized)
        model.cuda()
        if FLAGS.optimizer=='adam':
            optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

        elif FLAGS.optimizer=='adamr':
            # optimizer = optim.SGDR(model.parameters(), momentum=.9, lr=FLAGS.lr, weight_decay= FLAGS.weight_decay)
            optimizer = AdamR(model.parameters(), lr=FLAGS.lr)
            # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=FLAGS.lr_end)
        # print(model.conv1.quantize_input)
        # exit()
        print('\n\n\n********** RUN %d **********\n' % k)

        if FLAGS.loadpath is not None:
            loadpath = os.path.join(FLAGS.loadpath, 'Run%d' % (k), 'lenet')
            checkpoint = torch.load(loadpath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            for name, layer in model.named_modules():
                if ('conv' in name or 'fc' in name) and not 'quantize' in name:
                    layer.set_min_max()
            test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=eta_)
            print(' Restored Model Accuracy (Noisy): \t %.3f' % (test_acc))
            config_str += 'Restored Model Test Acc:\t%.3f' % (test_acc) + '\n'

            test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=None)
            print(' Restored Model Accuracy(Clean): \t %.3f' % (test_acc))
            config_str += 'Restored Model Test Acc:\t%.3f' % (test_acc) + '\n'
            # exit()

        if distillation:
            checkpoint = torch.load(FLAGS.fp_loadpath)

            teacher_model = models.Lenet5(is_quantized=False)
            teacher_model.cuda()
            teacher_model.load_state_dict(checkpoint['model_state_dict'])


            for param in teacher_model.parameters():
                param.requires_grad = False


            test_loss, test_acc = test_model(test_loader, teacher_model, criterion, printing=False, eta=eta_)
            print('Restored TEACHER MODEL Accuracy: \t %.3f' % (test_acc))
            config_str += 'Restored TEACHER MODEL Test Acc:\t%.3f' % (test_acc) + '\n'

        for epoch in range(n_epochs):
            model.train()
            start = time.time()

            for iter, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()


                inputs = inputs.cuda()
                targets = targets.cuda()

                output = model(inputs, eta=eta_)
                loss = criterion(output, targets)

                #adding distillation loss
                if distillation:
                    teacher_model.eval()
                    teacher_output = Variable(teacher_model(inputs), requires_grad=False)
                    # teacher_output = teacher_output.long()
                    kd_loss = loss_fn_kd(output, teacher_output, FLAGS.temperature)
                    loss = loss + FLAGS.alpha * kd_loss

                # TODO: fix this, there shouldn't be a retain_graph=True here necessary, figure out why it's happening, if it's a problem, and then remove if needed
                loss.backward(retain_graph=True)
                perts=[]
                fim = []
                #fisher loss without messing up gradient flow
                for l, (name, layer) in enumerate(model.named_modules()):
                    with torch.no_grad():
                        if hasattr(layer, 'weight'):
                            # print(name)
                            # print(layer.qweight.size())

                            pertw = layer.qweight - layer.weight
                            # pertb = layer.bias - layer.qbias
                            layer.weight.pert = pertw
                            # layer.bias.pert = pertb


                if iter==0 and epoch==0 and FLAGS.regularization is not None:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if hasattr(p, 'pert'):
                                # if FLAGS.fisher_method == 'adam':
                                # print(p.size())
                                p.fisher = optimizer.state[p]['exp_avg_sq'] * FLAGS.batch_size

                            # layer.weight.grad += FLAGS.gamma * 2 * inv_FIM * pertw
                            # layer.bias.grad += FLAGS.gamma * 2 * inv_FIM_bias *  pertb
                # exit()
                for l, (name, layer) in enumerate(model.named_modules()):
                    with torch.no_grad():
                        if hasattr(layer, 'weight'):
                            if etaval>0 and FLAGS.gamma>0:
                                # print(layer.qweight.size())
                                # print(name)
                                pertw = layer.qweight - layer.qweight_n
                                # pertb = layer.qweight- layer.qbias_n
                                perts.append(pertw)
                                # FIM_diag = layer.weight.grad * layer.weight.grad
                                fim.append(layer.weight.fisher)
                                FIM_diag = layer.weight.fisher + FLAGS.diag_load_const

                                if fisher:
                                    layer.weight.grad += FLAGS.gamma * 2 * (layer.weight.fisher * pertw)
                                    # layer.bias.grad += FLAGS.gamma * 2 * (layer.bias.fisher * pertb +
                                    #                                         FLAGS.diag_load_const * pertb)
                                elif l2:
                                    layer.weight.grad += FLAGS.gamma * 2 * FLAGS.diag_load_const * pertw
                                    # layer.bias.grad += FLAGS.gamma * 2 * FLAGS.diag_load_const * pertb

                                elif inv_fisher:
                                    # FIM_diag = layer.weight.grad * layer.weight.grad + FLAGS.diag_load_const
                                    # FIM_diag_bias = layer.bias.grad * layer.bias.grad + FLAGS.diag_load_const




                                    # print('Max')
                                    # print(torch.topk(FIM_diag.view(-1), 100)[0])
                                    # print('Min')
                                    # print(torch.topk(FIM_diag.view(-1), 100, largest=False)[0])
                                    # inv_FIM = (1/(FIM_diag+1e-7) )* 1e-7
                                    # inv_FIM_bias = (1/(FIM_diag_bias + 1e-7)) * 1e-7

                                    inv_FIM = (1/(FIM_diag ) ) * FLAGS.diag_load_const
                                    # inv_FIM_bias = (1/(FIM_diag_bias + 1e-7))* 1e-7
                                    # print('Max')
                                    # print(torch.topk(inv_FIM.view(-1), 100)[0])
                                    # print('Min')
                                    # print(torch.topk(inv_FIM.view(-1), 100, largest=False)[0])


                                    layer.weight.grad += FLAGS.gamma * 2 * inv_FIM * pertw
                                    # layer.bias.grad += FLAGS.gamma * 2 * inv_FIM_bias *  pertb
                # exit()
                # print('IM HERE')

                optimizer.step()

                train_acc = accuracy(output, targets).item()
                lossval = loss.item()

                if i%record_interval==0 or i==0:
                    msqe = sum([torch.sum(pert_ * pert_) for pert_ in perts])
                    fim_trace = sum([torch.sum(fim_) for fim_ in fim])
                    print('Step [%d] | Loss [%.4f] | Acc [%.3f]| MSQE [%.3f]| Trace [%.3f]' % (i, lossval, train_acc, msqe, fim_trace))

                i+=1

            end = time.time()
            elapsed = end - start
            model.eval()
            print('\n*** TESTING ***\n')
            test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=eta_)
            test_loss_clean, test_acc_clean = test_model(test_loader, model, criterion, printing=False, eta=0)
            print('End Epoch [%d]| Test Loss(Noisy)[%.3f]| Test Acc(Clean) [%.3f]| Test Acc(Noisy) [%.3f]| Ep Time [%.1f]' % (epoch, test_loss, test_acc, test_acc_clean, elapsed))
            # print(model.conv1.quantize_input.running_min)
            # print(model.conv1.quantize_input.running_max)

            # print(model.conv1.min_value)
            # print(model.conv1.max_value)
            # print(np.ravel(model.conv1.qweight.detach().cpu().numpy()))
            # exit()
            print('\n*** EPOCH END ***\n')


        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
        test_accs.append(test_acc)

        print('************* FINAL ACCURACY *************')
        print('TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]' % (test_loss, test_acc))
        print('************* END *************')

        #TODO: mc iters test accuracy

        print('\n\n\nBEGIN MC ITERS------------------------------')
        for kk in range(N_MC_ITERS):
            # test_loss, test_acc, kl_loss_ = test_model(test_loader, model, criterion, printing=False, eta=eta, teacher_model=teacher_model)
            test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=eta_)

            test_acc_matrix[kk, zz] = test_acc

        print('\n\nEta Level: %.3f\t' % eta_)
        print('Avg Test Acc (Noisy): %.3f\t' % np.mean(test_acc_matrix[:, zz], axis=0))


        j=0
        while os.path.exists(os.path.join(SAVEPATH, 'Run%d' % j)):
            j+=1

        SAVEPATH_run = os.path.join(SAVEPATH, 'Run%d' % j)

        os.makedirs(SAVEPATH_run, exist_ok=True)
        config_path = os.path.join(SAVEPATH_run, 'config_str.txt')
        model_path = os.path.join(SAVEPATH_run, 'lenet')

        f = open(config_path, 'w+')


        f.write(config_str)
        f.flush()
        f.write('Test Accuracy: %.3f' % test_acc)
        f.flush()
        f.close()

        print('Saving to: ' + model_path)
        torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, model_path)
        # exit()

print('Etas')
print(etas)
print('Accuracies')
print(np.mean(test_acc_matrix, axis=0))

accs_avg = np.mean(test_acc_matrix, axis=0)
np.savetxt('./NoisyModel/Lenet/'+FLAGS.regularization + '/noisy_accs_lenet.txt', test_acc_matrix)
np.savetxt('./NoisyModel/Lenet/'+FLAGS.regularization + '/noisy_accs_avg_lenet.txt', accs_avg)
np.savetxt('./NoisyModel/Lenet/'+FLAGS.regularization + '/etas.txt', etas)



plt.figure()
plt.plot(etas, np.mean(test_acc_matrix, axis=0))
plt.title('Eta vs Acc')
plt.grid(True)
exit()

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
    # exit()

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









