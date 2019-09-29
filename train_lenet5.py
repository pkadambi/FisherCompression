from data import get_dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualization_utils import *
import torch.optim as optim
import tensorflow as tf
import time as time
import numpy as np
import os

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

tf.app.flags.DEFINE_string('regularization', None, 'type of regularization to use')
tf.app.flags.DEFINE_float('gamma', 0.01, 'gamma value')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')



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


config_str += 'Is Quantized:\t' + str(FLAGS.is_quantized) + '\n'

if FLAGS.is_quantized:
    config_str += 'Q Min:\t' + str(FLAGS.q_min) + '\n'
    config_str += 'Q Max:\t' + str(FLAGS.q_max) + '\n'
    config_str += 'N Bits Act:\t' + str(FLAGS.n_bits_act) + '\n'
    config_str += 'N Bits Wt:\t' + str(FLAGS.n_bits_wt) + '\n'


config_str += 'Regularizer: \t' + str(FLAGS.regularization) + '\n'
if FLAGS.regularization is not None:
    config_str += '' + str(FLAGS.gamma) + '\n'

if FLAGS.noise_model is not None:
    config_str += '' + FLAGS.noise_model + '\n'


#Save path
if not FLAGS.debug and FLAGS.savepath is None:
    SAVEPATH = './SavedModels/Lenet/%dba_%dbw/' % (n_bits_act, n_bits_wt)
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
    model = models.Lenet5()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print('\n\n\n********** RUN %d **********\n' % k)

    if FLAGS.loadpath is not None:
        loadpath = os.path.join(FLAGS.loadpath, 'Run%d' % (k))
        checkpoint = torch.load(loadpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
        print(' RESTORED MODEL TEST ACCURACY: \t %.3f' % (test_acc))

    for epoch in range(n_epochs):
        model.train()
        start = time.time()

        for iter, (inputs, targets) in enumerate(train_loader):


            inputs = inputs.cuda()
            targets = targets.cuda()

            output = model(inputs, eta=etaval)

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc = accuracy(output, targets).item()
            lossval = loss.item()

            if i%record_interval==0 or i==0:
                print('Step [%d] | Loss [%.3f] | Acc [%.3f]' % (i, lossval, train_acc))

            i+=1

        end = time.time()
        elapsed = end - start
        model.eval()
        print('\n*** TESTING ***\n')
        test_loss, test_acc = test_model(test_loader, model, criterion, printing=False, eta=etaval)
        print('End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]| Ep Time [%.1f]' % (epoch, test_loss, test_acc, elapsed))
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


    j=0
    while os.path.exists(os.path.join(SAVEPATH, 'Run%d' % j)):
        j+=1

    SAVEPATH_run = os.path.join(SAVEPATH, 'Run%d' % j)

    os.makedirs(SAVEPATH_run, exist_ok=True)
    config_path = os.path.join(SAVEPATH_run, 'config_str.txt')
    model_path = os.path.join(SAVEPATH_run, 'lenet')

    f = open(config_path, 'w+')

    config_str += 'Test Accuracy: %.3f' % test_acc

    f.write(config_str)
    f.flush()
    f.close()

    torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss}, model_path)

results_str = config_str + '\n******************************\n'
print(test_accs)
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









