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

tf.app.flags.DEFINE_string( 'dataset', 'fashionmnist', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_integer( 'batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 25, 'num epochs' )
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')

tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 8, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 8, 'number of bits weight')
tf.app.flags.DEFINE_float('eta', .000001, 'noise eta')

tf.app.flags.DEFINE_string('noise_model', 'NVM', 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, '')
tf.app.flags.DEFINE_float('q_max', None, '')


FLAGS = tf.app.flags.FLAGS

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

i=0
model = models.Lenet5()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):
    model.train()
    start = time.time()

    for iter, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model(inputs, eta=etaval)

        loss = criterion(output, targets)
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

    print('\n*** EPOCH END ***\n')

