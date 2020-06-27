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
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from nonparametrics import *

'''

n epochs = 300
cosine decay rate
weight decay

'''

# MUST KEEP THIS AS THE FIRST FLAG
tf.app.flags.DEFINE_string('dataset', 'fashionmnist', 'either mnist or fashionmnist')
tf.app.flags.DEFINE_string('noise_scale', None, '`inv_fisher` or `fisher`')

tf.app.flags.DEFINE_string('activation', 'relu', '`tanh` or `relu`')
tf.app.flags.DEFINE_string('lr_decay_type', 'cosine', '`step` or `cosine`, defaults to cosine')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('n_epochs', 30, 'num epochs')
tf.app.flags.DEFINE_integer('record_interval', 100, 'how many iterations between printing to console')
tf.app.flags.DEFINE_float('weight_decay', 0., 'weight decay value')

tf.app.flags.DEFINE_integer('inflate', None, 'inflating factor for resnet (may need bigger factor if 1-b weights')
tf.app.flags.DEFINE_float('lr', .001, 'learning rate')

tf.app.flags.DEFINE_boolean('is_quantized', True, 'whether the network is quantized')
tf.app.flags.DEFINE_integer('n_bits_act', 4, 'number of bits activation')
tf.app.flags.DEFINE_integer('n_bits_wt', 4, 'number of bits weight')
tf.app.flags.DEFINE_float('eta', .0, 'noise eta')

tf.app.flags.DEFINE_string('noise_model', None, 'type of noise to add None, NVM, or PCM')

tf.app.flags.DEFINE_float('q_min', None, 'minimum quant value')
tf.app.flags.DEFINE_float('q_max', None, 'maximum quant value')
tf.app.flags.DEFINE_integer('n_runs', 5, 'number of times to train network')
tf.app.flags.DEFINE_integer('n_classes', 10, 'n classes')

tf.app.flags.DEFINE_boolean('enforce_zero', False, 'whether or not enfore that one of the quantizer levels is a zero')

tf.app.flags.DEFINE_string('regularization', None,
                           'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')
# tf.app.flags.DEFINE_string('regularization', 'distillation', 'type of regularization to use `l2,` `fisher` or `distillation` or `inv_fisher`')

tf.app.flags.DEFINE_float('gamma', 0.0001, 'gamma value')
tf.app.flags.DEFINE_float('diag_load_const', 0.005, 'diagonal loading constant')

# tf.app.flags.DEFINE_string('optimizer', 'sgd', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'optimizer to use `sgd` or `adam`')
tf.app.flags.DEFINE_boolean('lr_decay', True, 'Whether or not to decay learning rate')

tf.app.flags.DEFINE_string('savepath', None, 'directory to save model to')
tf.app.flags.DEFINE_string('loadpath', None, 'directory to load model from')

# TODO: find where this is actually used in the code
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')

# Distillation Params
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/cifar10/Resnet18/fp_buffer/Run3/resnet',
                           'path to FP model for loading for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation regularizer multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'temperature for distillation')

tf.app.flags.DEFINE_boolean('logging', True, 'whether to enable writing to a logfile')
tf.app.flags.DEFINE_float('lr_end', 2e-4, 'learning rate at end of cosine decay')
tf.app.flags.DEFINE_boolean('constant_fisher', True,
                            'whether to keep fisher/inv_fisher constant from when the checkpoint')

tf.app.flags.DEFINE_string('fisher_method', 'adam', 'which method to use when computing fisher')
tf.app.flags.DEFINE_boolean('layerwise_fisher', True, 'whether or not to use layerwise fisher')

tf.app.flags.DEFINE_boolean('eval', False,
                            'if this flag is enabled, the code doesnt write anyhting, it just loads from `FLAGS.loadpath` and evaluates test acc once')
tf.app.flags.DEFINE_boolean('loss_surf_eval_d_qtheta', default=False,
                            help='whether we are in loss surface generation mode')

tf.app.flags.DEFINE_string('regularizer', '', help='can have `SLS` OR `ULS`, `Fisher` OR `MSQE`, `distillation`')
tf.app.flags.DEFINE_boolean('learnminmax', True, 'whether to learn minmax')


'''

NOTE: the following imports must be after flags declarations since these files query the flags 

'''

from sgdR import SGDR
from adamR import AdamR
from models.resnet_quantized import ResNet_cifar10
from models.resnet_lowp import ResNet_cifar10_lowp
from models.resnet_binary import ResNet_cifar10_binary
import models
from cot_loss import *

FLAGS = tf.app.flags.FLAGS
flag_dict = FLAGS.flag_values_dict()
n_bits_wt = FLAGS.n_bits_wt
n_bits_act = FLAGS.n_bits_act

batch_size = FLAGS.batch_size
n_workers = 4
dataset = FLAGS.dataset
n_epochs = FLAGS.n_epochs
record_interval = FLAGS.record_interval
regularizer = FLAGS.regularizer

# Torch gpu stuff
cudnn.benchmark = True
torch.cuda.set_device(0)


def regularizer_multiplier(curr_epoch, n_epochs):
    midpt = int(n_epochs / 2)

    return 1 / (1 + np.exp(-0.5 * (curr_epoch - midpt)))


# TODO: add dataset label smoothing here


# 1. Weizhi's method

# 2. Distillation ( we already have that)

# 3. Not label smoothing but add complement entropy loss

# 4. Naive label smoothing - via the method in the medium article


DISTIL = False
SLS = False
MSQE = False
FISHER = False
COT = False

if 'distillation' in regularizer.lower():
    DISTIL = True

if 'SLS' in regularizer.lower():
    SLS = True

if 'MSQE' in regularizer.lower():
    MSQE = True

if 'fisher' in regularizer.lower():
    FISHER = True

if 'complement' in regularizer.lower() or 'cot' in regularizer.lower():
    COT = True


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        log_likelihood = - F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num

        return loss


criterion = nn.CrossEntropyLoss()
criterion_full_labels = SoftCrossEntropy()
cot_loss = ComplementEntropy()

# torch.scatter()

train_data = get_dataset(name=dataset, split='train', transform=get_transform(name=dataset, augment=True))
test_data = get_dataset(name=dataset, split='test', transform=get_transform(name=dataset, augment=False))



def compute_entropy_for_clusters(data_clusters, teacher_model, temperature):

    #loop through clusters

    teacher_model.eval()
    teacher_model.cuda()

    avg_entropy_per_cluster = np.zeros(len(data_clusters))

    per_sample_entropy_per_cluster = [np.zeros(len(d)) for d in data_clusters]
    for i, cluster in enumerate(data_clusters):
        input = torch.Tensor(cluster).cuda()
        input = input.view(-1, 1, 28, 28) #reshape for correct input to network
        teacher_logits = teacher_model(input)
        teacher_soft_logits = F.softmax(teacher_logits/temperature, dim=1)
        teacher_soft_logits = teacher_soft_logits.detach().cpu().numpy()


        sample_entropies = stats.entropy(teacher_soft_logits.T)
        avg_entropy_per_cluster[i] = np.mean(sample_entropies, axis=0)
        per_sample_entropy_per_cluster[i] = sample_entropies

    return avg_entropy_per_cluster, per_sample_entropy_per_cluster

ncomponents_gmm = 64
ncomponents_pca = 256
filename = './fashionmnist_clusters_gmm%d_pca%d.txt' % (ncomponents_gmm, ncomponents_pca)

'''

Step 1: get the cluster memberships

'''
npdata = np.vstack([tr[0].numpy().reshape(1, -1) for tr in train_data])
labels = np.array([tr[1] for tr in train_data])

if not os.path.exists(filename):

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    print('Started pca')
    pca = PCA(n_components=256).fit(npdata)
    data_reduc = pca.transform(npdata)

    print('Clustering with GMM')
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=64)
    gmm.fit(data_reduc)
    cluster_labels = gmm.predict(data_reduc)

    np.savetxt(filename, cluster_labels, fmt='%d')
else:
    cluster_labels = np.loadtxt(filename)

data_clustered, labels_clustered = split_data_into_clusters(npdata, labels, cluster_memberships=cluster_labels)

'''

Step 2: compute entropy of each cluster (using the teacher model) 
Doing this in lieu of SLS since sls giving weird results

'''

#restore teacher model
teacher_model = models.Lenet5(is_quantized=False)
teacher_model.cuda()
checkpoint = torch.load('./fashion_model_fp')
teacher_model.load_state_dict(checkpoint['model_state_dict'])

TEMP = 4
avg_entropies, entropy_per_cluster = compute_entropy_for_clusters(data_clustered, teacher_model, TEMP)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                          num_workers=n_workers, pin_memory=True)

median_entropies = [np.median(entropy_per_cluster[i]) for i in range(ncomponents_gmm)]

mean_log_entropy = [np.mean(np.log(entropy_per_cluster[i])) for i in range(ncomponents_gmm)]
median_log_entropy = [np.median(np.log(entropy_per_cluster[i])) for i in range(ncomponents_gmm)]

#Compute BER
print('Computing Delta_ij')
dijs = [compute_delta_ijs(data__, labels__, n_classes=10) for data__, labels__ in zip(data_clustered, labels_clustered)]

print('Computing BER')
bers = [ber_from_delta_ij(dij, n_classes=10) for dij in dijs]
examples_per_cluster = [len(d) for d in data_clustered]


plt.figure()
plt.scatter(np.arange(ncomponents_gmm), [np.median(entropy_per_cluster[i]) for i in range(ncomponents_gmm)])

plt.figure()
plt.hist(median_entropies, bins=20)
plt.title('Histogram of Median Entropy of Cluster')

# plt.figure()
# for i in range(ncomponents_gmm):
#     plt.hist(np.log(entropy_per_cluster[i]), bins=20)
# plt.xlabel('Log Entropy')
# plt.ylabel('Count')
# plt.title('Histogram of Entropy Per Cluster')

plt.figure()
plt.scatter(bers, median_log_entropy, label='Median Log Entropy')
plt.legend()
plt.xlabel('BER')
plt.ylabel('Log Entropy')
plt.title('Cluster Ber vs Log Entropy of Teacher Output for Cluster')

plt.figure()
plt.scatter(bers, mean_log_entropy, label='Mean Log Entropy')
plt.legend()
plt.xlabel('BER')
plt.ylabel('Log Entropy')
plt.title('Ber vs Log Entropy of Teacher')

inds = np.argsort(bers)
plt.figure()
plt.hist(entropy_per_cluster[inds[0]])
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.title('Entropy Histogram for Low BER Cluster')

plt.figure()
plt.hist(entropy_per_cluster[inds[-1]])
plt.xlabel('Entropy')
plt.ylabel('Count')
plt.title('Entropy Histogram for High BER Cluster')

plt.figure()
alphahats = calculate_alpha_hat(alpha=.1, beta=.1, n_classes=10,
                                clusterwise_ber=bers,
                                examples_per_cluster=examples_per_cluster)
plt.scatter(bers, alphahats)
plt.title('BER vs Alpha Hat')
plt.show()
exit()

npdata = np.vstack([tr[0].numpy().reshape(1, -1) for tr in train_data])
labels = np.array([tr[1] for tr in train_data])

clusters = np.loadtxt('fashionmnist_clusters.txt')
data = train_data.data.float().cuda().view(-1, 1, 28, 28)
labels = train_data.targets.cuda()
teacher_model.eval()
i = 0

n_test = 0.
n_correct = 0.
n_batch = 1000
teacher_model.eval()

while i < 60000:
    inputs_ = data[i:i + n_batch, :, :, :]
    target = labels[i:i + n_batch].cuda()
    output = teacher_model(inputs_)
    n_correct += accuracy(output, target, topk=(1,)).item() * n_batch / 100.
    n_test += n_batch
    i += n_batch

print(n_correct / n_test)

test_loss, tr_acc = test_model(train_loader, teacher_model, criterion, printing=False, eta=0.)
print(tr_acc)
exit()
test_loss, test_acc = test_model(test_loader, teacher_model, criterion, printing=False, eta=0.)

msg = '************* FINAL ACCURACY *************\n'
msg += 'TRAINING END | Test Loss [%.3f]| Test Acc [%.3f]\n' % (test_loss, test_acc)
msg += '************* END *************\n'
print(msg)
pdb.set_trace()
print()
