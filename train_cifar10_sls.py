import pdb
import torch
import numpy as np
from data import *
from preprocess import *
from nonparametrics import *
import tensorflow as tf
import torch.nn as nn
from utils.model_utils import *
import scipy.stats as stats
import matplotlib.pyplot as plt
import tqdm
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


#TODO: find where this is actually used in the code
tf.app.flags.DEFINE_boolean('debug', False, 'if debug mode or not, in debug mode, model is not saved')

#Distillation Params
tf.app.flags.DEFINE_string('fp_loadpath', './SavedModels/cifar10/Resnet18/fp_updated/Run0/resnet', 'path to FP model for loading for distillation')
tf.app.flags.DEFINE_float('alpha', 1.0, 'distillation regularizer multiplier')
tf.app.flags.DEFINE_float('temperature', 1.0, 'temperature for distillation')

tf.app.flags.DEFINE_boolean('logging', True,'whether to enable writing to a logfile')
tf.app.flags.DEFINE_float('lr_end', 2e-4, 'learning rate at end of cosine decay')
tf.app.flags.DEFINE_boolean('constant_fisher', True,'whether to keep fisher/inv_fisher constant from when the checkpoint')

tf.app.flags.DEFINE_string('fisher_method', 'adam','which method to use when computing fisher')
tf.app.flags.DEFINE_boolean('layerwise_fisher', True,'whether or not to use layerwise fisher')

tf.app.flags.DEFINE_boolean('eval', False,'if this flag is enabled, the code doesnt write anyhting, it just loads from `FLAGS.loadpath` and evaluates test acc once')
tf.app.flags.DEFINE_boolean('loss_surf_eval_d_qtheta', default=False, help='whether we are in loss surface generation mode')

dataset='cifar10'

train_data = get_dataset(name = dataset, split = 'train', transform=get_transform(name=dataset, augment=True))
test_data = get_dataset(name = dataset, split = 'test', transform=get_transform(name=dataset, augment=False))

npdata = np.vstack([tr[0].numpy().reshape(1,-1) for tr in train_data])
labels = np.array([tr[1] for tr in train_data])

cluster_memberships = np.load('./SLS_param/CIFAR10TrainClusterLoc.npy')
alphahats1 = np.load('./SLS_param/cifar10_myregularizer_disturbing288.npy')
alphahats2 = np.load('./SLS_param/cifar10_myregularizer_disturbing295.npy')
alphahats3 = np.load('./SLS_param/cifar10_myregularizer_disturbing299.npy')


from models.resnet_quantized import ResNet_cifar10


#Load the teacher model
FLAGS = tf.app.flags.FLAGS

checkpoint = torch.load(FLAGS.fp_loadpath)
print('Restoring teacher model from:\t' + FLAGS.fp_loadpath)

teacher_model = ResNet_cifar10(is_quantized=False, num_classes=10)
teacher_model.cuda()
teacher_model.load_state_dict(checkpoint['model_state_dict'])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=True,
                                          num_workers=4, pin_memory=True)

criterion = nn.CrossEntropyLoss()
test_loss, test_acc = test_model(test_loader, teacher_model, criterion, printing=False, eta=0.)
print('\nRestored TEACHER MODEL Test Acc:\t%.3f' % (test_acc))



clustered_data, clustered_labels = split_data_into_clusters(npdata, labels, cluster_memberships)

avg_entropy, sample_entropies = compute_entropy_for_clusters(clustered_data, teacher_model, temperature=10.)

nclusters = 64
plt.figure()
plt.hist([np.median(sample_entropies[i]) for i in range(nclusters)])
# plt.title('Median Soft Label Entropy Based on Weizhis clusters')
plt.title('Median Soft Label Entropy Based on Weizhis clusters')

plt.figure()
for ii in range(len(sample_entropies)):
    plt.hist(np.log(sample_entropies[ii]), bins=15)
plt.title('Soft Label Entropy Based on Weizhis clusters')

def compute_error_for_each_cluster(model, cluster_data, cluster_labels):

    cluster_accuracies = []
    for ii, (xcluster, ycluster) in enumerate(zip(cluster_data, cluster_labels)):
        with torch.no_grad():
            input = torch.Tensor(xcluster).cuda()
            input = input.view(-1, 3, 32, 32)  # reshape for correct input to network
            targets = torch.Tensor([ycluster]).cuda()
            teacher_logits = model(input)
            teacher_soft_logits = F.softmax(teacher_logits, dim=1)
            predictions = teacher_soft_logits.detach().cpu().numpy()
            # preds = predictions.argmax(axis=1)
            print(teacher_logits.shape)
            print(targets.shape)
            print(ii)
            _accuracy = 100 * sum(predictions.argmax(axis=1)==ycluster)/len(ycluster)
            cluster_accuracies.append(_accuracy)

    return cluster_accuracies




'''

Do my clustering here

'''

# dij=compute_delta_ijs(npdata, labels, n_classes=10)

import pickle as pkl
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
if not os.path.exists('./cifar10_pca'):
    print('Fitting PCA...')
    pca.fit(npdata)
    pkl.dump(pca, open('./cifar10_pca', 'wb'))
else:
    pca = pkl.load(open('./cifar10_pca', 'rb'))
data_reduc = pca.transform(npdata)

gmm = GaussianMixture(n_components=nclusters)
if not os.path.exists('./cifar10_gmm64'):
    print('Fitting GMM...')
    gmm.fit(data_reduc)
    pkl.dump(gmm, open('./cifar10_gmm64', 'wb'))
else:
    gmm = pkl.load(open('./cifar10_gmm64', 'rb'))
cluster_membership = gmm.predict(data_reduc)

myclustered_data, myclustered_labels = split_data_into_clusters(npdata, labels, cluster_membership)
clustered_data_reduc, _clustered_labels = split_data_into_clusters(data_reduc, labels, cluster_membership)
myavg_entropy, mysample_entropies = compute_entropy_for_clusters(myclustered_data, teacher_model, temperature=10.)

clustered_data_reduc = [pca.transform(cluster) for cluster in myclustered_data]
# dijs= pkl.load(open('./cifar10_pca256_gmm64_dij_matrix', 'rb'))
# dijs = dijs['dijs']
dijs=[compute_delta_ijs(data_,labels_, n_classes=10) for data_, labels_ in zip(clustered_data_reduc, myclustered_labels)]
examples_per_cluster = [len(d) for d in clustered_data]
bers = [ber_from_delta_ij(dij, n_classes=10) for dij in dijs]
alpha_hat = calculate_alpha_hat(.2, .1, 10, bers, examples_per_cluster)

mycluster_accuracies = compute_error_for_each_cluster(model=teacher_model,
                                                    cluster_data=myclustered_data,
                                                    cluster_labels=myclustered_labels)
pdb.set_trace()
plt.figure()
plt.scatter(bers, mycluster_accuracies)
plt.title('BER Estimate vs Train Accuracy on Cluster')
plt.grid()

cluster_accuracies = compute_error_for_each_cluster(model=teacher_model,
                                                    cluster_data=clustered_data,
                                                    cluster_labels=clustered_labels)
def bayesian_information_criterion():

    n_clusters = np.arange(2, 20)
    bics = []
    bics_err = []
    iterations = 2

    def SelBest(arr:list, X:int)->list:
        '''
        returns the set of X configurations with shorter distance
        '''
        dx=np.argsort(arr)[:X]
        return arr[dx]

    for n in tqdm.tqdm(n_clusters):
        tmp_bic = []
        for _ in tqdm.tqdm(range(iterations)):
            gmm = GaussianMixture(n, n_init=2).fit(data_reduc)

            tmp_bic.append(gmm.bic(data_reduc))
        val = np.mean(SelBest(np.array(tmp_bic), int(iterations / 5)))
        err = np.std(tmp_bic)
        bics.append(val)
        bics_err.append(err)

    plt.figure()
    plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
    plt.title("BIC Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("Score")
    plt.legend()

    plt.figure()
    plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
    plt.title("Gradient of BIC Scores", fontsize=20)
    plt.xticks(n_clusters)
    plt.xlabel("N. of clusters")
    plt.ylabel("grad(BIC)")
    plt.legend()
    plt.show()

# pdb.set_trace()
plt.figure()
plt.hist([np.median(sample_entropies[i]) for i in range(nclusters)])
plt.title('Median Entropy Per Cluster ')
plt.title('Median Entropy Per Cluster ')

plt.figure()
for ii in range(len(mysample_entropies)):
    plt.hist(np.log(mysample_entropies[ii]), bins=15)
# plt.title('Avg entropy based on my clusters')
plt.title('Per-Sample Entropy Histogram')

plt.figure()
plt.scatter(bers, alpha_hat, marker='o')

plt.figure()
plt.scatter([np.median(sample_entropies[i]) for i in range(len(sample_entropies))],
            [np.median(mysample_entropies[i]) for i in range(len(mysample_entropies))])
plt.title('PCA-GMM Clusters vs Autoencoder Clusters')
plt.xlabel('Median Sample Entropy for Clustering using Autoencoder')
plt.ylabel('Median Sample Entropy for Clustering using PCA-GMM')
plt.grid(True)

plt.figure()
plt.hist(alphahats1, alpha=.5)
plt.hist(calculate_alpha_hat(.1, .1, 10, bers, examples_per_cluster), alpha=.5)
# plt.title('Alpha Hats ')
plt.show()

print()


