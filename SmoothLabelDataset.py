import pdb
import pandas as pd
import numpy as np
from torch.utils import data
import pickle as pkl
# from data_loading import *
from utils import *
import os
from pathlib import Path
import tqdm
from data import *
from preprocess import *
from utils.model_utils import *


def compute_smoothed_labels(labels, method,
                            cluster_memberships=None,
                            cluster_alphas=None, uls_alpha=None):
    '''
    :param labels: n_samples x 1 array (ground truth class number)
    :param method: can be either `ULS` or `SLS`
    :param cluster_memberships: cluster membership for each datapoint, must be passed in if method is SLS
    :param cluster_alphas: alphas for each cluster, index of alpha corresponds to cluster number must be passed in if method is SLS
    :param alpha:

    :return:
    '''

    assert len(np.unique(cluster_alphas)) == len(np.unique(cluster_memberships))

    n_classes = len(np.unique(labels))
    K = n_classes

    # Step 1: get the one hot labels
    oh_labels = to_one_hot(labels, n_classes)

    if method.lower() == 'sls':

        #step 2: create the array of cluster alphas the same size as
        alphahat_per_sample = cluster_alphas[cluster_memberships.astype('int')].reshape(-1,1)

        #step 3: create the
        smoothed_labels = oh_labels + alphahat_per_sample * (1/(K - 1))

        #Step 4: for the gt label sutract out the alpha_hat

        for i in range(len(smoothed_labels)):
            smoothed_labels[i,labels[i]] = 1 - alphahat_per_sample[i]

    elif method.lower() == 'uls':

        smoothed_labels = oh_labels + uls_alpha * (1/(K - 1))

        for i in range(len(smoothed_labels)):
            smoothed_labels[i, labels[i]] = 1 - uls_alpha

    else:
        exit('Error!: Invalid label smoothing method specified.')

    return smoothed_labels



class SmoothedLabelDataset(data.Dataset):

    def __init__(self, split, dataset, smoothing_method, alpha_val=None):
        '''

        :param split:
        :param dataset: can be `fashionmnist` `cifar10` `cifar100`
        :param clustermember_path:
        :param avg_embedding:
        '''
        #avg_embedding - whether to average across the tiles (not across the entire proteins)

        super().__init__()

        data = get_dataset(name=dataset, split=split, transform=get_transform(name=dataset, augment=True))

        # Step1: get labels and
        if dataset=='fashionmnist':
            self.npdata = np.vstack([tr[0].numpy().reshape(-1, 1, 28, 28) for tr in data])
        elif dataset=='cifar10' or dataset=='cifar100':
            self.npdata = np.vstack([tr[0].numpy().reshape(-1, 3, 32, 32) for tr in data])
        else:
            exit('Error: Specify a valid dataset')


        #ONLY DO LABEL SMOTHING IF IT IS THE TRAINING SET
        self.labels = np.array([tr[1] for tr in data])
        self.oh_labels = to_one_hot(self.labels, n_classes=10)

        if split=='train':

            if smoothing_method.lower()=='uls':

                self.smoothed_labels = compute_smoothed_labels(self.labels, method='ULS', uls_alpha=alpha_val)

            elif smoothing_method.lower()=='sls':

                if dataset=='cifar10':
                    if alpha_val==.1:
                        alphas = np.load('./SLS_param/cifar10_myregularizer_disturbing288.npy')
                    elif alpha_val==.2:
                        alphas = np.load('./SLS_param/cifar10_myregularizer_disturbing295.npy')
                    elif alpha_val==.3:
                        alphas = np.load('./SLS_param/cifar10_myregularizer_disturbing299.npy')

                    cluster_membership = np.load('./SLS_param/CIFAR10TrainClusterLoc.npy')

                elif dataset.lower()=='fashionmnist':
                    #load the correct alpha file
                    #TODO: get the alpha files for fashionmnist
                    #TODO: load the correct cluster membership file
                    pass

                self.alphas = alphas[cluster_membership.astype('int')]

                self.smoothed_labels = compute_smoothed_labels(self.labels, method='SLS',cluster_memberships=
                cluster_membership, cluster_alphas=alphas)

        self.n_train = self.npdata.shape[0]
        self.data = torch.Tensor(self.npdata)
        self.smoothed_labels = torch.Tensor(self.smoothed_labels)

    def __getitem__(self, index):
        '''
        :param index:
        :return: descrip, seq, tiled_seq,
        '''
        return self.data[index], self.labels[index], self.smoothed_labels[index]

    def __len__(self):
        return self.n_train

    def to_one_hot(self, labels, n_classes, lowest_class_num=0):
        '''
        :param labels: should be size (n_samples, 1), values in the array should be integers or integer valued floats (will be cast)
        :param lowest_class_num: the value of the lowest class label (ie whether first class is labeled 0 or 1)
        :return: one hot encoded array size (n_samples, n_classes)
        '''
        n_samples = np.shape(labels)[0]
        # print(n_samples)
        # print(n_classes)
        one_hot_array = np.zeros([n_samples, n_classes])

        # if the lowest class number is
        labels = labels - lowest_class_num
        labels.astype('int')
        one_hot_array[np.arange(n_samples), labels.T.astype(int)] = 1
        return one_hot_array

    def compute_uls_labels(self):
        pass

membfile = './SLS_param/CIFAR10TrainClusterLoc.npy'
alphas_file_1 = './SLS_param/cifar10_myregularizer_disturbing288.npy'
train_data = SmoothedLabelDataset(split='train', dataset='cifar10', smoothing_method='ULS', alpha_val=.1)
get_dataset(name='train', split='test', transform=get_transform(name='cifar10', augment=True))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True,
                                           num_workers=4, pin_memory=True)


# for iter, (inputs, targets, smoothed_target) in enumerate(train_loader):
#     print(iter)
#     print(inputs)
#     print(targets)
#     print(smoothed_target)
#     print()



print()

