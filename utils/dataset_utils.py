import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit



class ResnetConfig():

    def __init__(self, binary = True, n_epochs = 50, REGULARIZATION=None, n_regularized_epochs = 0, TRAIN_FROM_SCRATCH = True,
                 gamma=.1, dataset='cifar10'):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.transform = None
        self.workers = 4
        self.REGULARIZATION = REGULARIZATION
        self.n_regularized_epochs = n_regularized_epochs
        self.batch_size = 128
        self.input_size = None
        # self.input_size = (3, 32, 28)
        self.TRAIN_FROM_SCRATCH = TRAIN_FROM_SCRATCH

        self.print_interval = 75
        self.model_name = 'resnet'
        self.gamma = gamma
        self.AUGMENT_TRAIN = True
        self.n_iters = 0
        self.record_interval = 50
        if binary:
            self.model_name += '_binary'

        self.model_savepath = './checkpoints/'+self.model_name+'/'+self.dataset+'/'


class VGGConfig():

    def __init__(self, binary = True, n_epochs = 50, USE_FISHER=False, n_regularized_epochs = 0, TRAIN_FROM_SCRATCH = True,
                 gamma=.1, dataset='cifar10'):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.transform = None
        self.workers = 4
        self.USE_FISHER_REG = USE_FISHER
        self.n_regularized_epochs = n_regularized_epochs
        self.batch_size = 128
        self.input_size = None
        # self.input_size = (3, 32, 28)
        self.TRAIN_FROM_SCRATCH = TRAIN_FROM_SCRATCH

        self.print_interval = 75
        self.model_name = 'vgg'
        self.gamma = gamma
        self.AUGMENT_TRAIN = True
        self.n_iters = 0
        if binary:
            self.model_name += '_binary'

        self.model_savepath = './checkpoints/'+self.model_name+'/'+self.dataset+'/'

class LenetFashionMNISTConfig():

    def __init__(self, binary = True, n_epochs = 50, REGULARIZATION=None, n_regularized_epochs = 0, TRAIN_FROM_SCRATCH=False,gamma=.1):
        self.dataset = 'fashionmnist'
        self.n_epochs = n_epochs
        self.transform = None
        self.workers = 4
        self.batch_size = 128
        self.input_size = (28, 28)
        self.print_interval = 75

        '''
        REGULARIZATION: can be either
        None
        'KL'
        'Fisher'
        'Identity'
        ''
        '''
        self.REGULARIZATION = REGULARIZATION
        self.TRAIN_FROM_SCRATCH = TRAIN_FROM_SCRATCH
        self.n_regularized_epochs = n_regularized_epochs
        self.model_name = 'lenet'
        self.gamma = gamma
        self.AUGMENT_TRAIN = False
        self.n_iters = 0
        self.record_interval = 50
        if binary:
            self.model_name += '_binary'

        self.model_savepath = './checkpoints/'+self.model_name+'/'+self.dataset+'/'


def generate_validation_split(dataset, validation_split):
    '''

    :param dataset:
    :param validation_split:
    :return:
    '''
    n_samps = np.shape(dataset.data)[0]
    CVClass = StratifiedShuffleSplit

    #random state kept constant for repeatability
    cv = CVClass(test_size=validation_split,
                 train_size=1-validation_split,
                 random_state=99)

    train, val = next(cv.split(X=np.reshape(dataset.data, [n_samps , -1]), y=dataset.targets))

    train_dataset = Subset(dataset, indices = train)
    validation_dataset = Subset(dataset, indices = val)

    return train_dataset, validation_dataset

