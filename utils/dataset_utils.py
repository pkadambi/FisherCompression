import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit



class ResnetCifar10Config():

    def __init__(self):
        self.dataset = 'cifar10'
        self.n_epochs = 10
        self.model_name = 'resnet_binary'
        self.transform = None
        self.workers = 4
        self.batch_size = 128
        self.input_size = (3, 32, 28)
        self.print_interval = 75



class LenetFashionMNISTConfig():

    def __init__(self, binary = True, n_epochs = 10, batch_size = 128):
        self.dataset = 'fashionmnist'
        self.n_epochs = n_epochs
        self.transform = None
        self.workers = 4
        self.batch_size = batch_size
        self.input_size = (28, 28)
        self.print_interval = 75

        self.model_name = 'lenet'
        if binary:
            self.model_name += '_binary'



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

