import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from notmnist_dataset import *


_DATASETS_MAIN_PATH = '/home/prad/Datasets'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'fashionmnist': os.path.join(_DATASETS_MAIN_PATH, 'FashionMNIST'),
    'notmnist' : './notMNISTDataset',
    'imagenet': {
    'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
    'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    }
}


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name.lower() == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name.lower() == 'cifar100':
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name.lower() == 'imagenet':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)

    elif name.lower() == 'fashionmnist':
        return datasets.FashionMNIST(root = _dataset_path['fashionmnist'],
                                     train = train,
                                     transform = transform,
                                     target_transform = target_transform,
                                     download = download)

    elif name.lower() == 'mnist':
        return datasets.MNIST(root = _dataset_path['mnist'],
                              train = train,
                              transform = transform,
                              target_transform = target_transform,
                              download = download)

    elif name.lower() == 'notmnist':
        return notMNIST(root = _dataset_path['notmnist'],
                        train = train,
                        transform = transform)

    else:
        print('ERROR!! Please specify valid dataset name!')
        raise Exception
