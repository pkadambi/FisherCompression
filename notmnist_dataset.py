import pdb
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
# from scipy.misc import imread
from imageio import imread
from torch import Tensor

"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.

Set root to point to the Train/Test folders.
"""

# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

    # The init method is called when this class will be instantiated.
    def __init__(self, root='./notMNISTDataset', train=True, transform=None):
        Images, Y = [], []

        if train:
            root = os.path.join(root, 'Train')
        else:
            root = os.path.join(root, 'Test')

        folders = os.listdir(root)

        for folder in folders:
            folder_path = os.path.join(root, folder)
            for ims in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, ims)
                    Images.append(np.array(imread(img_path)))
                    Y.append(ord(folder) - 65)  # Folders are A-J so labels will be 0-9
                except:
                    # Some images in the dataset are damaged
                    print("File {}/{} is broken".format(folder, ims))


        self.image_data = np.array([image for image in Images]).astype('float32')
        self.labels = np.array([y for y in Y])

        self.transform = transform

        # pdb.set_trace()

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        img = self.image_data[index]
        label = self.labels[index]

        # Input for Conv2D should be Channels x Height x Width
        # 8 bit images. Scale between [0,1]. This helps speed up our training
        img = img.reshape(28, 28) / 255.0

        if self.transform is not None:
            # pdb.set_trace()
            img = self.transform(img)

        return img, label

n = notMNIST()
# pdb.set_trace()
print()