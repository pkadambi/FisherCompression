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
        self.image_data = np.array([image for image in Images])
        self.image_data = self.image_data.astype('uint8')
        self.labels = np.array([label for label in Y])

        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt

        # trans = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.RandomRotation(45, fill=(0,)),
        #     # transforms.RandomCrop(input_size, pad_if_needed=True),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(),
        # ])
        # im = self.image_data[0]
        # plt.figure()
        # plt.imshow(im, cmap='gray')
        # pdb.set_trace()
        #
        # plt.figure()
        # imt = trans(self.image_data[0])
        #
        # plt.imshow(imt, cmap='gray')

        self.transform = transform

        # pdb.set_trace()

    # The number of items in the dataset
    def __len__(self):
        return int(self.image_data.shape[0])

    # The Dataloader is a generator that repeatedly calls the getitem method.
    # getitem is supposed to return (X, Y) for the specified index.
    def __getitem__(self, index):
        img = self.image_data[index]

        # 8 bit images. Scale between [0,1]. This helps speed up our training
        # img = img.reshape(28, 28) / 255.0
        # img = Image.fromarray(img)
        # Input for Conv2D should bee Channels x Height x Width

        label = self.labels[index]

        if self.transform is not None:
            # pdb.set_Trace
            img = self.transform(img)

        return img, label

# n = notMNIST()
# pdb.set_trace()
print()