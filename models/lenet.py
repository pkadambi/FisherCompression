import torchvision.transforms as transforms
from quantize import QConv2d, QLinear
import tensorflow as tf
import torch.nn as nn
import torch

FLAGS = tf.app.flags.FLAGS

class Lenet5(nn.Module):

    def __init__(self):
        super(Lenet5, self).__init__()

        is_quantized = FLAGS.is_quantized

        self.conv1 = QConv2d(in_channels=1, out_channels=32, kernel_size=(5,5), padding=2,
                                is_quantized=is_quantized)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv2 = QConv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=2,
                                is_quantized=is_quantized)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = QLinear(7* 7 * 64, 1024, is_quantized=is_quantized)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = QLinear(1024, 10, is_quantized=is_quantized)

    def forward(self, x):
        relu = nn.ReLU()

        x = self.conv1(x)
        x = self.pool1(x)
        x = relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = relu(x)

        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = relu(x)
        x = self.drop1(x)

        x = self.fc2(x)

        return x


def lenet(**kwargs):
    return Lenet5()








