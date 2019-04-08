import torchvision.transforms as transforms
import torch
import torch.nn as nn
import binarized_modules as bnn

class BinaryLenet(nn.Module):

    def __init__(self):
        super(BinaryLenet, self).__init__()

        self.conv = nn.Sequential(
            bnn.BinarizeConv2d(in_channels=1, out_channels=20, kernel_size=(5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            bnn.BinarizeConv2d(in_channels=20, out_channels=50, kernel_size=(5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

        )

        self.FC = nn.Sequential(
            bnn.BinarizeLinear(7*7*50, 1024),
            nn.ReLU(),
            # nn.Dropout(0.5),
            bnn.BinarizeLinear(1024, 10),
            # nn.Linear(28*28, 10),
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 50 * 7 * 7)
        x = self.FC(x)
        return x




def lenet_binary(**kwargs):
    return BinaryLenet()








