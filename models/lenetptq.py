import torchvision.transforms as transforms
from quantize_ptq import PTQConv2d, PTQLinear
import tensorflow as tf
import torch.nn as nn
import torch
import pdb

FLAGS = tf.app.flags.FLAGS

class Lenet5PTQ(nn.Module):

    def __init__(self, is_quantized):
        super(Lenet5PTQ, self).__init__()

        self.is_quantized = is_quantized
        noise = FLAGS.noise_model
        n_bits_wt = FLAGS.n_bits_wt
        n_bits_act = FLAGS.n_bits_act

        self.conv1 = PTQConv2d(in_channels=1, out_channels=32, kernel_size=(5,5), padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv2 = PTQConv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)


        self.fc1 = PTQLinear(7 * 7 * 64, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = PTQLinear(1024, 10)

        self.layers_dict = {'conv1': self.conv1,
                       'conv2': self.conv2,
                       'fc1': self.fc1,
                       'fc2': self.fc2}
        # self.conv1 = QConv2d(in_channels=1, out_channels=32, kernel_size=(5,5), padding=2, is_quantized=self.is_quantized,
        #                      noise=noise, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        # self.conv2 = QConv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=2, is_quantized=self.is_quantized,
        #                      noise=noise, num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)
        # self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        # self.fc1 = QLinear(7* 7 * 64, 1024, is_quantized=self.is_quantized, noise=noise,
        #                    num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)

        # self.drop1 = nn.Dropout(0.5)

        # self.fc2 = QLinear(1024, 10, is_quantized=self.is_quantized, noise=noise,
        #                    num_bits_weight=n_bits_wt, num_bits_act=n_bits_act)

    def forward(self, x, is_quantized=False, n_bits_wt=8, n_bits_act=8, STE=False):
        relu = nn.ReLU()

        x = self.conv1(x, is_quantized, n_bits_act, n_bits_wt, STE=STE)
        x = self.pool1(x)
        x = relu(x)

        x = self.conv2(x, is_quantized, n_bits_act, n_bits_wt, STE=STE)
        x = self.pool2(x)
        x = relu(x)

        x = x.view(-1, 64 * 7 * 7)
        # if is_quantized:https://people.cs.umass.edu/~wallach/how_to_be_a_successful_phd_student.pdf
        #     pdb.set_trace()
        x = self.fc1(x, is_quantized, n_bits_act, n_bits_wt, STE=STE)
        x = relu(x)

        # if not self.is_quantized:
        x = self.drop1(x)

        x = self.fc2(x, is_quantized, n_bits_act, n_bits_wt, STE=STE)

        return x

    def freeze_unfreeze_layers(self, layers_to_freeze, mode):
        '''
        :param layers_to_freeze:
        :param mode:
        :return:
        '''
        for layer in self.layers_dict.keys():
            if layer in layers_to_freeze:
                if mode.lower()=='freeze':
                    self.layers_dict[layer].weight.requires_grad = False
                    self.layers_dict[layer].bias.requires_grad = False
                elif mode.lower()=='unfreeze':
                    self.layers_dict[layer].weight.requires_grad = True
                    self.layers_dict[layer].bias.requires_grad = True
                else:
                    print('ERROR!!!!!!!!!!!!!!!!!')
                    print('MUST SPECIFY EITHER mode=`freeze` or mode=`unfreeze`')
                    print('For function `freeze_unfreeze_layers`')
                    exit()




def lenet(**kwargs):
    return Lenet5PTQ()


