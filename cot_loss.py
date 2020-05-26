import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
classes = FLAGS.n_classes

# Complement Entropy (CE)


class ComplementEntropy(nn.Module):

    def __init__(self):
        super(ComplementEntropy, self).__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        self.classes = classes
        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return loss

class ShannonEntropy(nn.Module):

    def __init__(self):
        super(ShannonEntropy, self).__init__()

    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat):
        self.batch_size = len(yHat)
        self.classes = classes
        yHat = F.softmax(yHat, dim=1)
        Px = yHat + 1e-10  # avoiding numerical issues (first)
        Px = Px / torch.sum(Px) #renormalize after avoid numerical issue
        Px_log = torch.log(yHat + 1e-10)
        output = torch.matmul(Px , Px_log.T)
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return -loss