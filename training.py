import torch
import torchvision
from data import get_dataset
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.model_utils import accuracy, test_model, xavier_initialize_weights, normal_initialize_biases, constant_initialize_bias
cudnn.benchmark = True

class Config():

    def __init__(self):
        self.dataset = 'fashionmnist'
        self.n_epochs = 15
        self.model_name = 'lenet'
        self.transform = None
        self.workers = 4
        self.batch_size = 128
        self.input_size = (28, 28)
        self.print_interval = 75
torch.cuda.set_device(0)

c = Config()

model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
model = model(**model_config)
model.apply(xavier_initialize_weights)
model.apply(normal_initialize_biases)
model.cuda()

optimizer = optim.Adam(model.parameters())
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()

transforms = { 'train': get_transform(name = c.dataset, augment=False),
               'test': get_transform(name = c.dataset, augment=False)}

train_data = get_dataset(c.dataset, 'train', transform = transforms['train'])
train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)


test_data= get_dataset(c.dataset, 'test', transform = transforms['test'])
test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)


lossval = np.array([])


model.eval()
for i in range(c.n_epochs):
    # if i>0:
    #     break
    accuracy_ = 0
    for iter, (inputs, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        # inputs = inputs.view(-1,28*28)

        target = target.cuda()

        output = model(inputs)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossval_ = loss.cpu().data.numpy()

        accuracy_ = accuracy(output, target)[0]

        if iter % c.print_interval ==0:
            print('Epoch %d | Train Loss %.3f | Acc %.2f' % (i+1, lossval_, accuracy_))

        lossval = np.hstack([lossval, lossval_])
    print('***********************************************')

test_model(test_loader, model)
plt.figure()
plt.plot(lossval)
plt.grid()
plt.title('Loss vs iterations')
plt.show()

















