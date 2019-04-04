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

cudnn.benchmark = True

class Config():

    def __init__(self):
        self.dataset = 'fashionmnist'
        self.n_epochs = 32
        self.model_name = 'lenet'
        self.transform = None
        self.workers = 4
        self.batch_size = 128
        self.input_size = (28, 28)

c = Config()

model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
model = model(**model_config)

optimizer = torch.optim.Adam(model.parameters())
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

model.cuda()

print(model)
for i ,p in enumerate(model.parameters()):
    if i==0:
        print(p)

lossval = np.array([])

for i in range(c.n_epochs):
    # if i>0:
    #     break
    for iter, (inputs, target) in enumerate(train_loader):
        inputs = inputs.cuda()
        # print(inputs[0])
        # print(inputs.size())
        # exit()
        target = target.cuda()

        output = model(inputs)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        lossval_ = loss.cpu().data.numpy()
        if iter %100 ==0:
            print('Loss %.3f' % lossval_)
            print(output[0])
            print()
        lossval = np.hstack([lossval, lossval_])


        # print()

    print('\n\n\nLoss after epoch %d: %.2f \n\n\n' % (i+1, lossval_))

for i ,p in enumerate(model.parameters()):
    if i==0:
        print(p)

plt.figure()
plt.plot(lossval)
plt.grid()
plt.title('Loss vs iterations')
plt.show()

















