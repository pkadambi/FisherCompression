from data import get_dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualization_utils import *
import torch.optim as optim

#Torch gpu stuff
cudnn.benchmark = True
torch.cuda.set_device(0)

batch_size = 128
n_workers = 4
dataset = 'fashionmnist'
n_epochs = 25
record_interval = 100


criterion = nn.CrossEntropyLoss()

train_data = get_dataset(name = dataset, split = 'train', transform=get_transform(name=dataset, augment=False))
test_data = get_dataset(name = dataset, split = 'test', transform=get_transform(name=dataset, augment=False))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                          num_workers=n_workers, pin_memory=True)

i=0
model = models.Lenet5()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(n_epochs):

    for iter, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = inputs.cuda()
        targets = targets.cuda()

        output = model(inputs)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        train_acc = accuracy(output, targets).item()
        lossval = loss.item()



        if i%record_interval==0 or i==0:
            print('Step [%d] | Loss [%.3f] | Acc [%.3f]' % (i, lossval, train_acc))

        i+=1


    print('\n*** TESTING ***\n')
    test_loss, test_acc = test_model(test_loader, model, criterion, printing=False)
    print('End Epoch [%d]| Test Loss [%.3f]| Test Acc [%.3f]' % (epoch, test_loss, test_acc))

    print('\n*** EPOCH END ***\n')



exit()









