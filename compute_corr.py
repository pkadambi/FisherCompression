from data import get_dataset
import numpy as np
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualizaiton_utils import *
from scipy.stats import spearmanr
import os
from tensorboardX import SummaryWriter

c = ResnetConfig(n_epochs=200, dataset='cifar10', REGULARIZATION=None, TRAIN_FROM_SCRATCH=True, n_regularized_epochs=0)
model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
MODEL_SAVEPATH = c.model_savepath + 'checkpoint.pth'
model = model(**model_config)
model.cuda()
# model.apply(xavier_initialize_weights)
# model.apply(normal_initialize_biases)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


if 'cifar' in c.dataset:
    AUGMENT_TRAIN = True

transforms = { 'train': get_transform(name = c.dataset, augment=c.AUGMENT_TRAIN),

               'test': get_transform(name = c.dataset, augment=False)}



train_data = get_dataset(c.dataset, 'train', transform = transforms['train'])
train_data, valid_data = generate_validation_split(train_data, validation_split=.05)

train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)


test_data= get_dataset(c.dataset, 'test', transform = transforms['test'])
test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=c.batch_size, shuffle=True,
        num_workers=c.workers, pin_memory=True)


checkpoint = torch.load(MODEL_SAVEPATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
best_epoch = checkpoint['epoch']
best_loss = checkpoint['loss']
test_loss, ste_testacc = test_model(test_loader, model, criterion, printing=False)
print('###############################')
print('#                             #')
print('#  LOADING PRETRAINED MODEL   #')
print('#                             #')
print('###############################')

print('###############################')
print('#        LOADED MODEL         #')
print('#        TEST ACC: %.3f     #' % (ste_testacc))
print('#                             #')
print('###############################')


model.train()
n_validbatches = 0.

'''
Use Validation Samples to calculate fisher
'''
for name, p in list(model.named_parameters()):
    if hasattr(p, 'org'):
        p.data.copy_(p.org)

for iter, (inputs, target) in enumerate(valid_loader):
    n_batch = inputs.size()[0]

    inputs = inputs.cuda()
    target = target.cuda()
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    if iter>10:
        break
    for name, p in model.named_parameters():
        if hasattr(p, 'org'):
            if hasattr(p, 'fisher'):
                p.fisher += (n_batch/c.batch_size) * p.grad * p.grad
            else:
                p.fisher = (n_batch / c.batch_size) * p.grad * p.grad
                p.perturbation = p.org - p.data

names = []
corcoeffs = []
spearmans = []
perts = []

for name, p in model.named_parameters():
    if hasattr(p, 'perturbation'):
        weight = p.org.cpu().numpy()
        weight = np.abs(weight.ravel())

        pert = p.perturbation.cpu().numpy()
        pert= np.abs(pert.ravel())

        fisher = p.fisher.cpu().numpy()
        fisher = fisher.ravel()+.1

        print(fisher.shape)
        pearson = np.corrcoef(fisher, weight)
        spearman = spearmanr(fisher, weight)
        # pearson = np.corrcoef(fisher, pert)
        # spearman = spearmanr(fisher, pert)
        print('Pearson\n'+str(pearson))
        print('Spearman\n'+str(spearman))
        names.append(str(name))

        corcoeffs.append(pearson[0,1])
        spearmans.append(spearman[0])
        # plt.scatter(weight, fisher)

plt.figure()
plt.plot(corcoeffs, 'r', label = 'Pearson')
plt.plot(spearmans, 'b', label = 'Spearman')
plt.title('Plot of Spearman and Pearson Between \n FIM Diagonal and Weight Magnitude Resnet-18')
plt.legend(loc='Upper Left')
plt.show()

