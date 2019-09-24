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

c = ResnetConfig(n_epochs=200, dataset='cifar10', REGULARIZATION=None, TRAIN_FROM_SCRATCH=True, n_regularized_epochs=50)
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

def find_lower_threshold(all_fishers, n_percent):
    n_th = int(len(all_fishers)*n_percent)+1
    print(n_th)
    nth_smallest = np.partition(all_fishers, n_th)[n_th]
    print(nth_smallest)
    return nth_smallest


fishers = []

pctage_perturbed = [.025, .05, .075, .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375, .4,
              .425, .45, .475, .5, .525, .55, .575, .6]

n_mc_iters = 20

'''
Accuracies based on perturbing the smallest percentage of fisher information values
'''
fisher_lower_pct_accuracies = np.zeros([len(pctage_perturbed), n_mc_iters])
for i, pctage in enumerate(pctage_perturbed):

    for j in range(n_mc_iters):
        fishers = []

        print('\nFraction of Fisher Perturbed: ' + str(pctage))
        for name, p in model.named_parameters():
            if hasattr(p, 'perturbation'):
                fisher = p.fisher.cpu().numpy()
                fisher = fisher.ravel()

                fishers.append(fisher)

        all_fishers = np.concatenate(fishers)
        fshr = find_lower_threshold(all_fishers, pctage)

        for name, p in model.named_parameters():
            if hasattr(p, 'perturbation'):
                fisher = p.fisher.cpu().numpy()
                weight = p.org.cpu().numpy()
                fishers.append(fisher)

        for name, p in list(model.named_parameters()):
            if hasattr(p, 'org'):
                if hasattr(p, 'fisher'):
                    p.tmp = p.org.clone()
                    # p.org[p.fisher<fshr] = p.org[p.fisher<fshr] + torch.randn(p.org[p.fisher<fshr].size()).clamp_(-.1,.1).cuda()
                    p.org[p.fisher<fshr] = p.org[p.fisher<fshr] + torch.randn(p.org[p.fisher<fshr].size()).cuda()

                    # weight = p.org.cpu().numpy()
                    # fishers.append(fisher)
        test_loss, ste_testacc = test_model(test_loader, model, criterion, printing=False)
        fisher_lower_pct_accuracies[i, j] = ste_testacc
        # fisher_lower_pct_accuracies.append(ste_testacc)
        print(ste_testacc)
        for name, p in list(model.named_parameters()):
            if hasattr(p, 'org'):
                if hasattr(p, 'fisher'):
                    p.org.copy_(p.tmp)
        # test_loss, ste_testacc = test_model(test_loader, model, criterion, printing=False)
        # print(ste_testacc)

'''
Accuracies based on perturbing the weights randomly
'''
randomly_perturbed_accuracy = np.zeros([len(pctage_perturbed), n_mc_iters])
for i, pctage in enumerate(pctage_perturbed):
    for j in range(n_mc_iters):

        print('\nFraction of Wts Perturbed: ' + str(pctage))

        for name, p in list(model.named_parameters()):
            if hasattr(p, 'org'):
                p.tmp = p.org.clone()

                num_wts_layer = p.org.numel()
                n_wts_perturb = int(num_wts_layer * pctage) + 1

                if hasattr(p, 'fisher') and num_wts_layer > 100:

                    noise = torch.rand(n_wts_perturb)
                    inds = np.random.choice(num_wts_layer, n_wts_perturb)
                    tmp = p.org.clone()
                    tmp = tmp.view(-1)
                    tmp[inds] = tmp[inds] + noise.cuda()
                    tmp = tmp.view(p.org.size())
                    # print(p.org.size())
                    # print(tmp.size())
                    p.org.copy_(tmp)

        test_loss, ste_testacc = test_model(test_loader, model, criterion, printing=False)
        randomly_perturbed_accuracy[i, j] = ste_testacc
        print(ste_testacc)

        for name, p in list(model.named_parameters()):
            if hasattr(p, 'org'):
                if hasattr(p, 'fisher'):
                    p.org.copy_(p.tmp)
        # test_loss, ste_testacc = test_model(test_loader, model, criterion, printing=False)
        # print(ste_testacc)


fisher_lower_pct_accuracies = np.mean(fisher_lower_pct_accuracies, axis=1)
randomly_perturbed_accuracy = np.mean(randomly_perturbed_accuracy, axis=1)
pctage_perturbed = np.array(pctage_perturbed).reshape([len(pctage_perturbed),1])
plt.figure()
plt.title('ResNet18 on CIFAR10:\n Perturb Weights Based on Smallest Fisher')
plt.xlabel('Fraction of Wts Perturbed')
plt.ylabel('Accuracy')
plt.plot(pctage_perturbed, fisher_lower_pct_accuracies)
plt.grid()

plt.figure()
plt.title('ResNet18 on CIFAR10:\n Accuracy Under Clamped (-0.1,0.1) Normal Perturb to Weights')
plt.xlabel('Fraction of Wts Perturbed')
plt.ylabel('Accuracy')
plt.plot(pctage_perturbed, fisher_lower_pct_accuracies, 'r', label='Fisher Based (smallest)')
plt.plot(pctage_perturbed, randomly_perturbed_accuracy, 'b', label='Randomly Perturb')
plt.legend(loc='Upper Right')
plt.grid()


# plt.figure()
# plt.hist(all_fishers[np.random.choice(len(all_fishers), 2000000)], bins=100, log = True)
# plt.grid()
plt.show()

