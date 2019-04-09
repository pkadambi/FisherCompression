from data import get_dataset
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualizaiton_utils import *
import os, sys
cudnn.benchmark = True

torch.cuda.set_device(0)

c = LenetFashionMNISTConfig(n_epochs=100, USE_FISHER=True, n_fisher_epochs=10)
# c = ResnetConfig(n_epochs=50, dataset='fashionmnist')

c.print_interval = 50
model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
MODEL_SAVEPATH = c.model_savepath + 'checkpoint.pth'
TRAIN_FROM_SCRATCH = False
model = model(**model_config)
# model.apply(xavier_initialize_weights)
# model.apply(normal_initialize_biases)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

transforms = { 'train': get_transform(name = c.dataset, augment=False),

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


'''
Note that the following code is basically model agnostic, 
'''

record_interval = 10


def train_from_scratch(config, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH, record_interval = 10):

    c = config
    lossval = np.array([])
    valid_acc = np.array([])
    test_acc = np.array([])
    tr_acc = np.array([])
    n_iters = 0

    for epoch in range(c.n_epochs):
        regime = getattr(model, 'regime')

        if 'lenet' not in c.model_name:
            optimizer = adjust_optimizer(optimizer, epoch, regime)

        print(optimizer)
        model.train()

        for iter, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            '''
            The STE procedure below
            '''
            output = model(inputs)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()

            accuracy_ = accuracy(output, target)[0].item()
            lossval_ = loss.item()
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
                    # print(name)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
            # exit()
            if iter % c.print_interval ==0:
                print('Epoch %d | Iters: %d | Train Loss %.3f | Acc %.2f' % (epoch+1, n_iters, lossval_, accuracy_))

            if iter % record_interval==0:
                lossval = np.hstack([lossval, lossval_])
                tr_acc = np.hstack([tr_acc, accuracy_])
            n_iters+=1

        val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)

        #Save model if validation acc has improved

        if len(valid_acc) >= 1:
            if val_acc > np.max(valid_acc):
                print('Found new best model! Saving model from this epoch.')
                if not os.path.exists(c.model_savepath):
                    os.makedirs(c.model_savepath)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, MODEL_SAVEPATH)

        valid_acc = np.hstack([valid_acc, val_acc])
        test_acc = np.hstack([test_acc, test_acc_])


        print('\nEpoch %d | Valid Loss %.3f | Valid Acc %.2f \n' % (epoch + 1, val_loss, val_acc))
        print('***********************************************\n')


    best_acc = np.max(valid_acc)
    best_epoch = np.argwhere(valid_acc==best_acc)+1
    print('--------------------------------------------------')
    print('-----FINISHED STE TRAINING-----')
    print('----------Results for Training END----------')
    print('End Validation Acc %.3f | End Epoch %d' % (valid_acc[c.n_epochs-1], c.n_epochs))
    print('End TEST Acc %.3f | End Epoch %d\n' % (test_acc[c.n_epochs-1], c.n_epochs))


    print('----------Results for BEST EPOCH----------')
    print('Best Validation Acc: %.3f | At Epoch: %d' % (valid_acc[best_epoch-1], best_epoch))
    print('Test Acc at Best Valid Epoch: %.3f | Model From Epoch: %d' % (test_acc[best_epoch-1], c.n_epochs))

    x = np.arange(len(lossval)*record_interval)
    two_scale_plot(lossval, tr_acc, y1_label= 'Train Loss', y2_label = 'Train Acc')

    training_loss = lossval
    training_accuracy = tr_acc

    return training_loss, training_accuracy, best_epoch

'''

Plotting Utilities

'''
def train_fisher(config, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH, load_model = False):

    c = config
    lossval = np.array([])
    valid_acc = np.array([])
    test_acc = np.array([])
    tr_acc = np.array([])
    n_iters = 0
    c.gamma = .1

    if load_model:
        checkpoint = torch.load(MODEL_SAVEPATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)
        print('###############################')
        print('#                             #')
        print('#  LOADING PRETRAINED MODEL   #')
        print('#                             #')
        print('###############################')

        print('###############################')
        print('#        LOADED MODEL         #')
        print('#        TEST ACC: %.3f       #' % (test_acc_))
        print('#                             #')
        print('###############################')

        print('###############################')
        print('#                             #')
        print('#  STARTING FISHER TRAINING   #')
        print('#                             #')
        print('###############################')

    else:
        best_epoch = 0
    FISHER_SAVEPATH = c.model_savepath+'checkpoint_fisher.pth'
    n_tot_epochs =  best_epoch+c.n_fisher_epochs
    #plot loss and accuracy curves
    for epoch in range(best_epoch, n_tot_epochs ):
        regime = getattr(model, 'regime')

        if 'lenet' not in c.model_name:
            optimizer = adjust_optimizer(optimizer, epoch, regime)

        print(optimizer)
        model.train()

        for iter, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            '''
            set fp wts
            '''
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.perturbation = p.org - p.data
                    p.data.copy_(p.org)
                # Next time, dont do a binary pass
                if hasattr(p, 'binary_pass'):
                    p.binary_pass = False

            '''
            fp_pass
            '''
            output = model(inputs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()

            for name, p in list(model.named_parameters()):
                if hasattr(p, 'grad'):
                    p.fp_grad = p.grad.clone()
                if hasattr(p, 'binary_pass'):
                    p.binary_pass = True

            '''
            quantized pass
            '''
            output = model(inputs)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()

            '''
            combine quantized update with regularizer
            '''
            for p in model.parameters():
                pert = p.org - p.data
                rg_grad = c.gamma * p.fp_grad * p.fp_grad * pert
                p.grad.copy_(p.grad + rg_grad)

            accuracy_ = accuracy(output, target)[0].item()
            lossval_ = loss.item()
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
                    # print(name)
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
            # exit()
            if iter % c.print_interval == 0:
                print('Epoch %d | Iters: %d | Train Loss %.3f | Acc %.2f' % (epoch + 1, n_iters, lossval_, accuracy_))

            if iter % record_interval == 0:
                lossval = np.hstack([lossval, lossval_])
                tr_acc = np.hstack([tr_acc, accuracy_])
            n_iters += 1

        val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)

        if len(valid_acc) >= 1:
            if val_acc > np.max(valid_acc):
                print('Found new best model! Saving model from this epoch.')
                if not os.path.exists(c.model_savepath):
                    os.makedirs(c.model_savepath)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, FISHER_SAVEPATH)

        valid_acc = np.hstack([valid_acc, val_acc])
        test_acc = np.hstack([test_acc, test_acc_])
        print('\nEpoch %d | Valid Loss %.3f | Valid Acc %.2f \n' % (epoch + 1, val_loss, val_acc))
        print('***********************************************\n')

    best_acc = np.max(valid_acc)

    #add the best epoch number to the previous best epoch
    best_epoch = np.argwhere(valid_acc == best_acc) + 1 + best_epoch

    print('-----------------F-I-S-H-E-R---------------------')
    print('-------------FINISHED FISHER TRAINING------------')
    print('-------------Results for Training END------------')
    print('End Validation Acc %.3f | End Epoch %d' % (valid_acc[c.n_fisher_epochs - 1],  n_tot_epochs))
    print('End TEST Acc %.3f | End Epoch %d\n' % (test_acc[c.n_fisher_epochs - 1], n_tot_epochs))

    print('--------------BEST FISHER-----------------')
    print('----------Results for BEST EPOCH----------')
    print('Best Validation Acc: %.3f | At Epoch: %d' % (valid_acc[c.n_fisher_epochs - 1], best_epoch))
    print('Test Acc at Best Valid Epoch: %.3f | Model From Epoch: %d' % (test_acc[c.n_fisher_epochs - 1], best_epoch))

if TRAIN_FROM_SCRATCH:
    train_from_scratch(c, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH)


if c.USE_FISHER_REG:
    MODEL_SAVEPATH = './checkpoints/lenet_binary_gold/fashionmnist/checkpoint.pth'
    train_fisher(c, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH, load_model = True)


plt.show()





