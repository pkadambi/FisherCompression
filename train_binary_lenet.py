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

c = LenetFashionMNISTConfig(n_epochs=100, USE_FISHER=True)
# c = ResnetConfig(n_epochs=50, dataset='fashionmnist')

c.print_interval = 50
model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
MODEL_SAVEPATH = c.model_savepath + 'checkpoint.pth'
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
lossval = np.array([])
valid_acc = np.array([])
test_acc = np.array([])
tr_acc = np.array([])

regime = getattr(model, 'regime')

n_iters = 0
record_interval = 10


for epoch in range(c.n_epochs):

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




'''

Plotting Utilities

'''

#plot loss and accuracy curves


x = np.arange(len(lossval)*record_interval)
two_scale_plot(lossval, tr_acc, y1_label= 'Train Loss', y2_label = 'Train Acc')
plt.show()

if c.USE_FISHER_REG:

    #TODO: CODE TO RESTORE THE CORRECT MODEL

    for epoch in range(c.n_fisher_epochs):

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
                #Next time, dont do a binary pass
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
            if iter % c.print_interval ==0:
                print('Epoch %d | Iters: %d | Train Loss %.3f | Acc %.2f' % (epoch+1, n_iters, lossval_, accuracy_))

            if iter % record_interval==0:
                lossval = np.hstack([lossval, lossval_])
                tr_acc = np.hstack([tr_acc, accuracy_])
            n_iters+=1

        val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)
        valid_acc = np.hstack([valid_acc, val_acc])
        test_acc = np.hstack([test_acc, test_acc_])


        print('\nEpoch %d | Valid Loss %.3f | Valid Acc %.2f \n' % (epoch + 1, val_loss, val_acc))
        print('***********************************************\n')











