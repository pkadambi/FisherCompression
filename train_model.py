from data import get_dataset
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualizaiton_utils import *
import os
from tensorboardX import SummaryWriter
cudnn.benchmark = True

torch.cuda.set_device(0)

#-----------------------------------------------------------------------------------------------------------------------
#
#                           CONFIGURATION
#
# c = LenetFashionMNISTConfig(n_epochs=10, REGULARIZATION='Fisher', n_fisher_epochs=1, TRAIN_FROM_SCRATCH=True, gamma=.1)
# c = LenetFashionMNISTConfig(n_epochs=50, REGULARIZATION=None, n_fisher_epochs=1, TRAIN_FROM_SCRATCH=True, gamma=.1)
# c.print_interval = 50

c = ResnetConfig(n_epochs=200, dataset='cifar10', REGULARIZATION=None, TRAIN_FROM_SCRATCH=True, n_fisher_epochs=0)
c.print_interval = 50
#
#-----------------------------------------------------------------------------------------------------------------------


model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
MODEL_SAVEPATH = c.model_savepath + 'checkpoint.pth'
model = model(**model_config)
# model.apply(xavier_initialize_weights)
# model.apply(normal_initialize_biases)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
kl_criterion = nn.KLDivLoss()
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

    #Clear the logfile
    os.makedirs(os.path.dirname(MODEL_SAVEPATH), exist_ok=True)
    logdir = os.path.dirname(MODEL_SAVEPATH)
    logfile = open(logdir + 'log.txt', 'a')
    logfile.close()

    for epoch in range(c.n_epochs):

        logdir = os.path.dirname(MODEL_SAVEPATH)
        logfile = open(logdir + 'log.txt', 'a')

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
                if hasattr(p, 'grad'):
                    if p.grad is not None:
                        # p.grad.copy_(p.grad.clamp_(-.1, .1))
                        pass

                if hasattr(p, 'org'):
                    p.perturbation = p.data - p.org

                    if config.n_iters % config.record_interval == 0:
                        writer.add_histogram(name+' perturbation', p.perturbation.clone().cpu().data.numpy(), config.n_iters)
                        writer.add_histogram(name+' FP', p.org.clone().cpu().data.numpy(), config.n_iters)
                        writer.add_histogram(name+' Quant', p.data.clone().cpu().data.numpy(), config.n_iters)
                    p.data.copy_(p.org)

            optimizer.step()

            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

            if iter % c.print_interval ==0:
                logstr = 'Epoch %d | Iters: %d | Train Loss %.5f | Acc %.3f' % (epoch+1, config.n_iters, lossval_, accuracy_)
                print(logstr)
                logstr +='\n'
                logfile.write(logstr)

            if iter % record_interval==0:
                lossval = np.hstack([lossval, lossval_])
                tr_acc = np.hstack([tr_acc, accuracy_])



            config.n_iters += 1

        val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)

        #Save model if validation acc has improved
        if len(valid_acc) >= 1:
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)

            if val_acc > np.max(valid_acc):
                print('Found new best model! Saving model from this epoch.')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, MODEL_SAVEPATH)

            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
        valid_acc = np.hstack([valid_acc, val_acc])
        test_acc = np.hstack([test_acc, test_acc_])


        epoch_logstring = '\nEpoch %d | Valid Loss %.5f | Valid Acc %.2f \n' % (epoch + 1, val_loss, val_acc)
        epoch_logstring = epoch_logstring + '***********************************************\n\n'
        print(epoch_logstring)
        logfile.write(epoch_logstring)

        logfile.close()


    best_acc = np.max(valid_acc)
    best_epoch = np.argwhere(valid_acc==best_acc)+1
    print('--------------------------------------------------')
    print('-------------FINISHED STE TRAINING----------------')
    print('-------------Results for Training END-------------')
    print('End Validation Acc %.3f | End Epoch %d' % (valid_acc[c.n_epochs-1], c.n_epochs))
    print('End TEST Acc %.3f | End Epoch %d\n' % (test_acc[c.n_epochs-1], c.n_epochs))


    print('--------------Results for BEST EPOCH--------------')
    print('Best Validation Acc: %.3f | At Epoch: %d' % (valid_acc[best_epoch-1], best_epoch))
    print('Test Acc at Best Valid Epoch: %.3f | Model From Epoch: %d' % (test_acc[best_epoch-1], c.n_epochs))


    #TODO: Save weight pdf into results file (weight pdf from optimal iteration, AND from end of training)
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
    c.gamma=0.1

    if load_model:
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
            model.train()
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.perturbation = p.org - p.data
                    p.data.copy_(p.org)

                    if config.n_iters % config.record_interval==0:
                        writer.add_histogram(name, p.perturbation.clone().cpu().data.numpy(), config.n_iters)

            '''
            fp_pass
            '''
            output_fp = model(inputs)
            loss = criterion(output_fp, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            for name, p in list(model.named_parameters()):
                if hasattr(p, 'grad'):
                    if p.grad is not None:
                        p.fp_grad = p.grad.clone()
                if hasattr(p, 'binary_pass'):
                    p.binary_pass = True
            model.train()

            '''
            quantized pass
            '''
            output = model(inputs)
            loss = criterion(output, target) + kl_criterion(output_fp, output)
            optimizer.zero_grad()
            loss.backward()

            '''
            combine quantized update with regularizer
            '''
            for p in model.parameters():

                # if hasattr(p, 'grad') and hasattr(p, 'org'):
                if hasattr(p, 'fp_grad') and hasattr(p, 'org'):
                    # rg_grad = c.gamma * pert
                    if c.REGULARIZATION == 'Fisher':
                        rg_grad = c.gamma * p.fp_grad * p.fp_grad * p.perturbation
                        p.grad.copy_(p.grad + rg_grad.clamp_(-.1,.1))
                    else:
                        pass
                if p.grad is not None:
                    p.grad.copy_(p.grad)

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
            config.n_iters += 1

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

    #compute the best epoch (ie TOTAL best, best_ste+best_fisher)
    best_epoch_total = np.argwhere(valid_acc == best_acc) + 1 + best_epoch

    best_epoch_fisher = np.argwhere(valid_acc == best_acc) + 1

    print('-----------------F-I-S-H-E-R---------------------')
    print('-------------FINISHED FISHER TRAINING------------')
    print('-------------Results for Training END------------')
    print('End Validation Acc %.3f | End Epoch %d' % (valid_acc[c.n_fisher_epochs - 1],  n_tot_epochs))
    print('End TEST Acc %.3f | End Epoch %d\n' % (test_acc[c.n_fisher_epochs - 1], n_tot_epochs))

    print('------------------BEST FISHER--------------------')
    print('--------------Results for BEST EPOCH-------------')
    print('Best Validation Acc: %.3f | At Epoch: %d' % (valid_acc[best_epoch_fisher - 1], best_epoch_total))
    print('Test Acc at Best Valid Epoch: %.3f | Model From Epoch: %d' % (test_acc[best_epoch_fisher - 1], best_epoch_total))
    print('Best FISHER Epoch %d' % (best_epoch_fisher ))


    #TODO: Save MSQE vs Iter, fisher loss vs iter, and
    fisher_testacc = max(test_acc[c.n_fisher_epochs - 1], test_acc[best_epoch_fisher - 1])
    return ste_testacc, fisher_testacc

if c.TRAIN_FROM_SCRATCH:
    # MODEL_SAVEPATH = './checkpoints/'+c.model_name+'_pert_clampedSTEandGrads/'+c.dataset+'/checkpoint.pth'
    MODEL_SAVEPATH = './checkpoints/'+c.model_name+'/'+c.dataset+'_2/checkpoint.pth'
    DATA_SAVEPATH = './checkpoints/'+c.model_name+'_2/training'
    # MODEL_SAVEPATH = './checkpoints/tmp/checkpoint.pth'
    '''
    Setup Summary Writer
    '''
    writer = SummaryWriter(log_dir=DATA_SAVEPATH)
    print('SAVING TO MODEL FILEPATH: ' + MODEL_SAVEPATH)
    train_from_scratch(c, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH)


if c.REGULARIZATION is not None:
    MODEL_SAVEPATH = './checkpoints/'+c.model_name+'/'+c.dataset+'_2/checkpoint.pth'
    print('Loading From:' + MODEL_SAVEPATH)
    DATA_SAVEPATH = './checkpoints/'+c.model_name+'_2/regularization'
    # MODEL_SAVEPATH = './checkpoints/'+c.model_name+'_gold/'+c.dataset+'/checkpoint.pth'
    # MODEL_SAVEPATH = './checkpoints/tmp/checkpoint.pth'
    '''
    Setup Summary Writer
    '''
    writer = SummaryWriter(log_dir=DATA_SAVEPATH)
    ste_testacc, fisher_testacc = train_fisher(c, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH, load_model = True)

print('STE ACCURACY:\t %.3f', ste_testacc)
print('FISHER+STE ACCURACY:\t %.3f', fisher_testacc)

plt.show()

