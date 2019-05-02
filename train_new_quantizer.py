from data import get_dataset
import torch.nn as nn
import torch.backends.cudnn as cudnn
from preprocess import get_transform
import models
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.model_utils import *
from utils.dataset_utils import *
from utils.visualizaiton_utils import *
import os
import time
from tensorboardX import SummaryWriter

cudnn.benchmark = True

torch.cuda.set_device(0)

# -----------------------------------------------------------------------------------------------------------------------
#
#                           CONFIGURATION
#
# c = LenetFashionMNISTConfig(n_epochs=10, REGULARIZATION='Fisher', n_fisher_epochs=1, TRAIN_FROM_SCRATCH=True, gamma=.1)
# c = LenetFashionMNISTConfig(n_epochs=50, REGULARIZATION=None, n_fisher_epochs=1, TRAIN_FROM_SCRATCH=True, gamma=.1)
# c.print_interval = 50

# c = ResnetConfig(n_epochs=200, dataset='cifar10', REGULARIZATION=None, TRAIN_FROM_SCRATCH=True, n_fisher_epochs=0)
c = ResnetConfig(n_epochs=200, dataset='cifar10', REGULARIZATION='KL', TRAIN_FROM_SCRATCH=True, n_regularized_epochs=50,
                 gamma=.01)
c.print_interval = 25
c.model_name = 'resnet_quantized_float_bn'
#
# -----------------------------------------------------------------------------------------------------------------------

# print(models.__dict__)

model = models.__dict__[c.model_name]
model_config = {'input_size': c.input_size, 'dataset': c.dataset}
MODEL_SAVEPATH = c.model_savepath + 'checkpoint.pth'
model = model(**model_config)
# model.apply(xavier_initialize_weights)
# model.apply(normal_initialize_biases)
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-1)
criterion = nn.CrossEntropyLoss()
kl_criterion = nn.KLDivLoss()
logsoftmax = nn.LogSoftmax()
smfx = nn.Softmax()
if 'cifar' in c.dataset:
    AUGMENT_TRAIN = True

transforms = {'train': get_transform(name=c.dataset, augment=c.AUGMENT_TRAIN),

              'test': get_transform(name=c.dataset, augment=False)}

train_data = get_dataset(c.dataset, 'train', transform=transforms['train'])
train_data, valid_data = generate_validation_split(train_data, validation_split=.05)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=c.batch_size, shuffle=True,
    num_workers=c.workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=c.batch_size, shuffle=True,
    num_workers=c.workers, pin_memory=True)

test_data = get_dataset(c.dataset, 'test', transform=transforms['test'])
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=c.batch_size, shuffle=True,
    num_workers=c.workers, pin_memory=True)

record_interval = 10

def train_from_scratch(config, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH,
                       record_interval=10):
    c = config
    lossval = np.array([])
    valid_acc = np.array([])
    test_acc = np.array([])
    tr_acc = np.array([])

    # Clear the logfile
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

            optimizer.step()


            # for m in model.modules():
            #     # print(m)
            #     if hasattr(m, 'qweight'):
            #         print(m.qweight)
            # exit()
            # # for n, p in model.named_parameters():
            # #     print(n)
            # #     if hasattr(p, 'qweight'):
            # #         print(p.qweight)
            # # exit()
            # if iter == 5:
            #     exit()

            if iter % c.print_interval == 0:
                logstr = 'Epoch %d | Iters: %d | Train Loss %.5f | Acc %.3f' % (
                    epoch + 1, config.n_iters, lossval_, accuracy_)
                print(logstr)
                logstr += '\n'
                logfile.write(logstr)

            if iter % record_interval == 0:
                lossval = np.hstack([lossval, lossval_])
                tr_acc = np.hstack([tr_acc, accuracy_])

            config.n_iters += 1

        val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)

        # Save model if validation acc has improved
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
    best_epoch = np.argwhere(valid_acc == best_acc) + 1
    print('--------------------------------------------------')
    print('-------------FINISHED STE TRAINING----------------')
    print('-------------Results for Training END-------------')
    print('End Validation Acc %.3f | End Epoch %d' % (valid_acc[c.n_epochs - 1], c.n_epochs))
    print('End TEST Acc %.3f | End Epoch %d\n' % (test_acc[c.n_epochs - 1], c.n_epochs))

    print('--------------Results for BEST EPOCH--------------')
    print('Best Validation Acc: %.3f | At Epoch: %d' % (valid_acc[best_epoch - 1], best_epoch))
    print('Test Acc at Best Valid Epoch: %.3f | Model From Epoch: %d' % (test_acc[best_epoch - 1], c.n_epochs))

    # TODO: Save weight pdf into results file (weight pdf from optimal iteration, AND from end of training)
    x = np.arange(len(lossval) * record_interval)
    two_scale_plot(lossval, tr_acc, y1_label='Train Loss', y2_label='Train Acc')

    training_loss = lossval
    training_accuracy = tr_acc

    return training_loss, training_accuracy, best_epoch


'''

Plotting Utilities

'''


def train_fisher(config, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH, load_model=False):
    c = config
    lossval = np.array([])
    valid_acc = np.array([])
    test_acc = np.array([])
    tr_acc = np.array([])

    if load_model:
        checkpoint = torch.load(MODEL_SAVEPATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        test_loss, ste_testacc = test_model(test_loader, model, criterion, printing=True)
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
        print('#  START REGULARIZER TRAINING #')
        print('#                             #')
        print('###############################')

    else:
        best_epoch = 0
    n_tot_epochs = best_epoch + c.n_regularized_epochs

    # plot loss and accuracy curves
    tr_start = time.time()
    for epoch in range(best_epoch, n_tot_epochs):
        regime = getattr(model, 'regime')

        if 'lenet' not in c.model_name:
            optimizer = adjust_optimizer(optimizer, epoch, regime)

        print(optimizer)
        ep_start = time.time()
        for iter, (inputs, target) in enumerate(train_loader):
            inputs = inputs.cuda()
            target = target.cuda()

            '''
            set fp wts
            NOTE: MUST SET model.eval() before FP pass in 
            order to stop batch norm from updating with FP parameters

            Else: Test/Validation accuracy will tank!!!
            '''
            model.eval()  # MUST SET model.eval() DURING FP PASS!!!
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.perturbation = p.org - p.data
                    p.data.copy_(p.org)
                    p.binary_pass = False

            '''
            fp_pass
            '''
            output_fp = model(inputs)
            optimizer.zero_grad()
            loss = criterion(output_fp, target)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            '''
            quantized pass
            '''
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'grad'):
                    if p.grad is not None:
                        p.fp_grad = p.grad.clone()
                        # print(p.fp_grad)
                if hasattr(p, 'binary_pass'):
                    p.binary_pass = True

            model.train()  # DO NOT DELETE THIS LINE!!!

            output = model(inputs)

            outfp_logprob = logsoftmax(output_fp)
            out_logprob = logsoftmax(output)

            outfp_s = smfx(output_fp)
            out_s = smfx(output)

            # Note on kl_loss(output, target) - output has to be log probs, targets have to be probs
            ce_loss_y_fq = criterion(output, target)
            kl_loss = kl_criterion(outfp_logprob, out_s)

            '''
            combine quantized update with regularizer
            '''
            if 'KL' in c.REGULARIZATION:
                loss = ce_loss_y_fq + c.gamma * kl_loss
                # loss = ce_loss_y_fq
                reg_loss = c.gamma * kl_loss
            else:
                loss = ce_loss_y_fq

            optimizer.zero_grad()
            loss.backward()

            fisherpert = None
            pertub = None

            for p in model.parameters():
                # if hasattr(p, 'grad') and hasattr(p, 'org'):
                if hasattr(p, 'fp_grad') and hasattr(p, 'org'):
                    pert_ = p.org - p.data
                    pert_sq = pert_ * pert_

                    if fisherpert is None:
                        fisherpert = torch.sum(p.fp_grad * p.fp_grad * pert_sq)
                    else:
                        fisherpert = fisherpert + torch.sum(p.fp_grad * p.fp_grad * pert_sq)
                    # pert_ = p.data - p.org
                    # pert_sq = pert_ * pert_
                    if pertub is None:
                        pertub = torch.sum(pert_sq)
                    else:
                        pertub = pertub + torch.sum(pert_sq)
                    # rg_grad = c.gamma * pert

                    if c.REGULARIZATION is None:
                        pass
                    elif 'Fisher' in c.REGULARIZATION and 'L2' in c.REGULARIZATION:
                        # rg_grad = c.gamma * 2 * p.fp_grad * p.fp_grad * p.perturbation.clamp(-.1,.1)
                        # rg_grad = c.gamma * 2 * p.grad * p.grad* p.perturbation.clamp(-.1,.1)
                        rg_grad = c.gamma * 2 * (p.fp_grad * p.fp_grad * pert_ + .000001 * pert_)
                        p.grad.copy_(p.grad + rg_grad.clamp_(-.1, .1))

                    elif 'Fisher' in c.REGULARIZATION:
                        rg_grad = c.gamma * 2 * p.fp_grad * p.fp_grad * pert_
                        p.grad.copy_(p.grad + rg_grad.clamp_(-.1, .1))

                    elif 'L2' in c.REGULARIZATION:
                        rg_grad = c.gamma * 2 * p.perturbation
                        p.grad.copy_(p.grad + rg_grad.clamp_(-.1, .1))
                    else:
                        pass
                if p.grad is not None:
                    p.grad.copy_(p.grad)

            reg_loss_ = 0
            if c.REGULARIZATION is None:
                reg_loss_ = 0
            if 'KL' in c.REGULARIZATION:
                reg_loss_ += reg_loss.item()
            if 'L2' in c.REGULARIZATION:
                reg_loss_ += pertub
            if 'Fisher' in c.REGULARIZATION:
                reg_loss_ += fisherpert

            if config.n_iters % config.record_interval == 0 and config.n_iters > 0:
                for name, p in list(model.named_parameters()):
                    if hasattr(p, 'org'):
                        if config.n_iters % config.record_interval == 0:
                            pert_ = p.data - p.org

                            writer.add_histogram(name + ' perturbation', pert_.clone().cpu().data.numpy(),
                                                 config.n_iters)
                            writer.add_histogram(name + ' FP', p.org.clone().cpu().data.numpy(), config.n_iters)
                            writer.add_histogram(name + ' Quant', p.data.clone().cpu().data.numpy(), config.n_iters)

                writer.add_scalar('metrics/kl_loss', kl_loss.item(), config.n_iters)
                writer.add_scalar('metrics/MSQE', pertub.item(), config.n_iters)

                # if 'Fisher' in c.REGULARIZATION:
                writer.add_scalar('metrics/FisherRegularizer', fisherpert.item(), config.n_iters)

                writer.add_scalar('loss/H(y, f_q)_STE_Loss:', loss.item(), config.n_iters)
                writer.add_scalar('loss/H(y, f_q)_STE_Loss:', loss.item(), config.n_iters)
                writer.add_scalar('loss/H(y, f_q)_STE_Loss:', loss.item(), config.n_iters)

                if config.REGULARIZATION == 'KL':
                    writer.add_scalars('used_losses/',
                                       {'H(y, f_q)_STE_Loss:': loss.item(), 'Regularizer Loss:': reg_loss.item()},
                                       config.n_iters)

            '''
            Gradient Clipping
            '''
            for p in model.parameters():

                # if hasattr(p, 'grad') and hasattr(p, 'org'):
                # if hasattr(p, 'fp_grad') and hasattr(p, 'org'):
                # rg_grad = c.gamma * pert
                # print('cliped')
                if p.grad is not None:
                    p.grad.copy_(p.grad.clamp_(-.1, .1))

            accuracy_ = accuracy(output, target)[0].item()
            lossval_ = loss.item()

            '''
            Restoring fp before updating weights
            '''
            for name, p in list(model.named_parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
            '''
            Data Logging
            '''

            optimizer.step()

            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))
            elapsed = time.time() - ep_start
            total_elapsed = time.time() - tr_start
            if iter % c.print_interval == 0 and iter > 0:
                print(pertub)
                print(fisherpert)
                print('Epoch %d | Iters: %d | Train Loss %.4f | %s Reg Loss %.4f|  Acc %.2f| Elapsed Time %1f' % (
                epoch + 1, config.n_iters, lossval_, c.REGULARIZATION, reg_loss_, accuracy_, total_elapsed))

            if iter % record_interval == 0:
                lossval = np.hstack([lossval, lossval_])
                tr_acc = np.hstack([tr_acc, accuracy_])

            config.n_iters += 1

        '''
        Do a forward pass to set the weights to their binary values before going to calculate validation acc
        '''

        val_loss, val_acc = test_model(valid_loader, model, criterion, printing=False)
        test_loss, test_acc_ = test_model(test_loader, model, criterion, printing=False)

        valid_acc = np.hstack([valid_acc, val_acc])
        test_acc = np.hstack([test_acc, test_acc_])

        REG_SAVEPATH = MODEL_SAVEPATH[:-4] + c.REGULARIZATION + '.pth'
        # REG_SAVEPATH = c.model_savepath + 'checkpoint_{}.pth'.format(c.REGULARIZATION)
        print('VALID ACC\n')
        print(valid_acc)
        if len(valid_acc) >= 1:
            if val_acc >= np.max(valid_acc):
                print('Found new best model! Saving model from this epoch.')
                if not os.path.exists(c.model_savepath):
                    os.makedirs(c.model_savepath)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, REG_SAVEPATH)

        ep_time = time.time() - ep_start
        print('\nEpoch %d | Valid Loss %.3f | Valid Acc %.2f | Epoch Time %.1f \n' % (
        epoch + 1, val_loss, val_acc, ep_time))
        print('***********************************************\n')
        # exit()
    best_acc = np.max(valid_acc)

    # compute the best epoch (ie TOTAL best, best_ste+best_fisher)
    best_epoch_total = np.argwhere(valid_acc == best_acc) + 1 + best_epoch

    best_epoch_fisher = np.argwhere(valid_acc == best_acc) + 1

    # print('-----------------F-I-S-H-E-R---------------------')
    print('-------------FINISHED REGULARIZER TRAINING--------')
    print('-------------Results for Training END------------')
    print('End Validation Acc %.3f | End Epoch %d' % (valid_acc[c.n_regularized_epochs - 1], n_tot_epochs))
    print('End TEST Acc %.3f | End Epoch %d\n' % (test_acc[c.n_regularized_epochs - 1], n_tot_epochs))

    print('------------------BEST REGULARIZED---------------')
    print('--------------Results for BEST EPOCH-------------')

    print('Best Validation Acc: %.3f | At Epoch: %d' % (valid_acc[best_epoch_fisher - 1], best_epoch_total))
    print('Test Acc at Best Valid Epoch: %.3f | Model From Epoch: %d' % (
    test_acc[best_epoch_fisher - 1], best_epoch_total))
    print('Best FISHER Epoch %d' % (best_epoch_fisher))
    print('USED REGULARIZATION SCHEME: ' + str(c.REGULARIZATION))

    # TODO: Save MSQE vs Iter, fisher loss vs iter, and
    fisher_testacc = max(test_acc[c.n_regularized_epochs - 1], test_acc[best_epoch_fisher - 1])
    return ste_testacc, fisher_testacc


if c.TRAIN_FROM_SCRATCH and c.n_epochs > 0:
    # MODEL_SAVEPATH = './checkpoints/'+c.model_name+'_pert_clampedSTEandGrads/'+c.dataset+'/checkpoint.pth'
    # MODEL_SAVEPATH = './checkpoints/'+c.model_name+'/'+c.dataset+'_4/checkpoint.pth'
    # DATA_SAVEPATH = './checkpoints/'+c.model_name+'_4/training'
    DATA_SAVEPATH = './checkpoints/tmp/training'
    MODEL_SAVEPATH = './checkpoints/tmp/checkpoint.pth'
    '''
    Setup Summary Writer
    '''
    writer = SummaryWriter(log_dir=DATA_SAVEPATH)
    print('SAVING TO MODEL FILEPATH: ' + MODEL_SAVEPATH)
    train_from_scratch(c, model, optimizer, train_loader, test_loader, valid_loader, MODEL_SAVEPATH)

if c.REGULARIZATION is not None:
    # MODEL_SAVEPATH = './checkpoints/'+c.model_name+'/'+c.dataset+'_4/checkpoint.pth'
    DATA_SAVEPATH = './checkpoints/' + c.model_name + '/regularization_' + c.REGULARIZATION
    MODEL_SAVEPATH = './checkpoints/' + c.model_name + '/' + c.dataset + '/checkpoint.pth'
    # DATA_SAVEPATH = './checkpoints/tmp/regularization'
    # MODEL_SAVEPATH = './checkpoints/tmp/cifar10/checkpoint.pth'
    print('Loading From:' + MODEL_SAVEPATH)
    '''
    Setup Summary Writer
    '''
    writer = SummaryWriter(log_dir=DATA_SAVEPATH)
    ste_testacc, regularized_testacc = train_fisher(c, model, optimizer, train_loader, test_loader, valid_loader,
                                                    MODEL_SAVEPATH, load_model=True)

print('STE TEST ACCURACY:\t %.3f', ste_testacc)
print('REGULARIZED TEST ACCURACY:\t %.3f', regularized_testacc)

plt.show()

