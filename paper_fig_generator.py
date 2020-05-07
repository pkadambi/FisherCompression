'''

The purpose of this file is to generate figures for the paper

'''
from plot_2D import plot_2d_contour

'''

Figure 1: Test Loss figures


'''
import h5py
import numpy as np

def load_surf_data(surf_file, surf_name='train_loss'):
    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    return X, Y, Z

# fp      = './SavedModels/cifar10/Resnet18/4ba_4bw/'
buffer4b= './SavedModels/cifar10/Resnet18/4ba_4bw_buffer/Run0/'
fisher  = './SavedModels/cifar10/Resnet18/4ba_4bw/fisher_buffer/Run1/'
msqe    = './SavedModels/cifar10/Resnet18/4ba_4bw/msqe_buffer/Run0/'
distil1  = './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_teq1/Run0/'
distil2  = './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_teq2/Run0/'
distil3  = './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_teq3/Run0/'
distil4  = './SavedModels/cifar10/Resnet18/4ba_4bw/distillation_buffer/Run0/'

files = [buffer4b, fisher, msqe, distil1, distil2, distil3, distil4]
descriptions = ['4b Quantized', 'Fisher Regularized', 'MSQE', 'Distillation T=1', 'Distillation T=2', 'Distillation T=3', 'Distillation T=4']
results_file = 'resnet_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,50]x[-1.0,1.0,50].h5'

X = {}
Y = {}
Z = {}

for file, description in zip(files, descriptions):
    X[description], Y[description], Z[description] = load_surf_data(file+results_file)



'''

Figure 2 : Train Loss figure

'''





'''

Figure 2b. 

'''