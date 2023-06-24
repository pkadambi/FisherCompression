import matplotlib.pyplot as plt
import numpy as np

levels = np.arange(.3, 5.5, .1)

results_dir = './results/grid_x_-.75:.75:30_y-.75:.75:30_test'
model_names = ['distillation_teq1', 'distillation_teq2', 'distillation_teq3', 'distillation_teq4', 'STE', 'msqe', 'fisher']
labels = ['Distil T=1', 'Distil T=2', 'Distil T=3', 'Distil T=4', 'STE', 'MSQE', 'Fisher']


# Choose the run numbers that show the paper story the best
run_numbers = [7, 7, 7, 7, 9, 9, 9]


def parse_results(results_dir, n_runs, model_names, labels, run_numbers = None, plot=True):
    landscapes = []

    data_dirs = results_dir + '/%s/%s.txt'
    X = np.loadtxt(results_dir + '/X.txt', delimiter=',')
    Y = np.loadtxt(results_dir + '/Y.txt', delimiter=',')


    if run_numbers is None:
        for i in range(n_runs):

            for j in range(len(model_names)):
                lossdata = np.loadtxt(data_dirs % ( str(i), str(model_names)), delimiter=',')
                titlestr = labels[j] + ' Run ' + str(i)
                if plot:
                    plt.figure()
                    plt.contour(X, Y, lossdata, cmap='summer', levels=np.arange(.1, 9., .1))
                    plt.grid()
                    plt.title(titlestr)
    else:

        for j in range(len(model_names)):
                _runnumber = run_numbers[j]
                titlestr = labels[j]
                lossdata = np.loadtxt(data_dirs % (_runnumber, str(model_names[j])), delimiter=',')

                # plt.contour(X, Y, lossdata, cmap='summer', levels=np.arange(.1, 9., .1))

                landscapes.append(lossdata)

                if plot:
                    plt.figure()

                    CS = plt.contour(X, Y, lossdata, cmap='summer', levels=np.arange(.1, 9., .1))

                    plt.clabel(CS, inline=1, fontsize=12)
                    plt.ylim((-.6, .6))
                    plt.xlim((-.6, .6))

                    plt.grid()
                    plt.title(titlestr)

    return landscapes, X, Y

n_runs = 10
landscapes, X, Y = parse_results(results_dir, n_runs, model_names, labels, run_numbers, plot=False)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# f, axarr = plt.subplots(1, 4, gridspec_kw = {'wspace':0, 'hspace':0})
titles = ['T=1, 93.4% Test Acc', 'T=2, 93.78% Test Acc', 'T=3, 94.05% Test Acc', 'T=4, 94.1% Test Acc']
plt.figure()
fig = plt.figure(constrained_layout=True)
spec1 = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)

for i in range(4):
    ax = fig.add_subplot(spec1[0, i])
    CS = ax.contour(X, Y, landscapes[i], cmap='summer', levels=np.arange(.1, 6., .1), bbox_inches='tight')
    ax.grid('on')
    ax.clabel(CS, inline=1, fontsize=13)
    ax.set_xlim((-.6,.6))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.grid('on')
    ax.set_title(titles[i], fontdict={'fontsize':15})


# titles = ['T=1, 93.4% Test Acc', 'T=2, 93.78% Test Acc', 'T=3, 94.05% Test Acc', 'T=4, 94.1% Test Acc']

landscapes, X, Y = parse_results(results_dir, n_runs, model_names, labels, run_numbers, plot=False)

titles = ['T=1, 93.4% Test Acc', 'T=2, 93.78% Test Acc', 'T=3, 94.05% Test Acc', 'T=4, 94.2% Test Acc', 'STE, 93.44% Test Acc', 'Fisher, 93.4% Test Acc', 'MSQE, 93.4% Test Acc']

plt.figure()
fig = plt.figure(constrained_layout=True)
spec2 = gridspec.GridSpec(ncols=7, nrows=1, figure=fig)

for i in range(7):
    ax = fig.add_subplot(spec2[0, i])
    CS = ax.contour(X, Y, landscapes[i], cmap='summer', levels=np.arange(.1, 6., .1), bbox_inches='tight')
    ax.clabel(CS, inline=1, fontsize=13)
    ax.set_xlim((-.6,.6))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.grid('on')

    # ax.set_yticklabels((-.6,.6))
    ax.set_title(titles[i], fontdict={'fontsize':15})

plt.figure()
fig = plt.figure(constrained_layout=True)
spec3 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

k=0
for i in range(4,7):
    ax = fig.add_subplot(spec3[0, k])
    CS = ax.contour(X, Y, landscapes[i], cmap='summer', levels=np.arange(.1, 6., .1), bbox_inches='tight')
    ax.grid('on')
    ax.clabel(CS, inline=1, fontsize=13)
    ax.set_xlim((-.6,.6))
    # ax.set_yticklabels((-.6,.6))
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.grid('on')

    ax.set_title(titles[i], fontdict={'fontsize':15})
    k+=1


# plt.figure()
fig4 = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(ncols=6, nrows=6, figure=fig4)


# fig4 = plt.subplots(ncols=6, nrows=2)
# fig4 = plt.subplots(constrained_layout=True)
# gs = fig4.add_gridspec(2, 6)

# for ax in axs[1:, -1]:
#     ax.remove()

# axbig = fig4.add_subplot(gs[1:, -1])
# axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5), xycoords='axes fraction', va='center')
fig4.tight_layout()

figteq1 = fig4.add_subplot(gs[0, 0])
CS = figteq1.contour(X, Y, landscapes[0], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figteq1.clabel(CS, inline=1, fontsize=7)
figteq1.set_xlim((-.6,.6))
figteq1.get_yaxis().set_visible(False)
figteq1.get_xaxis().set_visible(False)
figteq1.grid('on')
figteq1.set_title('(a) T=1')

figteq2 = fig4.add_subplot(gs[0, 1])
CS = figteq2.contour(X, Y, landscapes[1], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figteq2.clabel(CS, inline=1, fontsize=7)
figteq2.set_xlim((-.6,.6))
figteq2.get_yaxis().set_visible(False)
figteq2.get_xaxis().set_visible(False)
figteq2.grid('on')
figteq2.set_title('(b) T=2')

figteq3 = fig4.add_subplot(gs[0, 2])
CS = figteq3.contour(X, Y, landscapes[2], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figteq3.clabel(CS, inline=1, fontsize=7)
figteq3.set_xlim((-.6,.6))
figteq3.get_yaxis().set_visible(False)
figteq3.get_xaxis().set_visible(False)
figteq3.grid('on')
figteq3.set_title('(c) T=3')

figteq4 = fig4.add_subplot(gs[0, 3])
CS = figteq4.contour(X, Y, landscapes[3], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figteq4.clabel(CS, inline=1, fontsize=7)
figteq4.set_xlim((-.6,.6))
figteq4.get_yaxis().set_visible(False)
figteq4.get_xaxis().set_visible(False)
figteq4.grid('on')
figteq4.set_title('(d) T=4')

figste      = fig4.add_subplot(gs[1, 0])
CS = figste.contour(X, Y, landscapes[4], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figste.clabel(CS, inline=1, fontsize=7)
figste.set_xlim((-.6,.6))
figste.get_yaxis().set_visible(False)
figste.get_xaxis().set_visible(False)
figste.grid('on')
figste.set_title('(e) STE')

figfisher   = fig4.add_subplot(gs[1, 1])
CS = figfisher.contour(X, Y, landscapes[5], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figfisher.clabel(CS, inline=1, fontsize=7)
figfisher.set_xlim((-.6,.6))
figfisher.get_yaxis().set_visible(False)
figfisher.get_xaxis().set_visible(False)
figfisher.grid('on')
figfisher.set_title('(f) MSQE')

import matplotlib.image as mpimg
img = mpimg.imread('./flat_v_acc_trace.jpg')

figmsqe     = fig4.add_subplot(gs[1, 2])
CS = figmsqe.contour(X, Y, landscapes[5], cmap='summer', levels=np.arange(.1, 6., .1), )#bbox_inches='tight')
figfisher.clabel(CS, inline=1, fontsize=7)
figfisher.set_xlim((-.6,.6))
figmsqe.get_yaxis().set_visible(False)
figmsqe.get_xaxis().set_visible(False)
figmsqe.grid('on')
figmsqe.set_title('(g) Fisher')

fig_tvsminimum = fig4.add_subplot(gs[0:,4:])
fig_tvsminimum.imshow(img)
fig_tvsminimum.get_yaxis().set_visible(False)
fig_tvsminimum.get_xaxis().set_visible(False)
fig_tvsminimum.set_title('(h) Loss Flatness vs Acc')


plt.show()

# fig, ax = plt.subplot(1, 4, 1)
# CS = plt.contour(X, Y, landscapes[0], cmap='summer', levels=np.arange(.1, 9., .1), bbox_inches='tight')
# plt.clabel(CS, inline=1, fontsize=12)
# plt.title('T=1, 93.4% Test Acc.')
# plt.ylim((-.6, .6))
# plt.xlim((-.6, .6))
#
# plt.subplot(1, 4, 2)
# CS = plt.contour(X, Y, landscapes[1], cmap='summer', levels=np.arange(.1, 9., .1), bbox_inches='tight')
# plt.clabel(CS, inline=1, fontsize=12)
# plt.title('T=2, 93.78\% Test Acc.')
# plt.ylim((-.6, .6))
# plt.xlim((-.6, .6))
#
# plt.subplot(1, 4, 3)
# CS = plt.contour(X, Y, landscapes[2], cmap='summer', levels=np.arange(.1, 9., .1), bbox_inches='tight')
# plt.clabel(CS, inline=1, fontsize=12)
# plt.title('T=3, 94.05% Test Acc.')
# plt.ylim((-.6, .6))
# plt.xlim((-.6, .6))
#
# plt.subplot(1, 4, 4)
# CS = plt.contour(X, Y, landscapes[3], cmap='summer', levels=np.arange(.1, 9., .1), bbox_inches='tight')
# plt.clabel(CS, inline=1, fontsize=12)
# plt.title('T=4, 94.1% Test Acc.')
# plt.ylim((-.6, .6))
# plt.xlim((-.6, .6))

# plt.show()




