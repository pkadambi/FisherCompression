import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def histplot_from_tensor(tensor, bins=10, log=False):
    tmp = tensor.detach().cpu().numpy().ravel()
    plt.hist(tmp, bins=bins, log=log)
    plt.show()





