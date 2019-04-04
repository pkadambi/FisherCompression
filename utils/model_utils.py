import torch
import torchvision

import sklearn.cluster.k_means
from sklearn.cluster import k_means
import numpy as np

def quantizer_levels_from_wts(model, n_levels):
    """
    This function takes in a model and tries to find the levels of the quantizer by performing k-means on the weights
    :param model: trained pytorch model
    :param n_levels: number of levels in the quantizer
    :return: k-means scipy object that has the levels of the quantizer
    """
    params = []
    for p in model.parameters():
        p_arr = p.cpu().numpy()
        params.append(p_arr.ravel())

    params = np.hstack(params)

    return k_means(params.reshape(-1, 1), n_levels)






