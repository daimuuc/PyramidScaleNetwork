# -*- coding: <encoding name> -*-
"""
utils
"""
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import scipy.misc
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import h5py
import torch
import random
import matplotlib.pyplot as plt
import math
from matplotlib import cm as CM
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

# %matplotlib inline

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


################################################################################
# change the learning rate according to epoch.
################################################################################
def adjust_learning_rate(optimizer, epoch):
    if (epoch + 1) % 100 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5


################################################################################
# set the random seed.
################################################################################
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)