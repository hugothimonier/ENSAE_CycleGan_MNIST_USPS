import os
import pdb
import argparse
import pickle as pkl

from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from six.moves.urllib.request import urlretrieve
import tarfile
import pickle
import sys
import scipy
import scipy.misc

import cycle_model
import solver
import utilities

SEED = 5

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


args = AttrDict()
args_dict = { 'nc':1,
              'image_size':32, 
              'g_conv_dim':32, 
              'd_conv_dim':32,
              'init_zero_weights': False,
              'num_workers': 0,
              'train_iters':1000,
              'X':'MNIST',
              'Y':'USPS',
              'lambda_cycle': 0.045,
              'lr':0.0002,
              'beta1':0.5,
              'beta2':0.999,
              'batch_size':32, 
              'checkpoint_dir': 'checkpoints_cyclegan',
              'sample_dir': 'samples_cyclegan',
              'load': None,
              'log_step':200,
              'sample_every':200,
              'checkpoint_every':500,
}
args.update(args_dict)


print_opts(args)
G_XtoY, G_YtoX, D_X, D_Y = train(args)