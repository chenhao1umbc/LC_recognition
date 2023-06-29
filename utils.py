#%% loading dependency
import os
import h5py 
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
from datetime import datetime
import torchvision.datasets as datasets 
Tensor = torch.Tensor

# from torch.utils.tensorboard import SummaryWriter
"make the result reproducible"
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')


#%%
# KL(q(x)||p(x)) where p(x) is Gaussian, q(x) is Gaussian
def KLD_gauss(mu,logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

# KL(q(x)||p(x)) where p(x) is Laplace, q(x) is Gaussian
def KLD_laplace(mu,logvar,scale=1.0):
    v = logvar.exp()
    y = mu/torch.sqrt(2*v)
    y2 = y.pow(2)
    t1 = -2*torch.exp(-y2)*torch.sqrt(2*v/np.pi)
    t2 = -2*mu*torch.erf(y)
    t3 = scale*torch.log(np.pi*v/(2.0*scale*scale))
    temp = scale+t1+t2+t3
    KLD = -1.0/(2*scale)*torch.sum(1+t1+t2+t3)
    return KLD