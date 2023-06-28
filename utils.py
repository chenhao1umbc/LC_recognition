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