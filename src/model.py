from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
from torch.autograd import Variable, Function
from layers import GaussianDropout, ASC

class HyperspecAE(nn.Module):
    def __init__(self, num_bands: int=156, end_members: int=3, dropout: float=1.0,
                 activation: str='ReLU', threshold: int=5, ae_type: str='deep'):
      # Constructor
      super(HyperspecAE, self).__init__()

      if activation == 'ReLU':
        self.act = nn.ReLU()
      elif activation == 'LReLU':
        self.act = nn.LeakyReLU()
      else:
        self.act = nn.Sigmoid()

      self.gauss = GaussianDropout(dropout)
      self.asc = ASC()
      kernel_size = 3
      self.conv_enc = nn.Sequential(OrderedDict([
        ('conv_hidden_1', nn.Conv2d(num_bands, 9*end_members, kernel_size)),
        ('activation_1', self.act),
        ('conv_hidden_2', nn.Conv2d(9*end_members, 6*end_members, kernel_size)),
        ('activation_2', self.act),
        ('conv_hidden_3', nn.Conv2d(6*end_members, 3*end_members, kernel_size)),
        ('activation_3', self.act),
        ('conv_hidden_4', nn.Conv2d(3*end_members, end_members, kernel_size)),
        ('activation_4', self.act),
        ('batch_norm', nn.BatchNorm2d(end_members)),
        ('soft_thresholding', nn.Softplus(threshold=threshold)),
        ('ASC', self.asc),
        ('Gaussian_Dropout', self.gauss)
      ]))
      self.conv_dec = nn.Sequential(OrderedDict([
        ('conv_out_hidden_1', nn.Conv2d(end_members, num_bands, kernel_size))
      ]))
      
    def forward(self, img):
      encoded = self.conv_enc(img)
      decoded = self.conv_dec(encoded)
      return encoded, decoded