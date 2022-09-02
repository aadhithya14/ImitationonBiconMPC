## This is a demo for trot motion in mpc
## Author : Avadesh Meduri & Paarth Shah
## Date : 21/04/2021

import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(41, 512),      
      nn.Dropout(p=0.5),
      nn.SiLU(),
      nn.Linear(512, 512),
      nn.Dropout(p=0.5),
      nn.SiLU(),
      nn.Linear(512, 512),
      nn.Dropout(p=0.5),
      nn.SiLU(),
      nn.Linear(512, 12)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)