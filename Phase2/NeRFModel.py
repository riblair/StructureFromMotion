import torch
import torch.nn as nn
import numpy as np


class NeRFmodel(nn.Module):
    def __init__(self, embed_pos_L, embed_direction_L):
        super(NeRFmodel, self).__init__()
        #############################
        # network initialization
        #############################

        self.LL1 = nn.Linear(60,256)
        self.relu1 = nn.ReLU()

        self.LL2 = nn.Linear(256,256)
        self.relu2 = nn.ReLU()

        self.LL3 = nn.Linear(256,256)
        self.relu3 = nn.ReLU()

        self.LL4 = nn.Linear(256,256)
        self.relu4 = nn.ReLU()

        self.LL5 = nn.Linear(256,256)
        self.relu5 = nn.ReLU()

        self.LL6 = nn.Linear(316,256) # 316 from 256 + 60
        self.relu6 = nn.ReLU()

        self.LL7 = nn.Linear(256,256)
        self.relu7 = nn.ReLU()

        self.LL8 = nn.Linear(256,256)
        self.relu8 = nn.ReLU()

        self.LL9 = nn.Linear(256,256)

        self.LL10 = nn.Linear(280,128)
        self.relu10 = nn.ReLU()

        self.LL11 = nn.Linear(128,3)

    def position_encoding(self, x, L):
        #############################
        # Implement position encoding here
        #############################

        return y

    def forward(self, pos, direction):
        #############################
        # network structure
        #############################

        return output
